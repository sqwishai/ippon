"""Clef-Flash (Cloudflare; Apache-2.0): Qwen3.5 (9.4B) read once for every question, scored by a joint schema head.

A reimplementation of the checkpoint's joint_schema_model.py (encode_record, JointSchemaHead, systemone), checked
against it:

- one prompt for the whole request: a system turn, "STATE:" and the state (JSON with sorted keys and no spaces
  unless it is text), "SCHEMA FIELDS:" and for every question "FIELD i / ID / TYPE / INSTRUCTION" then each
  allowed option as "OPTION j: " and {"description": ..., "option_id": ...} in that JSON, then the assistant
  turn up to "JOINT SCHEMA DECISIONS:"; every piece is tokenized on its own, and the state is cut to the
  16,384-token window; a noul's options are true then false (with default descriptions), a choice's sorted by
  label, a score's by level;
- a request's images (`Request.images`) right after "STATE:\n", before the state: one
  `<|vision_start|><|image_pad|><|vision_end|>` each, then "\n", tokenized as one piece, each image token expanded
  to its image's rows (`_encode_media`); their rows count toward the window, so the state is cut to what they leave,
  and toward the usage; each image read by the checkpoint's vision tower (adapters/qwen.py) as decoded,
  without an EXIF turn: Clef's code hands the record's images to the processor as they are, and the processor turns
  neither a PIL image (the card's `Image.open`) nor a path (transformers 5.10.2, the card's, and 5.17.0 alike: an
  EXIF-rotated 640 x 480 JPEG stays 40 patches wide);
- the backbone's final hidden states over the whole prompt go to the head: span means of each question's
  instruction and each option, the option tokens' mean output embedding, two evidence-routing layers (options
  attending to the sequence), field vectors through PyTorch decoder layers, and a logit per option from a
  lexical prior, a cosine and a residual scorer; softmax.

Answers take Clef's shape (rounded to 4 decimals; a choice's confidence is its probability). Clef's code needs a
request `model` and a map for a choice's criteria; ippon needs neither (a list of labels is shown as options
without descriptions).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from ..adapter import AdapterBase, Cut, Hidden, LoadContext, Loaded, Rendering
from ..arch import qwen35
from ..arch.ops import activation
from ..errors import ContextTooLong, InvalidRequest, UnsupportedModel
from ..gguf.header import Header
from ..gguf.names import Reads, Rule, qwen35_rules, row
from ..gguf.source import qwen35_differences
from ..media.qwen import QwenImages
from ..model_dir import API_VERSION, ModelDir
from ..plan import Prompt, Readout, one_mark, state_end_marks
from ..quant import Table, rows
from ..schema import Question, Request, rounded, systemone_answer
from ..tokenizer import Tokenizer
from ..weights import Weights
from .qwen import ANSWER_TURN, VISION, QwenImageFormat, chat_head

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
TYPES = {"noul": 0, "choice": 1, "score": 2}
NOUL_DEFAULTS = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}
MAX_LENGTH = 16384
F32 = mx.float32
HEAD_ROOTS = (
    "evidence_layers",
    "field_norm",
    "global_projection",
    "hidden_norm",
    "joint_logit_scale",
    "layers",
    "memory_projection",
    "option_context_projection",
    "option_lexical_projection",
    "option_norm",
    "option_question_projection",
    "option_summary_norm",
    "prior_logit_scale",
    "question_projection",
    "residual_gate",
    "residual_scorer",
    "type_embedding",
)


def render(value: Any) -> str:
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def question_options(q: Question) -> list[tuple[str, Any]]:
    """The allowed options in the model's order, with their descriptions."""
    if q.type == "noul":
        criteria = {**NOUL_DEFAULTS, **dict(q.criteria)}
        return [(key, criteria[key]) for key in ("true", "false")]
    if q.type == "choice":
        return sorted((str(k), v) for k, v in q.criteria)
    return [(str(i), level) for i, (_, level) in enumerate(q.criteria)]


@dataclass(frozen=True)
class Spans:
    question: tuple[int, int]
    options: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]
    kind: int


def _layer_norm(x: mx.array, w: Mapping[str, mx.array], name: str) -> mx.array:
    return mx.fast.layer_norm(x, w[name + ".weight"], w[name + ".bias"], 1e-5)


def _attention(q: mx.array, kv: mx.array, w: Mapping[str, mx.array], name: str, heads: int) -> mx.array:
    """torch.nn.MultiheadAttention(q, kv, kv) without masks: q [n, width], kv [m, width]."""
    width = q.shape[-1]
    wq, wk, wv = mx.split(w[name + ".in_proj_weight"], 3, axis=0)
    bq, bk, bv = mx.split(w[name + ".in_proj_bias"], 3, axis=0)

    def split(x: mx.array) -> mx.array:
        return x.reshape(x.shape[0], heads, width // heads).transpose(1, 0, 2)[None]

    o = mx.fast.scaled_dot_product_attention(
        split(q @ wq.T + bq), split(kv @ wk.T + bk), split(kv @ wv.T + bv), scale=(width // heads) ** -0.5
    )
    return (
        o[0].transpose(1, 0, 2).reshape(q.shape[0], width) @ w[name + ".out_proj.weight"].T + w[name + ".out_proj.bias"]
    )


def _linear(x: mx.array, w: Mapping[str, mx.array], name: str) -> mx.array:
    y = x @ w[name + ".weight"].T
    return y + w[name + ".bias"] if name + ".bias" in w else y


def _gelu(x: mx.array) -> mx.array:
    """Exact GELU, x * Phi(x), in x's dtype: the head's activation (PyTorch's nn.GELU())."""
    return activation(x, "gelu")


def _normalize(x: mx.array) -> mx.array:
    return x / mx.maximum(mx.sqrt((x * x).sum(-1, keepdims=True)), 1e-12)


@dataclass(frozen=True)
class JointSchemaHead:
    """Clef's JointSchemaHead in fp32 (joint_head.safetensors, joint_head_config.json)."""

    w: Mapping[str, mx.array]
    heads: int
    routing_layers: int
    layers: int

    @classmethod
    def load(cls, weights: Weights, config: Mapping[str, Any]) -> JointSchemaHead:
        names = [name for name in weights.tensors if name.partition(".")[0] in HEAD_ROOTS]
        if any(not mx.issubdtype(weights.tensors[name].dtype, mx.floating) for name in names):
            raise UnsupportedModel("Clef's joint head must retain its published floating-point weights")
        w = {name: weights.get(name, F32) for name in names}
        mx.eval(list(w.values()))
        return cls(w, int(config["heads"]), int(config["routing_layers"]), int(config["layers"]))

    def __call__(self, hidden: mx.array, ids: Sequence[int], spans: Sequence[Spans], output: Table) -> list[mx.array]:
        """Option logits per question from the prompt's final hidden states [length, hidden] and its token ids, every
        question at once. All in fp32, the same arithmetic as the creator's head question by question (sums in another
        order)."""
        w, h = self.w, self.heads
        seq = _layer_norm(hidden.astype(F32), w, "hidden_norm")
        memory = seq @ w["memory_projection.weight"].T
        last = seq[-1]
        n = len(spans)
        counts = [len(s.options) for s in spans]
        owner = mx.array(np.repeat(np.arange(n), counts))  # each option's question
        # the spans' token positions (questions first, then every option in order) and the span each belongs to
        ranges = [s.question for s in spans] + [o for s in spans for o in s.options]
        positions = np.concatenate([np.arange(a, b) for a, b in ranges])
        members = np.repeat(np.arange(len(ranges)), [b - a for a, b in ranges])
        sizes = mx.array([float(b - a) for a, b in ranges])[:, None]
        means = mx.zeros((len(ranges), seq.shape[-1]), F32).at[mx.array(members)].add(seq[mx.array(positions)]) / sizes
        questions, contexts = means[:n], means[n:]
        option_ranges = ranges[n:]
        option_tokens = np.concatenate([np.asarray(ids[a:b], dtype=np.int64) for a, b in option_ranges])
        option_members = np.repeat(np.arange(len(option_ranges)), [b - a for a, b in option_ranges])
        embedded = rows(output, mx.array(option_tokens)).astype(F32)
        lexical = mx.zeros((len(option_ranges), embedded.shape[-1]), F32).at[mx.array(option_members)].add(embedded)
        lexical = lexical / sizes[n:]
        routed = (
            contexts @ w["option_context_projection.weight"].T
            + lexical @ w["option_lexical_projection.weight"].T
            + (questions @ w["option_question_projection.weight"].T)[owner]
        )
        for i in range(self.routing_layers):
            p = f"evidence_layers.{i}."
            mem = _layer_norm(memory, w, p + "memory_norm")
            routed = routed + _attention(_layer_norm(routed, w, p + "query_norm"), mem, w, p + "attention", h)
            ff = _gelu(_linear(_layer_norm(routed, w, p + "feedforward_norm"), w, p + "feedforward.0"))
            routed = routed + _linear(ff, w, p + "feedforward.3")
        base = questions @ w["question_projection.weight"].T
        # each question's options as one row of a padded [questions, most options] array; padding scores -inf
        widest = max(counts)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        slots = starts[:, None] + np.arange(widest)[None]
        valid = np.arange(widest)[None] < np.asarray(counts)[:, None]
        padded = routed[mx.array(np.where(valid, slots, 0))]  # [questions, widest, width]
        attention = (padded * base[:, None]).sum(-1) / math.sqrt(routed.shape[-1])
        attention = mx.where(mx.array(valid), attention, -mx.inf)
        summaries = (mx.softmax(attention, axis=-1)[..., None] * padded).sum(axis=1)
        fields = (
            base
            + _layer_norm(summaries, w, "option_summary_norm")
            + (last @ w["global_projection.weight"].T)[None]
            + w["type_embedding.weight"][mx.array([s.kind for s in spans])]
        )
        for i in range(self.layers):  # TransformerDecoderLayer(norm_first=True, activation="gelu")
            p = f"layers.{i}."
            x = _layer_norm(fields, w, p + "norm1")
            fields = fields + _attention(x, x, w, p + "self_attn", h)
            fields = fields + _attention(_layer_norm(fields, w, p + "norm2"), memory, w, p + "multihead_attn", h)
            x = _layer_norm(fields, w, p + "norm3")
            fields = fields + _linear(_gelu(_linear(x, w, p + "linear1")), w, p + "linear2")
        fields = _layer_norm(fields, w, "field_norm")
        cap = math.log(100.0)
        prior_scale = mx.exp(mx.minimum(w["prior_logit_scale"], cap))
        joint_scale = mx.exp(mx.minimum(w["joint_logit_scale"], cap))
        gate = mx.sigmoid(w["residual_gate"])
        anchor = _normalize(questions + last[None])[owner]
        prior = prior_scale * (_normalize(lexical) * anchor).sum(-1)
        o = _layer_norm(routed, w, "option_norm")
        f = fields[owner]
        norms = mx.sqrt((f * f).sum(-1)) * mx.sqrt((o * o).sum(-1))
        cosine = (f * o).sum(-1) / mx.maximum(norms, 1e-8)
        features = mx.concatenate([f, o, f * o, mx.abs(f - o)], axis=-1)
        residual = _linear(_gelu(_linear(features, w, "residual_scorer.0")), w, "residual_scorer.3")[:, 0]
        logits = prior + gate * (joint_scale * cosine + residual)
        return [logits[int(a) : int(a) + c] for a, c in zip(starts, counts, strict=True)]


@dataclass
class ClefFormat(QwenImageFormat):
    tokenizer: Tokenizer
    head: JointSchemaHead
    output: Table  # the backbone's output embedding [vocabulary, hidden], maybe quantized
    window: int = MAX_LENGTH  # the creator's window: a prompt's tokens at most (Decider.load's `window` replaces it)
    images: QwenImages | None = None  # None: a build without the vision tower, text only

    def render(self, request: Request) -> Rendering:
        pieces: list[str] = ["\n\nSCHEMA FIELDS:\n"]
        layout: list[tuple[str, int, str | None]] = []  # (question id, piece index, option, None: the question)
        for n, q in enumerate(request.questions):
            if q.type != "noul" and not q.criteria:
                raise InvalidRequest(f"{q.id}: criteria must not be empty")
            pieces.append(f"\nFIELD {n + 1}\nID: {q.id}\nTYPE: {q.type}\nINSTRUCTION: ")
            instructions = q.instructions if q.instructions not in (None, "") else q.id
            layout.append((q.id, len(pieces), None))
            pieces.append(render(instructions))
            pieces.append("\nALLOWED OPTIONS:\n")
            for j, (option, description) in enumerate(question_options(q)):
                pieces.append(f"OPTION {j + 1}: ")
                semantics: dict[str, Any] = {"option_id": option}
                if description is not None:
                    semantics["description"] = description
                layout.append((q.id, len(pieces), option))
                pieces.append(render(semantics))
                pieces.append("\n")
            pieces.append("END FIELD\n")
        head = chat_head(SYSTEM_PROMPT) + "STATE:\n"
        tail = "\n" + ANSWER_TURN + "JOINT SCHEMA DECISIONS:"
        state_text = render(request.state)
        media = [VISION * len(request.images) + "\n"] if request.images else []
        encoded = self.tokenizer.encode_batch([head, tail, *pieces, *media])
        prefix, suffix, schema = encoded[0], encoded[1], encoded[2 : 2 + len(pieces)]
        rows, pixels = self.place(request)
        if rows:  # the state is tokenized on its own, so no text but the images' holds the image token
            assert self.images is not None
            prefix = prefix + self.images.expand(encoded[-1], rows)
        ((state, state_offsets),) = self.tokenizer.encode_offsets([state_text])
        fixed = len(prefix) + sum(len(p) for p in schema) + len(suffix)
        if fixed > self.window:
            what = "the images and the schema need" if request.images else "the schema needs"
            raise ContextTooLong(f"{what} {fixed} tokens before the state; the window is {self.window}")
        room = self.window - fixed
        cuts = (Cut("", "state", len(state) - room, room),) if len(state) > room else ()  # its end, as Clef's code
        state = state[:room]
        starts, at = [], len(prefix) + len(state)
        for piece in schema:
            starts.append(at)
            at += len(piece)
        spans: dict[str, dict[str, Any]] = {q.id: {"options": [], "ids": []} for q in request.questions}
        for qid, index, option_id in layout:
            span = (starts[index], starts[index] + len(schema[index]))
            if option_id is None:
                spans[qid]["question"] = span
            else:
                spans[qid]["options"].append(span)
                spans[qid]["ids"].append(option_id)
        ids = tuple(prefix + state + [t for p in schema for t in p] + suffix)
        result = [
            Spans(spans[q.id]["question"], tuple(spans[q.id]["options"]), tuple(spans[q.id]["ids"]), TYPES[q.type])
            for q in request.questions
        ]
        # The head reads every position, listed as the first question's readouts. The state is tokenized on its own,
        # so a later request with the same state, or with more appended to it, starts with the same tokens up to its
        # marks; the reading of those tokens is kept with their rows (`prefixes.py`) for the head.
        marks = [len(prefix) + m for m in one_mark(state_end_marks(state_text, state_offsets), len(state))]
        first = request.questions[0].id
        prompt = Prompt(
            ids,
            tuple(Readout(first, i) for i in range(len(ids))),
            tuple(m for m in marks if m > 0),
        )
        return Rendering(request, (prompt,), {"spans": result}, cuts, pixels=pixels)

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        request = rendering.request
        logits = self.head(
            hidden.rows(request.questions[0].id), rendering.prompts[0].tokens, rendering.meta["spans"], self.output
        )
        out = {}
        for q, s, z in zip(request.questions, rendering.meta["spans"], logits, strict=True):
            p = mx.softmax(z)
            order = {option: i for i, option in enumerate(s.option_ids)}
            out[q.id] = p if q.type == "noul" else p[mx.array([order[str(o)] for o in q.options])]  # request order
        return out

    def answer(self, question: Question, scores: Any) -> tuple[tuple[float, ...], dict[str, Any]]:
        """The System One answer with its numbers rounded to 4 decimals, as Clef's code returns it."""
        probabilities = tuple(float(x) for x in scores)  # the request's option order (a noul: true, false)
        return probabilities, rounded(systemone_answer(question, probabilities), 4)


# -- its GGUF builds -------------------------------------------------------------------------------------------------

HEAD_FILE = "joint_head.safetensors"  # the joint schema head, a file of its own in the creator's checkpoint


def gguf_scales(x: mx.array) -> dict[str, mx.array]:
    """Clef's [exp(min(prior, ln 100)), exp(min(joint, ln 100)), sigmoid(gate)] as the head's parameters: their logs
    (the cap already applied, which applying it again leaves alone) and the gate's logit."""
    s = [float(v) for v in np.array(x.astype(mx.float32))]
    return {
        "prior_logit_scale": mx.array(math.log(s[0]), dtype=mx.float32),
        "joint_logit_scale": mx.array(math.log(s[1]), dtype=mx.float32),
        "residual_gate": mx.array(math.log(s[2] / (1.0 - s[2])), dtype=mx.float32),
    }


def gguf_head_rules() -> list[Rule]:
    """Clef's joint schema head (joint_head.safetensors): its two evidence-routing layers (dec.blk.0, 1), its four
    decoder layers (dec.blk.2 to 5, layers.0 to 3), its projections, norms, scorer, type embedding and scalars."""
    out = []

    def add(pattern: str, name: str, **kw: object) -> None:
        out.append(Rule(pattern, name, dense=True, part=HEAD_FILE, **kw))  # type: ignore[arg-type]

    def packed(prefix: str, gguf: str, name: str) -> None:
        for piece, x in enumerate("qkv"):
            add(prefix + rf"{gguf}_{x}", name + "in_proj_", piece=piece)

    e, d = r"dec\.blk\.([01])\.", r"dec\.blk\.([2-5])\."
    add(e + "cross_attn_norm", "evidence_layers.{0}.query_norm")
    add(e + "cross_attn_norm_kv", "evidence_layers.{0}.memory_norm")
    packed(e, "cross_attn", "evidence_layers.{0}.attention.")
    add(e + "cross_attn_o", "evidence_layers.{0}.attention.out_proj")
    add(e + "ffn_norm", "evidence_layers.{0}.feedforward_norm")
    add(e + "ffn_up", "evidence_layers.{0}.feedforward.0")
    add(e + "ffn_down", "evidence_layers.{0}.feedforward.3")
    layer = "layers.{n}."
    for pattern, name in (
        ("attn_norm", "norm1"),
        ("cross_attn_norm", "norm2"),
        ("ffn_norm", "norm3"),
        ("ffn_up", "linear1"),
        ("ffn_down", "linear2"),
        ("attn_o", "self_attn.out_proj"),
        ("cross_attn_o", "multihead_attn.out_proj"),
    ):
        add(d + pattern, layer + name, offset=2)
    for piece, x in enumerate("qkv"):
        add(d + f"attn_{x}", layer + "self_attn.in_proj_", piece=piece, offset=2)
        add(d + f"cross_attn_{x}", layer + "multihead_attn.in_proj_", piece=piece, offset=2)
    for gguf, name in (
        ("hidden_norm", "hidden_norm"),
        ("option_summary_norm", "option_summary_norm"),
        ("field_norm", "field_norm"),
        ("option_norm", "option_norm"),
        ("proj_memory", "memory_projection"),
        ("proj_question", "question_projection"),
        ("proj_option_question", "option_question_projection"),
        ("proj_global", "global_projection"),
        ("proj_option_context", "option_context_projection"),
        ("proj_option_lexical", "option_lexical_projection"),
        ("scorer", "residual_scorer.0"),
    ):
        add(rf"decision\.{gguf}", name)
    add(r"decision\.scorer_out", "residual_scorer.3", transform=row)
    add(r"decision\.scales", "", split=gguf_scales)
    add(r"token_types", "type_embedding")
    return out


def gguf_tensors(header: Header) -> tuple[list[Rule], int]:
    """A GGUF build's Qwen3.5 and joint head (as joint_head.safetensors), and its backbone's blocks."""
    return [*qwen35_rules(header), *gguf_head_rules()], int(header.get("block_count"))


class Clef(AdapterBase):
    api = API_VERSION
    name = "cloudflare/clef"
    compare_arms = ("community-clef",)  # a community MLX build's own clef_mlx.py (`compare.py`)
    gguf_rules = Reads(gguf_tensors)  # how its GGUF builds read (no mmproj file)
    files = (
        "config.json",
        "joint_head_config.json",
        HEAD_FILE,
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model*.safetensors",
        "model.safetensors.index.json",
    )

    def matches(self, model: ModelDir) -> bool:
        head = model.json("joint_head_config.json")
        return isinstance(head, dict) and {"routing_layers", "layers", "heads"} <= set(head)

    def gguf_check(self, header: Header, model: ModelDir) -> list[str]:
        """A GGUF build's Qwen3.5 against the config, and its joint head's layers and heads against
        joint_head_config.json."""
        head = model.json("joint_head_config.json") or {}
        expected = {
            "decision.routing_block_count": head.get("routing_layers"),
            "decision.block_count": head.get("layers"),
            "decision.head_count": head.get("heads"),
        }
        differ = qwen35_differences(header, model.config)
        return differ + [f"{k} {header.get(k)} against {v}" for k, v in expected.items() if header.get(k) != v]

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        weights = ctx.weights(model, exclude=(HEAD_FILE,))
        backbone = ctx.backbone(qwen35, model, weights=weights)
        head = JointSchemaHead.load(ctx.weights(model, [HEAD_FILE]), model.json("joint_head_config.json"))
        if head.w["memory_projection.weight"].shape[1] != backbone.hidden_size:
            raise UnsupportedModel(f"{model.path}: the joint head does not fit the backbone")
        tokenizer = ctx.tokenizer(model)
        # Ollama's build carries the vision tower, but its answers on images are unchecked (verified.py lists
        # none for it), so it reads text only
        images = (
            None
            if model.source.kind == "ollama"
            else QwenImages.attach(model, ctx, backbone, weights, tokenizer, exif=False)
        )
        fmt = ClefFormat(tokenizer, head, backbone.output_matrix, images=images)
        shown = {"images": "up to the window" if images is not None else "none (text only)"}
        return Loaded(backbone, fmt, {"architecture": "qwen3.5", "head": "joint schema", **shown})
