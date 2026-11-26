"""Sqwish's Kettei Core and Dot (Kettei Community Model License 1.0), as the sqwishai SDK runs them on a Mac
(`sqwishai.kettei`, 0.5.0): its prompts, checked token for token against it, its option labels past 26
(kettei_labels.json) and System One's confidences.
"""

from __future__ import annotations

import json
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any

import mlx.core as mx

from ..adapter import AdapterBase, FormatBase, Hidden, LoadContext, Loaded, Rendering
from ..arch import dense
from ..arch.rope import RoPE
from ..errors import ContextTooLong, InvalidRequest, UnsupportedModel
from ..formulas import chance_rescaled, median_level
from ..model_dir import API_VERSION, ModelDir
from ..plan import Prompt, Readout, shared_prefix, state_end_marks
from ..quant import Table
from ..readout import last_label_probabilities
from ..schema import Question, QuestionType, Request, as_text, systemone_answer
from ..tokenizer import Tokenizer

SIZES = {64256: "dot", 250624: "core"}  # a checkpoint's vocab_size -> its size, as the SDK tells them apart
NO_CONTEXT = "(no context)"
USER = "<|ifm|im_start|>user\n"
ANSWER = "<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>Answer:"
RESERVED = "abstain"  # the API keeps this outcome for its abstain policy, so the SDK refuses an option of that name


@cache
def size_labels(size: str) -> tuple[str, ...]:
    """The labels past 26 options, as the SDK's alphabets.json lists them for `size`."""
    found = json.loads(files(__package__).joinpath("kettei_labels.json").read_text(encoding="utf-8"))
    return tuple(found[size])


def config(c: Mapping[str, Any]) -> dense.Config:
    """config.json as a dense decoder, or UnsupportedModel for a variant these layers don't cover."""
    if c.get("model_type") != "k2_horizon":
        raise UnsupportedModel(f"not a Kettei checkpoint: {c.get('model_type')!r}")
    layers = int(c["num_hidden_layers"])
    if c.get("num_experts") or c.get("mova_num_experts") or len(c.get("mlp_only_layers") or []) != layers:
        raise UnsupportedModel("a Kettei checkpoint with experts: these layers are dense only")
    if c.get("attention_gate_func") or c.get("hidden_act", "silu") != "silu":
        raise UnsupportedModel("a Kettei variant these layers don't cover (attention gate or activation)")
    head_dim = int(c["head_dim"])
    if int(c.get("rope_head_dim") or head_dim) != head_dim:
        raise UnsupportedModel("RoPE over part of the head")
    bias = bool(c.get("attention_bias", False))
    return dense.Config(
        family="k2_horizon",
        hidden=int(c["hidden_size"]),
        layers=layers,
        heads=int(c["num_attention_heads"]),
        kv_heads=int(c["num_key_value_heads"]),
        head_dim=head_dim,
        eps=float(c.get("rms_norm_eps", 1e-6)),
        qkv_bias=bias,
        o_bias=bias,
        qk_norm=bool(c.get("query_key_norm", False)),
        tied=bool(c.get("tie_word_embeddings", False)),
        rope=RoPE.from_hf(
            head_dim,
            c.get("rope_parameters"),
            theta=float(c.get("rope_theta") or 10_000.0),
            max_positions=int(c["max_position_embeddings"]),
        ),
        norm_groups=int(c.get("layernorm_num_groups", 1)),
    )


def median_closeness(p: Sequence[float]) -> float:
    """One minus the expected distance, in levels, from the median level (`median_level`): Kettei's score
    confidence, as its SDK computes it."""
    median = median_level(p)
    return max(0.0, 1 - sum(x * abs(level - median) for level, x in enumerate(p)))


def question_text(q: Question) -> tuple[str, list[str], bool]:
    """A question's text, its options' texts and whether it is asked Yes or No, as the SDK writes the API's
    binary, single and ordinal decisions that a System One noul, choice and score become."""
    if q.instructions is None or (isinstance(q.instructions, str) and not q.instructions.strip()):
        raise InvalidRequest(f"{q.id}: a question needs instructions")  # the SDK refuses an empty question
    instructions = as_text(q.instructions)
    if q.type == "noul":
        given = dict(q.criteria)
        described = [f"{word}: {as_text(given[key])}" for word, key in (("Yes", "true"), ("No", "false"))
                     if given.get(key) is not None]  # fmt: skip
        return ("\n\n".join([instructions, "\n".join(described)]) if described else instructions), ["Yes", "No"], True
    if q.type == "choice":
        if RESERVED in q.options:
            raise InvalidRequest(f'{q.id}: "{RESERVED}" is reserved for the abstain policy; rename that option')
        return instructions, [k if d is None else f"{k}: {as_text(d)}" for k, d in q.criteria], False
    return instructions, [f"{i}: {as_text(level)}" for i, (_, level) in enumerate(q.criteria)], False


@dataclass
class KetteiFormat(FormatBase):
    tokenizer: Tokenizer
    output: Table  # the output matrix [vocabulary, hidden]
    size: str
    bos: str  # what the chat format starts with: Core's begin-of-text, nothing for Dot
    positions: int

    def most_options(self, kind: QuestionType) -> int | None:
        return max(26, len(size_labels(self.size)))  # A..Z, then the size's own labels

    def labels(self, q: Question, count: int, yes_no: bool) -> tuple[list[str], list[int], str]:
        """The answer labels, their tokens and how to answer, as the SDK's decision_labels gives them."""
        if yes_no:
            labels, how = [" Yes", " No"], "Answer Yes or No, nothing else."
        elif count <= 10:
            labels, how = [str(i) for i in range(count)], "Answer with the number of the best option, nothing else."
        else:
            labels = list(string.ascii_uppercase[:count]) if count <= 26 else list(size_labels(self.size)[:count])
            how = f"Answer with the {'letter' if count <= 26 else 'class symbol'} of the best option, nothing else."
        ids = [self.tokenizer.encode(label) for label in labels]
        if len(labels) < count or any(len(t) != 1 for t in ids) or len({t[0] for t in ids}) != count:
            raise InvalidRequest(f"{q.id}: {count} options are more than this model has single-token labels for")
        if special := {t[0] for t in ids} & self.tokenizer.special_ids:  # a tokenizer that changed under the labels
            raise UnsupportedModel(f"answer labels read as special tokens {sorted(special)}, as the SDK refuses them")
        return labels, [t[0] for t in ids], how

    def render(self, request: Request) -> Rendering:
        context = as_text("" if request.state is None else request.state)
        context = context if context.strip() else NO_CONTEXT
        head = self.bos + USER + "<read>\n" + context
        texts, labels = [], {}
        for q in request.questions:
            text, options, yes_no = question_text(q)
            if not yes_no and len(options) < 2:  # the SDK refuses these too, and chance_rescaled has no answer for them
                raise InvalidRequest(
                    f"{q.id}: {len(options)} option; Kettei answers a question with 2 or more, as its SDK does"
                )
            names, labels[q.id], how = self.labels(q, len(options), yes_no)
            shown = (
                "\n".join(options)
                if yes_no
                else "\n".join(f"{n.strip()}. {o}" for n, o in zip(names, options, strict=True))
            )
            texts.append(f"{head}\n</read>\n\nQuestion:\n{text.strip()}\n\nOptions:\n{shown}\n\n{how}{ANSWER}")
        prompts = []
        for q, text, (ids, offsets) in zip(request.questions, texts, self.tokenizer.encode_offsets(texts), strict=True):
            if len(ids) > self.positions:
                raise ContextTooLong(f"{q.id}: a prompt of {len(ids)} tokens, past the model's {self.positions}")
            prompts.append(
                Prompt(tuple(ids), (Readout(q.id, len(ids) - 1),), state_end_marks(text, offsets, len(head)))
            )
        return Rendering(request, tuple(prompts), meta=labels)

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        return {
            q.id: last_label_probabilities(hidden.rows(q.id), self.output, rendering.meta[q.id])
            for q in rendering.request.questions
        }

    def input_tokens(self, rendering: Rendering) -> int:
        """The prompts' shared start once and the rest of each, as the SDK counts them on a Mac."""
        shared = shared_prefix(rendering.prompts)
        return shared + sum(len(p.tokens) - shared for p in rendering.prompts)

    def answer(self, question: Question, scores: Any) -> tuple[tuple[float, ...], dict[str, Any]]:
        p = tuple(float(x) for x in scores)  # a noul's Yes, No: true, then false
        return p, systemone_answer(question, p, choice_confidence=chance_rescaled, score_confidence=median_closeness)


class Kettei(AdapterBase):
    api = API_VERSION
    name = "sqwish/kettei"
    files = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "*.safetensors",
        "model.safetensors.index.json",
    )

    def matches(self, model: ModelDir) -> bool:
        return model.config.get("model_type") == "k2_horizon" and int(model.config.get("vocab_size", 0)) in SIZES

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        backbone = ctx.backbone(dense, model, config=config(model.config))
        fmt = kettei_format(model, ctx.tokenizer(model), backbone.output_matrix)
        return Loaded(backbone, fmt, {"size": fmt.size})


def kettei_format(model: ModelDir, tokenizer: Tokenizer, output: Table) -> KetteiFormat:
    """A checkpoint's format: its size, what its chat format starts with and its positions."""
    template = model.file("chat_template.jinja").read_text()
    bos = str(model.json("tokenizer_config.json")["bos_token"]) if template.startswith("{{- bos_token }}") else ""
    size = SIZES[int(model.config["vocab_size"])]
    return KetteiFormat(tokenizer, output, size, bos, int(model.config["max_position_embeddings"]))
