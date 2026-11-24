"""A GGUF build's tensors under the names of its creator's checkpoint, llama.cpp's conversion undone.

llama.cpp's converter renames every tensor (`blk.N.ffn_down` for `model.layers.N.feed_forward.w2`) and transforms
some. Each table maps a GGUF base name (its suffix, `.weight` or `.bias`, carried along) to the creator's, with
the transform to undo, and was found tensor by tensor against the creator's own checkpoint: every GGUF tensor mapped,
equal after dequantizing within its type's rounding (Q8_0: max|block| / 254), or exactly for the ones stored
unquantized. The architectures' tables are here; a model's own parts (a decision head stored in the build) are its
adapter's, which declares how its builds read (`Reads`, the adapter's `gguf_rules`). The transforms met so far:

- LFM3: convolutions stored [channels, taps] ([channels, 1, taps] in the checkpoint); its zero-tail RoPE added as
  a tensor of per-pair frequency factors (`rope_freqs`: 1, then 1e30 for the pairs that never turn), which the
  checkpoint has no tensor for and its adapter's `gguf_check` holds to its config;
- Qwen3.5: zero-centred norms stored as 1 + w; A_log stored as -exp(A_log); the Gated DeltaNet's value heads stored
  tiled where the checkpoint groups them under their key head (GGUF head g is head (g mod n_k) * r + g div n_k of
  the checkpoint, r = n_v / n_k), in the value rows of the qkv projection and the convolution, the z projection, the
  per-head a, b, A_log and dt_bias, and the output projection's columns: undone by gathering whole rows, or whole
  column blocks of the packed matrix, never by dequantizing;
- decision heads: packed attention projections stored as q, k and v apiece; scalars stored applied, handed back as
  the parameters the creator's head applies them to (a rule's `split`).

A quantized matrix stays quantized (MLX's affine groups of 32) when the creator's name ends in `.weight`, which is how
ippon reads quantized checkpoints; a head's tensors are dequantized to fp32 (its code reads them so).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from ..errors import UnsupportedModel
from .header import GROUP_SIZE, QUANTIZED, Header

F32 = mx.float32
SUFFIXES = (".weight", ".bias")  # a GGUF name's (MLX splits a quantized one into .weight, .scales, .biases)

# x, which array of a quantized matrix it is ("dense", "weight", "scales", "biases"), and the bits it is packed at
Transform = Callable[[mx.array, str, int], mx.array]


@dataclass(frozen=True)
class Reads:
    """How a model's GGUF builds read, as its adapter declares them (`gguf_rules`): the file's tensors under the
    creator's names and the blocks of its backbone (`tensors(header)`), and its mmproj file's (`media`; None: ippon
    reads no mmproj of these builds)."""

    tensors: Callable[[Header], tuple[list[Rule], int]]
    media: Callable[[], list[Rule]] | None = None


@dataclass(frozen=True)
class Rule:
    pattern: str  # a GGUF base name, a regular expression whose groups fill `name`
    name: str  # the creator's base name, "{0}" for the first group, "{n}" for it minus `offset`
    transform: Transform | None = None
    offset: int = 0
    dense: bool = False  # dequantized to fp32 (a head's tensors, read in full precision)
    part: str = ""  # the creator's file it belongs to: "" the model's own, else that file (a head stored apart)
    piece: int | None = None  # the slice of a packed creator tensor it is (q, k, v: 0, 1, 2), joined in order
    join: Callable[[list[mx.array]], mx.array] | None = field(default=None)  # its pieces joined (default: q, k, v)
    rows: tuple[int, ...] | None = None  # only these rows, taken before anything else (still quantized)
    split: Callable[[mx.array], dict[str, mx.array]] | None = field(default=None)  # one tensor to several
    skip: bool = False  # a tensor llama.cpp adds that the checkpoint doesn't hold, checked instead (`gguf_check`)


# -- transforms -----------------------------------------------------------------------------------------------------


def conv(x: mx.array, kind: str, bits: int) -> mx.array:
    """[channels, taps] to the checkpoint's [channels, 1, taps]."""
    return x.reshape(x.shape[0], 1, x.shape[-1])


def row(x: mx.array, kind: str, bits: int) -> mx.array:
    """A [n] weight vector to [1, n]."""
    return x.reshape(1, -1) if x.ndim == 1 and x.size > 1 else x


def frames(pieces: list[mx.array]) -> mx.array:
    """A temporal patch's convolution, stored one frame apiece ([out, channels, h, w] each), back to the
    checkpoint's [out, channels, frames, h, w]; its bias, stored once, as it is."""
    return mx.stack(pieces, axis=2) if pieces[0].ndim == 4 else pieces[0]


def flat_frames(pieces: list[mx.array]) -> mx.array:
    """The same as one linear patchifier, [out, channels * frames * h * w]."""
    joined = frames(pieces)
    return joined.reshape(joined.shape[0], -1) if joined.ndim == 5 else joined


def flat_bias(x: mx.array, kind: str, bits: int) -> mx.array:
    """A convolution's bias stored [out, 1, 1] back to [out]."""
    return x.reshape(-1) if x.ndim == 3 else x


def taps(x: mx.array, kind: str, bits: int) -> mx.array:
    """A one-tap convolution stored as a matrix [out, in] back to [out, in, 1]."""
    return x.reshape(*x.shape, 1) if x.ndim == 2 else x


def minus_one(x: mx.array, kind: str, bits: int) -> mx.array:
    """A zero-centred norm stored as 1 + w, back to w (exact in fp32 for stored weights of these sizes)."""
    return x.astype(F32) - 1.0


def log_minus(x: mx.array, kind: str, bits: int) -> mx.array:
    """-exp(A_log) back to A_log."""
    return mx.log(-x.astype(F32))


def tiled(n_k: int, n_v: int) -> np.ndarray:
    """For each of the checkpoint's value heads, the GGUF's head that holds it."""
    r = n_v // n_k
    return np.array([(o % r) * n_k + o // r for o in range(n_v)])


def heads_rows(n_k: int, n_v: int, block: int, offset: int = 0) -> Transform:
    """Rows from `offset` on are n_v heads of `block` rows, stored tiled: put them in the checkpoint's order (the same
    rows of a quantized matrix's packed weight, scales and biases)."""
    order = tiled(n_k, n_v)

    def apply(x: mx.array, kind: str, bits: int) -> mx.array:
        index = np.concatenate([np.arange(offset), *(offset + g * block + np.arange(block) for g in order)])
        return x[mx.array(index)]

    return apply


def heads_columns(n_k: int, n_v: int, block: int) -> Transform:
    """Columns are n_v heads of `block` columns, stored tiled: put them in the checkpoint's order. A quantized
    matrix's columns move as whole groups: `block` columns are block * bits / 32 packed words and block / 32 scales."""
    order = tiled(n_k, n_v)

    def apply(x: mx.array, kind: str, bits: int) -> mx.array:
        width = {"dense": block, "weight": block * bits // 32}.get(kind, block // GROUP_SIZE)
        index = np.concatenate([g * width + np.arange(width) for g in order])
        return x[:, mx.array(index)]

    return apply


def _then(*steps: Transform) -> Transform:
    def apply(x: mx.array, kind: str, bits: int) -> mx.array:
        for step in steps:
            x = step(x, kind, bits)
        return x

    return apply


# -- the tables -----------------------------------------------------------------------------------------------------


def lfm3_rules(root: str = "model.language_model.") -> list[Rule]:
    """LFM3 (Liquid's d1-2B and d1-Omni-2B, their text model): sandwich norms, gated attention, and the RoPE factors
    llama.cpp stores apart."""
    layer = root + "layers.{0}."
    return [
        Rule(r"token_embd", root + "embed_tokens"),
        Rule(r"output_norm", root + "embedding_norm"),
        Rule(r"rope_freqs", "", skip=True),
        Rule(r"blk\.(\d+)\.attn_norm", layer + "operator_norm"),
        Rule(r"blk\.(\d+)\.post_attention_norm", layer + "post_operator_norm"),
        Rule(r"blk\.(\d+)\.ffn_norm", layer + "ffn_norm"),
        Rule(r"blk\.(\d+)\.post_ffw_norm", layer + "post_ffn_norm"),
        Rule(r"blk\.(\d+)\.ffn_gate", layer + "feed_forward.w1"),
        Rule(r"blk\.(\d+)\.ffn_up", layer + "feed_forward.w3"),
        Rule(r"blk\.(\d+)\.ffn_down", layer + "feed_forward.w2"),
        Rule(r"blk\.(\d+)\.attn_q", layer + "self_attn.q_proj"),
        Rule(r"blk\.(\d+)\.attn_k", layer + "self_attn.k_proj"),
        Rule(r"blk\.(\d+)\.attn_v", layer + "self_attn.v_proj"),
        Rule(r"blk\.(\d+)\.attn_gate", layer + "self_attn.gate_proj"),
        Rule(r"blk\.(\d+)\.attn_output", layer + "self_attn.o_proj"),
        Rule(r"blk\.(\d+)\.attn_q_norm", layer + "self_attn.q_norm"),
        Rule(r"blk\.(\d+)\.attn_k_norm", layer + "self_attn.k_norm"),
        Rule(r"blk\.(\d+)\.shortconv\.in_proj", layer + "conv.in_proj"),
        Rule(r"blk\.(\d+)\.shortconv\.conv", layer + "conv.conv", conv),
        Rule(r"blk\.(\d+)\.shortconv\.out_proj", layer + "conv.out_proj"),
    ]


def lfm3_media_rules() -> list[Rule]:
    """LFM3's mmproj file (llama.cpp's picture and audio parts beside the text model): the vision tower and its
    projector, and on d1-Omni the speech encoder and its projector. Its patchifier is stored as a convolution per
    frame of a temporal patch (`.weight` and `.weight.1`, [out, channels, 16, 16] each), joined back."""
    vision, audio = "model.vision_tower.", "model.audio_encoder."
    v, a = vision + "layers.{0}.", audio + "lfm.layers.{0}."
    blocks = (
        ("ln1", "operator_norm"),
        ("ln2", "ffn_norm"),
        ("ffn_gate", "feed_forward.w1"),
        ("ffn_up", "feed_forward.w3"),
        ("ffn_down", "feed_forward.w2"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_out", "self_attn.out_proj"),
        ("attn_q_norm", "self_attn.q_layernorm"),
        ("attn_k_norm", "self_attn.k_layernorm"),
    )
    rules = [
        Rule(r"v\.patch_embd", vision + "patchifier", piece=0, join=flat_frames),
        Rule(r"v\.patch_embd\.weight\.1", vision + "patchifier.weight", piece=1, join=flat_frames),
        Rule(r"v\.embedding_norm", vision + "embedding_norm"),
        Rule(r"v\.blk\.(\d+)\.conv2d_in_proj", v + "conv.in_proj"),
        Rule(r"v\.blk\.(\d+)\.conv2d_out_proj", v + "conv.out_proj"),
        Rule(r"v\.blk\.(\d+)\.conv2d_spatial", v + "conv.spatial_conv"),
        Rule(r"mm\.input_norm", "model.multi_modal_projector.layer_norm"),
        Rule(r"mm\.0", "model.multi_modal_projector.linear_1"),
        Rule(r"mm\.2", "model.multi_modal_projector.linear_2"),
        Rule(r"a\.conv1d\.0", audio + "downsample.conv_in", flat_bias),
        Rule(r"a\.conv1d\.2", audio + "downsample.conv_mid", flat_bias),
        Rule(r"a\.conv1d\.3", audio + "downsample.lin_mid", flat_bias),
        Rule(r"a\.conv1d\.5", audio + "downsample.conv_end", flat_bias),
        Rule(r"a\.conv1d\.6", audio + "downsample.lin_end", flat_bias),
        Rule(r"a\.pre_encode\.out", audio + "downsample.lin_out", taps, dense=True),
        Rule(r"a\.post_ln", audio + "lfm.embedding_norm"),
        Rule(r"a\.blk\.(\d+)\.conv_dw", a + "conv.conv", conv),
        Rule(r"a\.blk\.(\d+)\.conv_pw1", a + "conv.in_proj"),
        Rule(r"a\.blk\.(\d+)\.conv_pw2", a + "conv.out_proj"),
        Rule(r"mm\.a\.mlp\.0", "model.audio_projector.layer_norm"),
        Rule(r"mm\.a\.mlp\.1", "model.audio_projector.linear_1"),
        Rule(r"mm\.a\.mlp\.3", "model.audio_projector.linear_2"),
    ]
    for prefix, layer in ((r"v\.blk", v), (r"a\.blk", a)):
        rules += [Rule(prefix + rf"\.(\d+)\.{gguf}", layer + name) for gguf, name in blocks]
    return rules


def qwen3vl_media_rules(root: str = "model.visual.") -> list[Rule]:
    """Qwen3-VL's vision tower and merger as llama.cpp's mmproj stores them (H2O-Lightning-4B's and pplx-decider's):
    the patch convolution one frame apiece, its learned position table, and the merger's norm as the tower's post_ln."""
    block = root + "blocks.{0}."
    return [
        Rule(r"v\.patch_embd", root + "patch_embed.proj", piece=0, join=frames),
        Rule(r"v\.patch_embd\.weight\.1", root + "patch_embed.proj.weight", piece=1, join=frames),
        Rule(r"v\.position_embd", root + "pos_embed"),
        Rule(r"v\.post_ln", root + "merger.norm"),
        Rule(r"mm\.0", root + "merger.linear_fc1"),
        Rule(r"mm\.2", root + "merger.linear_fc2"),
        Rule(r"v\.blk\.(\d+)\.ln1", block + "norm1"),
        Rule(r"v\.blk\.(\d+)\.ln2", block + "norm2"),
        Rule(r"v\.blk\.(\d+)\.attn_qkv", block + "attn.qkv"),
        Rule(r"v\.blk\.(\d+)\.attn_out", block + "attn.proj"),
        Rule(r"v\.blk\.(\d+)\.ffn_up", block + "mlp.linear_fc1"),
        Rule(r"v\.blk\.(\d+)\.ffn_down", block + "mlp.linear_fc2"),
    ]


def qwen35_rules(header: Header, root: str = "model.language_model.") -> list[Rule]:
    """Qwen3.5's text model under `root` (its output matrix as lm_head)."""
    n_k, n_v = int(header.get("ssm.group_count")), int(header.get("ssm.time_step_rank"))
    d_k, d_v = int(header.get("ssm.state_size")), int(header.get("ssm.inner_size")) // n_v
    values = 2 * n_k * d_k  # the qkv projection's (and the convolution's) value rows start here
    layer, g = root + "layers.{0}.", root + "layers.{0}.linear_attn."
    return [
        Rule(r"token_embd", root + "embed_tokens"),
        Rule(r"output_norm", root + "norm", minus_one),
        Rule(r"output", "lm_head"),
        Rule(r"blk\.(\d+)\.attn_norm", layer + "input_layernorm", minus_one),
        Rule(r"blk\.(\d+)\.post_attention_norm", layer + "post_attention_layernorm", minus_one),
        Rule(r"blk\.(\d+)\.ffn_gate", layer + "mlp.gate_proj"),
        Rule(r"blk\.(\d+)\.ffn_up", layer + "mlp.up_proj"),
        Rule(r"blk\.(\d+)\.ffn_down", layer + "mlp.down_proj"),
        Rule(r"blk\.(\d+)\.attn_q", layer + "self_attn.q_proj"),  # with its output gate, as the checkpoint has it
        Rule(r"blk\.(\d+)\.attn_k", layer + "self_attn.k_proj"),
        Rule(r"blk\.(\d+)\.attn_v", layer + "self_attn.v_proj"),
        Rule(r"blk\.(\d+)\.attn_output", layer + "self_attn.o_proj"),
        Rule(r"blk\.(\d+)\.attn_q_norm", layer + "self_attn.q_norm", minus_one),
        Rule(r"blk\.(\d+)\.attn_k_norm", layer + "self_attn.k_norm", minus_one),
        Rule(r"blk\.(\d+)\.attn_qkv", g + "in_proj_qkv", heads_rows(n_k, n_v, d_v, values)),
        Rule(r"blk\.(\d+)\.attn_gate", g + "in_proj_z", heads_rows(n_k, n_v, d_v)),
        Rule(r"blk\.(\d+)\.ssm_alpha", g + "in_proj_a", heads_rows(n_k, n_v, 1)),
        Rule(r"blk\.(\d+)\.ssm_beta", g + "in_proj_b", heads_rows(n_k, n_v, 1)),
        Rule(r"blk\.(\d+)\.ssm_a", g + "A_log", _then(heads_rows(n_k, n_v, 1), log_minus)),
        Rule(r"blk\.(\d+)\.ssm_dt", g + "dt_", heads_rows(n_k, n_v, 1)),  # ssm_dt.bias: dt_bias
        Rule(r"blk\.(\d+)\.ssm_conv1d", g + "conv1d", _then(heads_rows(n_k, n_v, d_v, values), conv)),
        Rule(r"blk\.(\d+)\.ssm_norm", g + "norm"),
        Rule(r"blk\.(\d+)\.ssm_out", g + "out_proj", heads_columns(n_k, n_v, d_v)),
    ]


# -- renaming -------------------------------------------------------------------------------------------------------


def _split_suffix(name: str) -> tuple[str, str]:
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return name, ""


def _join(target: str, suffix: str) -> str:
    return target + suffix[1:] if target.endswith("_") else target + suffix


def _match(
    rules: list[Rule], base: str, backbone: int, header: Header, gguf_name: str
) -> tuple[Rule, re.Match[str], int | None]:
    """The one rule a tensor's name (`base`, without its suffix) matches, the match, and its block number; a block at
    or above `backbone` matches only rules with an offset, those below only the others. UnsupportedModel when no
    rule or several do."""
    found = []
    for rule in rules:
        m = re.fullmatch(rule.pattern, base)
        if m is None:
            continue
        block = int(m.group(1)) if m.groups() else None
        if block is not None and backbone and (block >= backbone) != bool(rule.offset) and rule.part == "":
            continue
        found.append((rule, m, block))
    if len(found) != 1:
        raise UnsupportedModel(
            f"{header.path.name}: tensor {gguf_name!r} is "
            + ("not one ippon knows" if not found else "ambiguous")
            + f" for a {header.architecture} build"
        )
    return found[0]


def rename(
    header: Header,
    tensors: Mapping[str, mx.array],
    rules: list[Rule],
    backbone: int,
    part: str | None = None,
) -> dict[str, mx.array]:
    """`tensors` (as mx.load reads the GGUF) under the creator's names; with `part`, only that part's (a rule's
    `part`: "" for the model's own files). Every tensor must match exactly one rule (UnsupportedModel otherwise);
    a block numbered at or above `backbone` matches only rules with an offset, those below only the others; a
    packed tensor's pieces are joined in order."""
    out: dict[str, mx.array] = {}
    pieces: dict[str, dict[int, mx.array]] = {}
    joins: dict[str, Callable[[list[mx.array]], mx.array] | None] = {}
    for gguf_name in sorted(header.tensors):
        base, suffix = _split_suffix(gguf_name)
        rule, m, block = _match(rules, base, backbone, header, gguf_name)
        if rule.skip or (part is not None and rule.part != part):
            continue
        # an empty name: the tensor alone in its part's file, by its suffix ("weight"), or split into several
        target = _join(rule.name.format(*m.groups(), n=(block or 0) - rule.offset), suffix) if rule.name else suffix[1:]
        quantized = header.tensors[gguf_name].type in QUANTIZED
        transform = rule.transform or (lambda x, kind, bits: x)
        rows = None if rule.rows is None else mx.array(rule.rows)

        def taken(x: mx.array, rows: mx.array | None = rows) -> mx.array:
            return x if rows is None else x[rows]

        if quantized:
            w, s, b = (taken(tensors[base + x]) for x in (".weight", ".scales", ".biases"))
            bits = w.shape[-1] * 32 // (s.shape[-1] * GROUP_SIZE)
            if rule.dense:
                value = mx.dequantize(w, s.astype(F32), b.astype(F32), group_size=GROUP_SIZE, bits=bits)
                arrays = {target: transform(value, "dense", bits)}
            else:
                stem = target.removesuffix(".weight")
                arrays = {
                    stem + x: transform(a, x[1:], bits) for x, a in ((".weight", w), (".scales", s), (".biases", b))
                }
        else:
            value = taken(tensors[gguf_name])
            arrays = {target: transform(value.astype(F32) if rule.dense else value, "dense", 0)}
        if rule.split is not None:
            out |= rule.split(arrays[target])
        elif rule.piece is not None:
            pieces.setdefault(target, {})[rule.piece] = arrays[target]
            joins[target] = rule.join
        else:
            out |= arrays
    for target, got in pieces.items():
        join = joins[target]
        if join is not None:
            if sorted(got) != list(range(len(got))):
                raise UnsupportedModel(f"{header.path.name}: {target} is missing a piece")
            out[target] = join([got[i] for i in range(len(got))])
            continue
        if sorted(got) != [0, 1, 2]:
            raise UnsupportedModel(f"{header.path.name}: {target} is missing a piece of its q, k and v")
        out[target] = mx.concatenate([got[0], got[1], got[2]], axis=0)
    return out


def plain(header: Header, rules: list[Rule], backbone: int, part: str | None = None) -> dict[str, str]:
    """The creator's names of the tensors `rename` takes as the file stores them (no transform, rows, pieces, split
    or dequantizing), each to its GGUF name: those whose rows can be read from the file (`reader.host_rows`)."""
    out = {}
    for gguf_name in sorted(header.tensors):
        base, suffix = _split_suffix(gguf_name)
        rule, m, block = _match(rules, base, backbone, header, gguf_name)
        if rule.skip or (part is not None and rule.part != part) or not rule.name:
            continue
        if rule.transform or rule.rows is not None or rule.piece is not None or rule.split or rule.dense:
            continue
        out[_join(rule.name.format(*m.groups(), n=(block or 0) - rule.offset), suffix)] = gguf_name
    return out


def targets(header: Header, rules: list[Rule]) -> set[str]:
    """The creator's names `rename` gives the file's tensors, from its header alone (nothing read)."""
    out = set()
    for gguf_name in header.tensors:
        base, suffix = _split_suffix(gguf_name)
        rule, m, block = _match(rules, base, 0, header, gguf_name)
        if not rule.skip and rule.name:
            out.add(_join(rule.name.format(*m.groups(), n=(block or 0) - rule.offset), suffix))
    return out
