"""From a model's output at a readout to a question's option probabilities.

Most decision models read their answer off the language-model head. Each option has one or more label
tokens (" Yes" and " No", "A" to "Z", digits); an option's logit is the log-sum-exp of its tokens'
logits, and its probability is a softmax over the options' logits, optionally at a temperature.

A softmax over the options needs only the options' tokens: normalizing over the whole vocabulary first
cancels out. So label_logits() multiplies the hidden state by the label rows of the output matrix only,
where a full readout reads the whole matrix (for a 250k-token vocabulary and 2,560 channels, 1.3 GB of
bf16 per readout pass) to use a handful of its rows. Models with their own head (a joint schema head, a learned
row per option code) produce option logits directly; the code readout is here too.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx

from . import quant
from .errors import InvalidRequest, UnsupportedModel

if TYPE_CHECKING:
    from .adapter import LoadContext
    from .model_dir import ModelDir
    from .tokenizer import Tokenizer

CODE_ROWS = 255  # option codes a code readout holds (vllm-sr d3's, Perplexity's)


@dataclass(frozen=True)
class LabelSet:
    """The label tokens of each option, in the order of the options."""

    tokens: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.tokens or any(not group for group in self.tokens):
            raise ValueError("every option needs at least one label token")

    @property
    def flat(self) -> tuple[int, ...]:
        """Every label token, option after option: the columns option_probabilities() expects."""
        return tuple(t for group in self.tokens for t in group)


# Up to this many logits (rows x labels) a multiply-and-sum: from 2 rows MLX's fp32 matmul takes a GEMM route that
# costs about 100 us at 2,048 channels and 200 us at 4,096 for a handful of labels, against 0 to 35 us this way up to
# 8 rows by 26 labels (M5 Pro); the two meet around 16 rows by 40 labels. It is also exact fp32 where an M5's matmul
# rounds about as TF32 does.
TINY_PRODUCT = 256


def answer_tokens(tokenizer: Tokenizer, prompt: str, answers: Sequence[str]) -> tuple[int, ...]:
    """The token each answer text (a letter, a code) is read as right after `prompt`: one token of its own, which
    leaves the prompt's tokens as they are, and no two the same; UnsupportedModel for a tokenizer that reads them
    otherwise."""
    start = tokenizer.encode(prompt)
    found = []
    for answer in answers:
        ids = tokenizer.encode(prompt + answer)
        if len(ids) != len(start) + 1 or ids[:-1] != start:
            raise UnsupportedModel(f"{answer!r} is not one token of its own after the prompt")
        found.append(ids[-1])
    if len(set(found)) != len(found):
        raise UnsupportedModel("two answers are read as the same token")
    return tuple(found)


def label_logits(hidden: mx.array, weight: quant.Table, token_ids: Sequence[int]) -> mx.array:
    """Logits [rows, len(token_ids)] in fp32 of hidden states [rows, channels] for the given tokens,
    from the output matrix `weight` [vocabulary, channels] (quantized or not), reading only those tokens' rows."""
    rows = quant.rows(weight, mx.array(list(token_ids))).astype(mx.float32)
    h = hidden.astype(mx.float32)
    if h.shape[0] * rows.shape[0] <= TINY_PRODUCT:  # a few rows by a few labels: MLX's matmul is the slow route there
        return (h[:, None, :] * rows[None]).sum(axis=-1)
    return h @ rows.T


def softmax(logits: mx.array, temperature: float = 1.0) -> mx.array:
    """Softmax over the last axis at a temperature, in fp32; the temperature must be finite and positive."""
    if not (temperature > 0 and math.isfinite(temperature)):
        raise ValueError(f"temperature must be finite and positive, not {temperature}")
    return mx.softmax(logits.astype(mx.float32) / temperature, axis=-1)


def last_label_probabilities(
    hidden: mx.array, weight: quant.Table, token_ids: Sequence[int], temperature: float = 1.0
) -> mx.array:
    """Probabilities [len(token_ids)] in fp32 at the last of hidden states [rows, channels]: each token's logit
    (`label_logits`), divided by `temperature`, softmax. The readout of a model that answers with one token per
    option at its prompt's end."""
    return softmax(label_logits(hidden[-1:], weight, token_ids)[0], temperature)


def log_sum_exp(grouped: mx.array) -> mx.array:
    """Each option's label logits [rows, options, tokens] combined by log-sum-exp: the probability of any of its
    tokens."""
    return mx.logsumexp(grouped, axis=-1)


def option_probabilities(
    logits: mx.array,
    labels: LabelSet,
    temperature: float = 1.0,
    pool: Callable[[mx.array], mx.array] = log_sum_exp,
) -> mx.array:
    """Probabilities [rows, options] from label logits [rows, len(labels.flat)] whose columns follow
    labels.flat: each option's tokens combined (`pool`, from [rows, options, tokens], a missing token's logit -inf;
    by log-sum-exp unless the creator combines them otherwise), divided by `temperature`, softmax."""
    if logits.shape[-1] != len(labels.flat):
        raise ValueError(f"{logits.shape[-1]} logits for {len(labels.flat)} label tokens")
    width = max(len(group) for group in labels.tokens)
    index, valid, start = [], [], 0
    for group in labels.tokens:
        index.append([start + min(i, len(group) - 1) for i in range(width)])
        valid.append([i < len(group) for i in range(width)])
        start += len(group)
    grouped = mx.take(logits.astype(mx.float32), mx.array(index), axis=-1)  # [rows, options, width]
    grouped = mx.where(mx.array(valid), grouped, mx.array(-mx.inf, dtype=mx.float32))
    return softmax(pool(grouped), temperature)


@dataclass(frozen=True)
class CodeReadout:
    """A head of one learned row per option code, read at the position after the options are listed: the prompt
    names option i by `codes[i]`, and its probability is a softmax over hidden @ rows[:n].T at the saved
    temperature (vllm-sr d3's readout, Perplexity's)."""

    weight: mx.array  # [CODE_ROWS, hidden] fp32
    codes: tuple[str, ...]  # the codes the prompt shows, in row order
    temperature: float

    @classmethod
    def load(cls, ctx: LoadContext, model: ModelDir, codes: Sequence[str], temperature: float) -> CodeReadout:
        """readout.safetensors's "weight", checked once here: CODE_ROWS rows, a code for each option shown, a
        positive finite temperature."""
        weight = ctx.weights(model, ["readout.safetensors"]).get("weight", mx.float32)
        if weight.shape[0] != CODE_ROWS:
            raise UnsupportedModel(f"{model.path}: the readout has {weight.shape[0]} codes, not {CODE_ROWS}")
        if not codes or len(codes) > CODE_ROWS or len(set(codes)) != len(codes):
            raise UnsupportedModel(f"{model.path}: the readout needs 1 to {CODE_ROWS} distinct codes")
        if not (math.isfinite(temperature) and temperature > 0):
            raise UnsupportedModel(f"{model.path}: the saved temperature must be positive and finite")
        return cls(weight, tuple(codes), float(temperature))

    def check(self, question: str, options: int) -> None:
        """InvalidRequest unless a question's options each have a code."""
        if not 1 <= options <= len(self.codes):
            raise InvalidRequest(f"{question}: {options} options; 1 to {len(self.codes)} are supported")

    def __call__(self, row: mx.array, options: int) -> mx.array:
        """The probabilities of the first `options` options from the hidden state [hidden] at the readout."""
        logits = row.astype(mx.float32) @ self.weight[:options].T
        return mx.softmax(logits / self.temperature, axis=-1)
