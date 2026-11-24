"""Qwen3.5 reading image rows (pplx-decider flash v1.2's and Clef-Flash's images) against the float64 numpy model at
the (t, h, w) positions transformers' get_rope_index gives them (qwen35_numpy.rope_index, held to it by
test_media_qwen.py): reading both ways in a padded batch, and causally however the runtime splits the prompt (chunks
that start inside an image or end with one, branches after a shared image prefix, prefixes kept in memory and on
disk). Text alone is read as before, bit for bit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from helpers import tokens
from ippon.arch import qwen35
from ippon.arch.common import MediaRows, media_id
from ippon.plan import Prompt, Readout, plan
from ippon.prefixes import Prefixes
from ippon.runtime import run
from qwen35_numpy import forward, rope_index, tiny_config, tiny_weights

LAYERS = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
A, B = 0xA1, 0xB2  # two images' digests
GRIDS = {A: (1, 2, 3), B: (1, 3, 2)}  # in rows (merged patches): [frames, block rows, block columns]


def setup(
    tmp_path: Path, causal: bool
) -> tuple[qwen35.Qwen35, dict[str, Any], dict[str, np.ndarray], dict[int, np.ndarray]]:
    """A tiny Qwen3.5 whose 8 rotary frequency pairs split across all three axes and both section limits (mrope_section
    [3, 2, 1]: h takes pairs 1 and 4, w pair 2, t the rest), with channels past the rotary ones, and two images."""
    config = tiny_config(LAYERS) | {"head_dim": 32}
    config["rope_parameters"] = config["rope_parameters"] | {"mrope_section": [3, 2, 1]}
    weights = tiny_weights(config)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"model." + k: mx.array(v) for k, v in weights.items()})
    m = qwen35.load(str(tmp_path), config, causal=causal, dtype=mx.float32)
    rng = np.random.default_rng(5)
    images = {d: rng.standard_normal((*g, config["hidden_size"])).astype(np.float32) for d, g in GRIDS.items()}
    m.media = {d: mx.array(x) for d, x in images.items()}
    return m, config, weights, images


def image(digest: int) -> list[int]:
    t, h, w = GRIDS[digest]
    return [media_id(digest, row) for row in range(t * h * w)]


def expected(
    config: dict[str, Any], weights: dict[str, np.ndarray], ids: list[int], images: dict[int, np.ndarray], causal: bool
) -> np.ndarray:
    """The numpy model's rows of `ids`: an image's rows where its placeholders are, at get_rope_index's positions."""
    table = weights["embed_tokens.weight"]
    rows, starts, at = [], [], 0
    while at < len(ids):
        if ids[at] >= 0:
            rows.append(table[ids[at]])
            at += 1
            continue
        digest = next(d for d in GRIDS if ids[at] == image(d)[0])
        t, h, w = GRIDS[digest]
        starts.append((at, (t, 2 * h, 2 * w)))  # rope_index takes the grid in patches
        rows.extend(images[digest].reshape(-1, images[digest].shape[-1]))
        at += t * h * w
    positions, _ = rope_index(len(ids), starts)
    return forward(config, weights, [], causal=causal, embeds=np.stack(rows), positions=positions)


def every_row(ids: list[int], marks: tuple[int, ...] = ()) -> Prompt:
    return Prompt(tuple(ids), tuple(Readout("q", i) for i in range(len(ids))), marks)


def test_both_ways_in_a_padded_batch(tmp_path: Path) -> None:
    m, config, weights, images = setup(tmp_path, causal=False)
    prompts = [
        tokens(5, 1) + image(A) + tokens(4, 2) + image(B) + tokens(6, 3),
        tokens(3, 4) + image(B) + tokens(2, 5),  # shorter: padded
        tokens(9, 6),  # text alone in the same batch
    ]
    got = m.batch(prompts, [list(range(len(p))) for p in prompts])
    for g, ids in zip(got, prompts, strict=True):
        np.testing.assert_allclose(np.array(g), expected(config, weights, ids, images, False), atol=2e-4, rtol=2e-4)


def test_causally_however_the_prompt_is_split(tmp_path: Path) -> None:
    """A lone prompt in chunks of 5 (two start at an image's last row), then grown and read again from the prefix
    kept after both images, and two questions after the images as branches whose head
    reads the end of an image: every row the numpy model's. Images are encoded only when a pass reads them
    (MediaRows), so the grown prompt, resumed past both, encodes none."""
    m, config, weights, images = setup(tmp_path, causal=True)
    prompt = tokens(5, 1) + image(A) + tokens(4, 2) + image(B) + tokens(6, 3)  # A at 5..10, B at 15..20
    store = Prefixes(1 << 30)
    encoded: list[set[int]] = []

    def encode(digests: set[int]) -> dict[int, mx.array]:
        encoded.append(digests)
        return {d: mx.array(images[d]) for d in digests}

    m.media = MediaRows({}, encode)
    (got,) = run(m, plan([every_row(prompt, marks=(23,))]), chunk=5, prefixes=store)
    np.testing.assert_allclose(np.array(got), expected(config, weights, prompt, images, True), atol=2e-4, rtol=2e-4)

    assert set().union(*encoded) == {A, B}
    calls = len(encoded)
    grown = prompt + tokens(4, 7)
    want = expected(config, weights, grown, images, True)
    (again,) = run(m, plan([every_row(grown, marks=(27,))]), chunk=5, prefixes=store)
    assert store.stats.hits == 1
    assert store.stats.tokens_reused == 23
    assert len(encoded) == calls  # resumed past both images, so neither was encoded again
    np.testing.assert_allclose(np.array(again), want, atol=2e-4, rtol=2e-4)

    tails = [tokens(3, 8), tokens(5, 9)]
    asked = [Prompt(tuple(prompt + t), (Readout("q", len(prompt) + len(t) - 1),)) for t in tails]
    for h, t in zip(run(m, plan(asked), chunk=16), tails, strict=True):  # the head: rows 16.., B's last five
        want = expected(config, weights, prompt + t, images, True)[-1:]
        np.testing.assert_allclose(np.array(h), want, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("causal", [True, False])
def test_text_alone_reads_as_before(tmp_path: Path, causal: bool) -> None:
    """With images given to the pass but none in its prompt, the rows are exactly those of a pass without them."""
    m, _, _, _ = setup(tmp_path, causal=causal)
    ids = tokens(17, 3)
    with_media = np.array(m.batch([ids], [list(range(17))])[0])
    m.media = None
    plain = np.array(m.batch([ids], [list(range(17))])[0])
    np.testing.assert_array_equal(with_media, plain)
