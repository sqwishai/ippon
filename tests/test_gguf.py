"""GGUF builds: the header read without loading, K-quants refused before MLX would unpack them, an LFM3 build's
tensors renamed to the creator's and computing as its checkpoint does, unknown tensors and disagreeing settings
refused, local files recognized by their header, and admission sizing a build as MLX holds it."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

import lfm3_numpy
from backbone_fixtures import LFM3_LAYERS
from gguf_writer import write
from ippon import hashes, hub, locate, registry
from ippon.adapter import LoadContext
from ippon.adapters import clef
from ippon.arch import lfm3
from ippon.errors import UnsupportedModel
from ippon.gguf import builds, names
from ippon.gguf.header import read_header
from ippon.gguf.reader import host_rows, load
from ippon.gguf.source import Gguf, bound
from ippon.host_rows import HostRows
from ippon.model_dir import ModelDir, config_of
from ippon.names import parse
from ippon.quant import QuantizedMatrix, rows

LFM3_NAMES = {  # the creator's LFM3 names to llama.cpp's, written from llama.cpp b11960's LFM3 tensor map
    "embed_tokens": "token_embd",
    "embedding_norm": "output_norm",
    "operator_norm": "attn_norm",
    "post_operator_norm": "post_attention_norm",
    "ffn_norm": "ffn_norm",
    "post_ffn_norm": "post_ffw_norm",
    "feed_forward.w1": "ffn_gate",
    "feed_forward.w3": "ffn_up",
    "feed_forward.w2": "ffn_down",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.gate_proj": "attn_gate",
    "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm",
    "self_attn.k_norm": "attn_k_norm",
    "conv.in_proj": "shortconv.in_proj",
    "conv.conv": "shortconv.conv",
    "conv.out_proj": "shortconv.out_proj",
}
TURNING = [1.0, 1.0, 1e30, 1e30]  # the tiny model's full layers: 2 of 4 angle pairs turn


def gguf_name(name: str) -> str:
    base = name.removesuffix(".weight")
    if base.startswith("layers."):
        _, i, rest = base.split(".", 2)
        return f"blk.{i}.{LFM3_NAMES[rest]}.weight"
    return LFM3_NAMES[base] + ".weight"


def tiny_gguf(
    path: Path,
    factors: list[float] = TURNING,
    extra: dict[str, tuple[np.ndarray, str]] | None = None,
    changes: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """A tiny d1 on LFM3 written as llama.cpp b11960 writes d1-2B: Q8_0 where rows allow, convolutions [channels,
    taps], the full layers' RoPE as `factors` per angle pair (rope_freqs); its config and the values it stores."""
    c = lfm3_numpy.tiny_config(LFM3_LAYERS)
    w = lfm3_numpy.tiny_weights(c)
    tensors = {}
    for name, x in w.items():
        flat = x.reshape(x.shape[0], x.shape[-1]) if name.endswith("conv.conv.weight") else x
        quantized = flat.ndim == 2 and flat.shape[-1] % 32 == 0 and "conv.conv" not in name
        tensors[gguf_name(name)] = (flat, "Q8_0" if quantized else "F32")
    tensors["rope_freqs.weight"] = (np.array(factors, np.float32), "F32")
    types = c["layer_types"]
    meta = {
        "general.architecture": "lfm3",
        "lfm3.block_count": len(types),
        "lfm3.embedding_length": c["hidden_size"],
        "lfm3.feed_forward_length": c["intermediate_size"],
        "lfm3.attention.head_count": c["num_attention_heads"],
        "lfm3.attention.head_count_kv": [0 if t == "conv" else c["num_key_value_heads"] for t in types],
        "lfm3.attention.key_length": c["head_dim"],
        "lfm3.attention.layer_norm_rms_epsilon": float(c["norm_eps"]),
        "lfm3.attention.sliding_window": c["sliding_window"],
        "lfm3.attention.sliding_window_pattern": [t == "sliding_attention" for t in types],
        "lfm3.rope.freq_base": 1_000_000.0,
        "lfm3.rope.freq_base_swa": 10_000.0,
        "lfm3.shortconv.l_cache": c["conv_L_cache"],
        "lfm3.attention.causal": False,
        "lfm3.decision.type": "lfm3-d1",
        **{f"lfm3.decision.temperature.{k}": 1.5 for k in ("noul", "choice", "score")},
        "tokenizer.ggml.tokens": ["a", "b"],  # skipped by the reader
    }
    stored = write(path, meta | (changes or {}), tensors | (extra or {}))
    return c, {name: stored[gguf_name(name)].reshape(x.shape) for name, x in w.items()}


def base_dir(tmp_path: Path, text: dict[str, Any]) -> Path:
    """The creator's small files of the tiny d1: config.json around `text` and calibration.json."""
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    config = {"model_type": "lfm3_vl", "architectures": ["D1Model"], "d1_attention": "set", "text_config": text}
    (base / "config.json").write_text(json.dumps(config))
    (base / "calibration.json").write_text(json.dumps({"default": 1.5, "temperatures": {}}))
    return base


def build_dir(tmp_path: Path, text: dict[str, Any]) -> ModelDir:
    base = base_dir(tmp_path, text)
    build = builds.Build("org/tiny-GGUF", "a" * 40, "m.gguf", "b" * 64, "", "org/tiny", "c" * 40)
    model = ModelDir(base, config_of(base), Gguf(base, tmp_path / "m.gguf", build))
    reads = registry.select(model).gguf_rules  # as a load reads it once its adapter is chosen (`bound`)
    return ModelDir(base, config_of(base), Gguf(base, tmp_path / "m.gguf", build, reads=reads))


def source(model: ModelDir) -> Gguf:
    assert isinstance(model.source, Gguf)
    return model.source


def checked(model: ModelDir) -> None:
    """`model`'s GGUF metadata against its creator's config, as a load checks it after choosing its adapter."""
    bound(model, read_header(source(model).path), registry.select(model))


@pytest.mark.parametrize("kind", ["Q4_K", "Q5_K", "Q6_K"])
def test_k_quants_read_as_ggml_unpacks_them(tmp_path: Path, kind: str) -> None:
    """Against ggml's dequantize_row loops (gguf_writer.dequantize): Q6_K reads as those values in fp16; Q4_K and
    Q5_K, kept in MLX's affine groups, dequantize to them within their scales' fp16 rounding."""
    from gguf_writer import dequantize

    write(tmp_path / "m.gguf", {"general.architecture": "test"}, {"m.weight": (np.zeros((6, 512), np.float32), kind)})
    header = read_header(tmp_path / "m.gguf")
    t = header.tensors["m.weight"]
    expected = dequantize((tmp_path / "m.gguf").read_bytes()[t.offset : t.offset + t.nbytes], kind).reshape(6, 512)
    mine = load(header)
    if kind == "Q6_K":
        assert mine["m.weight"].dtype == mx.float16
        np.testing.assert_array_equal(np.array(mine["m.weight"]), expected.astype(np.float16))
        return
    bits = 4 if kind == "Q4_K" else 5
    scales, biases = mine["m.scales"].astype(mx.float32), mine["m.biases"].astype(mx.float32)
    got = np.array(mx.dequantize(mine["m.weight"], scales, biases, group_size=32, bits=bits))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-3 * np.abs(expected).max())


def test_a_file_cut_short_is_refused(tmp_path: Path) -> None:
    tiny_gguf(tmp_path / "m.gguf")
    data = (tmp_path / "m.gguf").read_bytes()
    (tmp_path / "m.gguf").write_bytes(data[: len(data) - 64])
    with pytest.raises(UnsupportedModel, match="runs past the end of the file"):
        load(read_header(tmp_path / "m.gguf"))


def test_an_lfm3_build_computes_as_its_checkpoint_and_holds_its_rope_to_the_config(tmp_path: Path) -> None:
    c, stored = tiny_gguf(tmp_path / "m.gguf")
    model = build_dir(tmp_path, c)
    checked(model)
    weights = LoadContext().weights(model)
    w1 = "model.language_model.layers.0.feed_forward.w1.weight"
    assert isinstance(LoadContext().weights(model).matrix(w1), QuantizedMatrix)
    m = lfm3.load(str(model.path), model.config, dtype=mx.float32, weights=weights)
    assert isinstance(m.embed, HostRows)  # read from the file row by row, the forward pass below included
    ids = [int(t) for t in np.random.default_rng(5).integers(0, 64, 19)]
    got = m.prefill(ids, m.new_cache(len(ids)), list(range(len(ids))))
    np.testing.assert_allclose(np.array(got), lfm3_numpy.forward(c, stored, ids), atol=2e-4, rtol=2e-4)

    # admission counts the bytes the build holds once loaded, Q8_0 matrices as MLX keeps them
    width = {"U32": 4, "F16": 2, "F32": 4}
    counted = sum(math.prod(t.shape) * width[t.dtype] for t in model.source.stored(model))
    assert counted == sum(a.nbytes for a in load(read_header(tmp_path / "m.gguf")).values())

    tiny_gguf(tmp_path / "m.gguf", [1.0, 1.0, 1.0, 1.0])  # every pair turning: another RoPE
    with pytest.raises(UnsupportedModel, match="rope_freqs are not 2 of 4 pairs turning"):
        checked(model)


def test_a_tensor_no_rule_knows_is_refused(tmp_path: Path) -> None:
    c, _ = tiny_gguf(tmp_path / "m.gguf", extra={"blk.0.surprise.weight": (np.zeros(32, np.float32), "F32")})
    with pytest.raises(UnsupportedModel, match=r"'blk\.0\.surprise\.weight' is not one ippon knows"):
        LoadContext().weights(build_dir(tmp_path, c))


def test_settings_that_differ_from_the_makers_are_refused(tmp_path: Path) -> None:
    c, _ = tiny_gguf(tmp_path / "m.gguf", changes={"lfm3.rope.freq_base": 5000.0, "lfm3.attention.causal": True})
    model = build_dir(tmp_path, c)
    with pytest.raises(UnsupportedModel, match=r"rope.freq_base 5000.0 against 1000000.0; its attention is not d1's"):
        checked(model)
    tiny_gguf(tmp_path / "m.gguf")
    checked(model)


def test_a_local_file_is_recognized_by_its_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, _ = tiny_gguf(tmp_path / "m.gguf")
    base = base_dir(tmp_path, c)
    digest = read_header(tmp_path / "m.gguf").digest
    build = builds.Build("org/tiny-GGUF", "a" * 40, "tiny-Q8_0.gguf", "b" * 64, digest, "org/tiny", "c" * 40)
    monkeypatch.setattr(builds, "BUILDS", (build,))
    asked: list[tuple[str, tuple[str, ...]]] = []

    def fetch(model: str, patterns: Any) -> Path:
        asked.append((model, tuple(patterns)))
        return base

    monkeypatch.setattr(hub, "fetch", fetch)
    model = locate.open_model(tmp_path / "m.gguf").directory
    assert source(model).path == tmp_path / "m.gguf"
    assert source(model).reads is registry.select(model).gguf_rules  # read by d1's rules, its adapter's
    assert model.config["text_config"] == c
    assert asked == [("org/tiny@" + "c" * 40, builds.SMALL_FILES)]  # the creator's small files, never weights
    tiny_gguf(tmp_path / "m.gguf", changes={"lfm3.rope.freq_base": 5000.0})
    with pytest.raises(UnsupportedModel, match="a GGUF file ippon doesn't know"):
        locate.open_model(tmp_path / "m.gguf")


def test_an_mmproj_beside_the_file_gives_the_picture_and_audio_parts_under_the_makers_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, _ = tiny_gguf(tmp_path / "m.gguf")
    base = base_dir(tmp_path, c)
    rng = np.random.default_rng(0)
    frame0, frame1 = (rng.standard_normal((8, 3, 4, 4)).astype(np.float32) for _ in range(2))
    stored = write(
        tmp_path / "mmproj.gguf",
        {"general.architecture": "clip", "general.type": "mmproj"},
        {
            "v.patch_embd.weight": (frame0, "F32"),
            "v.patch_embd.weight.1": (frame1, "F32"),
            "v.blk.0.ffn_up.weight": (rng.standard_normal((64, 32)).astype(np.float32), "Q8_0"),
            "a.conv1d.0.bias": (rng.standard_normal((8, 1, 1)).astype(np.float32), "F32"),
            "a.pre_encode.out.weight": (rng.standard_normal((16, 64)).astype(np.float32), "Q8_0"),
        },
    )
    build = builds.Build(
        "org/tiny-GGUF", "a" * 40, "tiny-Q8_0.gguf", "b" * 64, read_header(tmp_path / "m.gguf").digest,
        "org/tiny", "c" * 40, "mmproj.gguf", "d" * 64, read_header(tmp_path / "mmproj.gguf").digest,
    )  # fmt: skip
    monkeypatch.setattr(builds, "BUILDS", (build,))
    monkeypatch.setattr(hub, "fetch", lambda model, patterns: base)
    model = locate.open_model(tmp_path / "m.gguf").directory
    assert source(model).mmproj == tmp_path / "mmproj.gguf"
    assert source(model).media_names(model, LoadContext().weights(model)) == {
        "model.vision_tower.patchifier.weight",
        "model.vision_tower.layers.0.feed_forward.w3.weight",
        "model.audio_encoder.downsample.conv_in.bias",
        "model.audio_encoder.downsample.lin_out.weight",
    }
    media = LoadContext().media_weights(model)
    # the two frames joined as the checkpoint's patchifier [out, channels * frames * h * w], channels outermost
    joined = np.stack([frame0, frame1], axis=2).reshape(8, -1)
    np.testing.assert_array_equal(np.array(media.get("model.vision_tower.patchifier.weight")), joined)
    np.testing.assert_array_equal(
        np.array(media.get("model.audio_encoder.downsample.conv_in.bias")), stored["a.conv1d.0.bias"].reshape(8)
    )
    lin_out = np.array(media.get("model.audio_encoder.downsample.lin_out.weight"))
    np.testing.assert_allclose(lin_out, stored["a.pre_encode.out.weight"].reshape(16, 64, 1), atol=1e-6)
    # without the mmproj beside it, a local file reads text only, as llama.cpp does without --mmproj
    (tmp_path / "mmproj.gguf").unlink()
    model = locate.open_model(tmp_path / "m.gguf").directory
    assert source(model).mmproj is None
    assert source(model).media_names(model, LoadContext().weights(model)) == set()


def test_builds_are_named_by_repository_and_quantization() -> None:
    assert parse("LiquidAI/d1-2B-GGUF:Q8_0").form == "gguf"
    assert parse("model.gguf").form == "gguf-file"
    assert parse("LiquidAI/d1-2B").form == "hub"
    b = builds.find("liquidai/d1-2b-gguf:q8_0")
    assert (b.repo, b.quant, b.base) == ("LiquidAI/d1-2B-GGUF", "Q8_0", "LiquidAI/d1-2B")
    assert builds.find(f"LiquidAI/d1-2B-GGUF@{b.revision}:Q8_0") == b
    with pytest.raises(UnsupportedModel, match="LiquidAI/d1-2B-GGUF:Q8_0"):
        builds.find("LiquidAI/d1-2B-GGUF:Q2_K")


def stored_tiled(x: np.ndarray, n_k: int, n_v: int, block: int, axis: int = 0) -> np.ndarray:
    """x's n_v value heads of `block` rows (or columns) as llama.cpp stores them: tiled, GGUF head g holding the
    checkpoint's head (g mod n_k) * (n_v / n_k) + g div n_k."""
    r = n_v // n_k
    heads = np.split(x, n_v, axis=axis)
    return np.concatenate([heads[(g % n_k) * r + g // n_k] for g in range(n_v)], axis=axis)


def test_value_heads_go_back_to_the_checkpoints_order() -> None:
    rng = np.random.default_rng(0)
    n_k, n_v, block, offset = 2, 6, 3, 4
    original = rng.standard_normal((offset + n_v * block, 5)).astype(np.float32)
    stored = np.concatenate([original[:offset], stored_tiled(original[offset:], n_k, n_v, block)])
    undo = names.heads_rows(n_k, n_v, block, offset)
    np.testing.assert_array_equal(np.array(undo(mx.array(stored), "dense", 0)), original)
    assert not np.array_equal(stored, original)


@pytest.mark.parametrize("bits", [8, 4])
def test_a_quantized_matrixs_column_heads_move_as_whole_groups(bits: int) -> None:
    """The output projection's columns: undone on the packed weight, scales and biases, the same as on the values."""
    rng = np.random.default_rng(1)
    n_k, n_v, block = 2, 4, 64
    original = mx.array(rng.standard_normal((8, n_v * block)).astype(np.float32))
    stored = mx.array(stored_tiled(np.array(original), n_k, n_v, block, axis=1))
    w, s, b = mx.quantize(stored, group_size=32, bits=bits)
    undo = names.heads_columns(n_k, n_v, block)
    back = mx.dequantize(undo(w, "weight", bits), undo(s, "scales", bits), undo(b, "biases", bits), 32, bits)
    want = mx.dequantize(*mx.quantize(original, group_size=32, bits=bits), group_size=32, bits=bits)
    np.testing.assert_array_equal(np.array(back), np.array(want))


def test_clefs_applied_scalars_become_the_parameters_its_head_applies() -> None:
    prior, joint, gate = 2.296875, 5.5, -2.0  # joint past the cap of ln 100
    cap = np.log(100.0)
    applied = mx.array([np.exp(min(prior, cap)), np.exp(min(joint, cap)), 1 / (1 + np.exp(-gate))], dtype=mx.float32)
    got = {k: float(np.array(v)) for k, v in clef.gguf_scales(applied).items()}
    assert np.exp(min(got["prior_logit_scale"], cap)) == pytest.approx(np.exp(prior), rel=1e-6)
    assert np.exp(min(got["joint_logit_scale"], cap)) == pytest.approx(100.0, rel=1e-6)
    assert 1 / (1 + np.exp(-got["residual_gate"])) == pytest.approx(1 / (1 + np.exp(-gate)), rel=1e-6)


@pytest.mark.parametrize("kind", ["Q8_0", "Q4_0"])
def test_quantized_tensors_read_exactly_as_stored_and_as_mlx_reads_them(tmp_path: Path, kind: str) -> None:

    rng = np.random.default_rng(5)
    x = rng.standard_normal((6, 96)).astype(np.float32)
    stored = write(tmp_path / "m.gguf", {"general.architecture": "test"}, {"m.weight": (x, kind), "v": (x[0], "F32")})
    mine = load(read_header(tmp_path / "m.gguf"))
    bits = 8 if kind == "Q8_0" else 4
    scales, biases = mine["m.scales"].astype(mx.float32), mine["m.biases"].astype(mx.float32)  # exact, as read
    got = mx.dequantize(mine["m.weight"], scales, biases, group_size=32, bits=bits)
    np.testing.assert_array_equal(np.array(got.astype(mx.float32)), stored["m.weight"])
    theirs = mx.load(str(tmp_path / "m.gguf"))
    assert isinstance(theirs, dict)
    for k in ("m.weight", "m.scales", "m.biases", "v"):
        assert mine[k].dtype == theirs[k].dtype
        assert mx.array_equal(mine[k], theirs[k]).item(), k
    header = read_header(tmp_path / "m.gguf")  # and row by row from the file, as a vocabulary table is read
    host = host_rows(header, header.tensors["m.weight"], mx.float16)
    assert host is not None
    whole = QuantizedMatrix(mine["m.weight"], mine["m.scales"], mine["m.biases"], bits, 32).astype(mx.float16)
    ids = mx.array([[4, 0], [5, 4]])
    assert mx.array_equal(rows(host, ids), rows(whole, ids)).item()


def test_a_file_is_hashed_once_per_version_and_a_rewrite_keeping_its_size_and_time_is_hashed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import os

    monkeypatch.setenv("IPPON_CACHE", str(tmp_path / "cache"))
    blob = tmp_path / "blob"
    blob.write_bytes(b"gguf bytes" * 1000)
    sha = hashlib.sha256(blob.read_bytes()).hexdigest()
    assert not hashes.verified(blob, "0" * 64)
    assert hashes.verified(blob, sha)
    before = blob.stat()
    blob.write_bytes(b"GGUF BYTES" * 1000)  # the same size ...
    os.utime(blob, ns=(before.st_atime_ns, before.st_mtime_ns))  # ... and modification time, other content
    assert not hashes.verified(blob, sha)
    for damaged in ("null", "[]", "{"):  # a record that isn't one: hashed afresh, never a crash
        (tmp_path / "cache" / "hashes.json").write_text(damaged)
        assert hashes.verified(blob, hashlib.sha256(blob.read_bytes()).hexdigest())


@pytest.mark.parametrize(
    ("name", "body"),
    [
        # a metadata string that claims 4 EB: read as is, it asked for that much memory, and the process ended
        ("huge-string", struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2**62) + b"abc"),
        # an alignment of 0, which the tensors' offsets are divided by
        ("zero-align", struct.pack("<IQQQ", 3, 0, 1, 17) + b"general.alignment" + struct.pack("<II", 4, 0)),
        # arrays of arrays 100,000 deep: read one call deeper each, they overflowed the stack (an abort)
        (
            "deep-arrays",
            struct.pack("<IQQQ", 3, 0, 1, 1) + b"a" + struct.pack("<I", 9) + struct.pack("<IQ", 9, 1) * 100_000,
        ),
        # an array claiming 2^60 bools in a file of a few bytes: stepping through them never ended
        ("endless-array", struct.pack("<IQQQ", 3, 0, 1, 1) + b"a" + struct.pack("<IIQ", 9, 7, 2**60) + b"\0"),
    ],
)
def test_a_crafted_header_is_refused_not_a_crash(tmp_path: Path, name: str, body: bytes) -> None:
    path = tmp_path / f"{name}.gguf"
    path.write_bytes(b"GGUF" + body)
    with pytest.raises(UnsupportedModel):
        read_header(path)
