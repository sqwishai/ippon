"""Offline Ollama cache resolution, physical encodings, integrity and malformed manifests."""

from __future__ import annotations

import hashlib
import json
import socket
import struct
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from ippon import check, locate, ollama, provenance, registry
from ippon.adapter import AdapterBase
from ippon.device import Device
from ippon.errors import UnsupportedModel
from ippon.quant import QuantizedMatrix
from ippon.safetensors import tensor_header


def layer(root: Path, data: bytes, media: str, name: str = "") -> dict[str, Any]:
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    (root / "blobs").mkdir(exist_ok=True)
    (root / "blobs" / digest.replace(":", "-")).write_bytes(data)
    return {"mediaType": media, "digest": digest, "size": len(data), "name": name}


def fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OLLAMA_MODELS", str(root))
    monkeypatch.setenv("IPPON_CACHE", str(root.with_name(root.name + "-ippon-cache")))  # beside the store, never in it
    layers = [layer(root, b"{}", ollama.JSON, name) for name in ("config.json", "tokenizer.json")]
    for mode in ("nvfp4", "mxfp8"):
        q, scales = mx.quantize(mx.ones((8, 64)), mode=mode)[:2]
        name = mode + ".weight"
        values = {name: q, name + ".scale": scales}
        if mode == "nvfp4":
            values[name + ".global_scale"] = mx.array(0.03125)
        path = root / (mode + ".safetensors")
        mx.save_safetensors(str(path), values, metadata={"quant_type": mode})
        layers.append(layer(root, path.read_bytes(), ollama.TENSOR, name))
    manifest = root / "manifests/registry.ollama.ai/library/toy/latest"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schemaVersion": 2, "layers": layers}))
    return manifest


def test_import_is_read_only_preserves_bytes_and_never_contacts_a_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = fixture(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def no_network(*args: Any) -> None:
        raise AssertionError("the importer opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    build = ollama.open_build("ollama://toy")
    assert build.manifest == manifest
    weights = build.weights(build.model_dir(), None, ())
    for mode in ("nvfp4", "mxfp8"):
        q = weights.matrix(mode + ".weight", mx.bfloat16)
        assert isinstance(q, QuantizedMatrix)
        assert q.mode == mode
        assert q.scales.dtype == mx.uint8
        assert not q.global_scale_is_amax

    class Found(AdapterBase):
        name, files = "acme/toy", ()

    monkeypatch.setattr(registry, "select", lambda directory, adapter: Found())
    found = check.check("ollama://toy", device=Device("Apple M5 Pro", "applegpu_g17s", 2 << 30, 1 << 30, (27, 0), True))
    assert (found["revision"], found["formats"]) == (build.checkpoint, {"nvfp4": 1, "mxfp8": 1})
    assert [m["precision"] for m in found["memory"]] == ["auto", "bf16", "fp32"]  # as stored: never quantized again
    # a load opens the same folder, with settings given in code
    settings = {"labels": ["yes", "no"]}
    directory = locate.open_model("ollama://toy", settings=settings).directory
    assert (directory.path, directory.config, directory.source) == (manifest.parent, {}, build)
    assert (directory.json("ippon.json"), directory.quantization) == (settings, build.packing)
    built = provenance.describe("ollama://toy", directory)  # named by the build and its checkpoint, not a path
    assert (built["ollama"], built["manifest"], "path" in built) == ("ollama://toy", build.checkpoint, False)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_checkpoint_identity_ignores_metadata_layers_and_follows_the_tensor_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provenance, licence or params layer leaves the identity as it was; a different tensor blob changes it."""
    manifest = fixture(tmp_path, monkeypatch)
    first = ollama.open_build("ollama://toy").checkpoint
    value = json.loads(manifest.read_text())
    value["layers"] += [
        layer(tmp_path, b'{"schemaVersion": 2}', "application/vnd.docker.distribution.manifest.v2+json"),
        layer(tmp_path, b"Apache-2.0", "application/vnd.ollama.image.license"),
        layer(tmp_path, b"{}", "application/vnd.ollama.image.params"),
        layer(tmp_path, b"{}", ollama.JSON, "generation_config.json"),
    ]
    manifest.write_text(json.dumps(value))
    assert ollama.open_build("ollama://toy").checkpoint == first
    tensor = next(item for item in value["layers"] if item["mediaType"] == ollama.TENSOR)
    blob = tmp_path / "blobs" / tensor["digest"].replace(":", "-")
    changed = bytearray(blob.read_bytes())
    changed[-1] ^= 1
    replacement = layer(tmp_path, bytes(changed), ollama.TENSOR, tensor["name"])
    value["layers"] = [replacement if item is tensor else item for item in value["layers"]]
    manifest.write_text(json.dumps(value))
    assert ollama.open_build("ollama://toy").checkpoint != first


def test_a_manifest_list_selects_only_the_mlx_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = fixture(tmp_path, monkeypatch)
    child = layer(tmp_path, manifest.read_bytes(), "application/vnd.docker.distribution.manifest.v2+json")
    manifest.write_text(json.dumps({"manifests": [{"runner": "llamacpp"}, {**child, "runner": "mlx"}]}))
    assert len(ollama.open_build("ollama://toy:latest").blobs) == 2


@pytest.mark.parametrize(
    "fault", ["missing", "checksum", "duplicate", "digest", "asset", "gguf", "ambiguous", "nesting"]
)
def test_bad_cache_entries_fail_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    """Each a model error (`nesting`: a manifest past Python's recursion limit, a RecursionError, not a ValueError)."""
    manifest = fixture(tmp_path, monkeypatch)
    value = json.loads(manifest.read_text())
    tensor = value["layers"][-1]
    path = tmp_path / "blobs" / tensor["digest"].replace(":", "-")
    if fault == "missing":
        path.unlink()
    elif fault == "checksum":  # after a load that checked it, so the change is caught though the check is remembered
        ollama.open_build("ollama://toy")
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    elif fault == "duplicate":
        value["layers"].append(tensor)
    elif fault == "digest":
        tensor["digest"] = "../../arbitrary-file"
    elif fault == "asset":
        value["layers"][0]["name"] = "../config.json"
    elif fault == "gguf":
        value["layers"] = value["layers"][:2]
    elif fault == "ambiguous":
        value = {"manifests": [{"runner": "mlx"}, {"runner": "mlx"}]}
    manifest.write_text("[" * 100_000 if fault == "nesting" else json.dumps(value))
    with pytest.raises(UnsupportedModel):
        ollama.open_build("ollama://toy")


@pytest.mark.parametrize("name", ["../toy", "ns/../toy", "toy:..", "/tmp/toy", "toy:../../x"])
def test_model_names_cannot_escape_the_cache(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    with pytest.raises(UnsupportedModel):
        ollama.open_build("ollama://" + name)


@pytest.mark.parametrize("fault", ["metadata", "shape", "dtype", "offset", "length", "overlap", "nesting"])
def test_malformed_tensor_headers_are_model_errors(tmp_path: Path, fault: str) -> None:
    value: dict[str, Any] = {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    if fault == "metadata":
        value["__metadata__"] = []
    elif fault == "shape":
        value["x"]["shape"] = [-2]
    elif fault == "dtype":
        value["x"]["dtype"] = "F8_UNKNOWN"
    elif fault == "offset":
        value["x"]["data_offsets"] = [-1, 7]
    elif fault == "length":
        value["x"]["shape"] = [3]
    elif fault == "overlap":
        value["y"] = value["x"]
    data = json.dumps(value).encode() if fault != "nesting" else b"[" * 100_000
    path = tmp_path / "blob"
    path.write_bytes(struct.pack("<Q", len(data)) + data + b"\0" * 8)
    with pytest.raises(UnsupportedModel):
        tensor_header(path)
