"""ptfile reads what torch.save writes for a dict of tensors and plain values, and refuses any other callable before it
can run (a pickle naming os.system must never reach it)."""

from __future__ import annotations

import collections
import io
import pickle
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ippon import downloads, ptfile
from ippon.errors import UnsupportedModel


class Named:
    """Pickled as `module.name`, the reference torch.save writes, without importing it."""

    def __init__(self, module: str, name: str):
        self.module, self.name = module, name

    def __call__(self, *args: Any) -> None:  # pickle reduces only to callables; this one is never called
        raise AssertionError("never called while writing")


class Tensor:
    def __init__(self, key: str, array: np.ndarray):
        self.key, self.array = key, array


PurePickler: Any = pickle._Pickler  # the pure-Python pickler, whose save_global and save a subclass can override


class TorchLike(PurePickler):  # type: ignore[misc]
    def save_global(self, obj: Any, name: Any = None) -> None:
        if isinstance(obj, Named):
            self.write(pickle.GLOBAL + f"{obj.module}\n{obj.name}\n".encode())
            self.memoize(obj)
            return
        super().save_global(obj, name)

    def save(self, obj: Any, save_persistent_id: bool = True) -> None:
        if isinstance(obj, Named):  # a name inside a value (the storage kind in a persistent id), written as a global
            self.save_global(obj)
            return
        super().save(obj, save_persistent_id)

    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, Tensor):  # torch's tensor reduce: _rebuild_tensor_v2(storage, offset, size, stride, ...)
            size = obj.array.shape
            stride = tuple(int(np.prod(size[i + 1 :])) for i in range(len(size)))
            storage = ("storage", Named("torch", "FloatStorage"), obj.key, "cpu", obj.array.size)
            return Named("torch._utils", "_rebuild_tensor_v2"), (
                storage,
                0,
                size,
                stride,
                False,
                collections.OrderedDict(),
            )
        return NotImplemented

    def persistent_id(self, obj: Any) -> Any:
        return obj if isinstance(obj, tuple) and obj and obj[0] == "storage" else None


def save(path: Path, value: dict[str, Any], tensors: dict[str, np.ndarray]) -> None:
    data = io.BytesIO()
    TorchLike(data, protocol=2).dump(value)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("head/data.pkl", data.getvalue())
        for key, array in tensors.items():
            z.writestr(f"head/data/{key}", array.astype(np.float32).tobytes())


def test_reads_a_dict_of_tensors_and_values_as_torch_save_writes_it(tmp_path: Path) -> None:
    q = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    b = np.array([0.5, -1.25, 3.0], dtype=np.float32)
    path = tmp_path / "head.pt"
    save(
        path,
        {
            "head": collections.OrderedDict([("q.weight", Tensor("0", q)), ("q.bias", Tensor("1", b))]),
            "temperature": 2.35,
        },
        {"0": q, "1": b},
    )
    got = ptfile.read(path)
    assert got["temperature"] == 2.35
    assert np.array_equal(got["head"]["q.weight"], q)
    assert np.array_equal(got["head"]["q.bias"], b)


def test_a_pickle_naming_anything_else_is_refused_before_it_runs(tmp_path: Path) -> None:
    path = tmp_path / "head.pt"
    payload = (
        pickle.GLOBAL + b"os\nsystem\n" + pickle.MARK + pickle.UNICODE + b"touch pwned\n" + pickle.TUPLE + pickle.REDUCE
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("head/data.pkl", pickle.PROTO + bytes([2]) + payload + pickle.STOP)
    with pytest.raises(UnsupportedModel, match=r"refused to load os\.system"):
        ptfile.read(path)
    assert not (Path.cwd() / "pwned").exists()


def test_only_a_pickle_an_adapter_names_exactly_is_downloaded() -> None:
    repo = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename=n) for n in ("head.pt", "evil.pt", "model.bin", "adapter_model.safetensors")
        ]
    )
    named = [f.rfilename for f in downloads.listing(repo, ("head.pt", "*.pt", "*.safetensors"))]
    assert named == ["head.pt", "adapter_model.safetensors"]  # never evil.pt, which only a wildcard matched
    assert "head.pt" not in [f.rfilename for f in downloads.listing(repo, ("*.pt",))]
