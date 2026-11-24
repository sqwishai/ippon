"""Tensors and plain values from a PyTorch `.pt` file, read without running any of it.

A `.pt` file is a zip whose `data.pkl` is a pickle. Unpickling runs whatever the pickle names, so ippon never unpickles
model weights (weights.py refuses `.bin`, `.pt` and `.pth`). A few creators publish only a small head this way (Kev's
`head.pt`: two linear maps and its settings). This reader takes exactly the layout `torch.save` writes for a dict of
tensors and plain values: the pickle may name `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2` and the
float storages, each answered by ippon's own code here, and anything else it names is refused before it runs. Tensors
come back as float32 NumPy arrays, read from the zip's raw storage records.
"""

from __future__ import annotations

import collections
import io
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .errors import UnsupportedModel

STORAGES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64}


class _Storage(str):
    """A storage kind the pickle names, as the persistent ids it rebuilds tensors from carry it."""


def read(path: Path) -> dict[str, Any]:
    """The dict `torch.save` wrote at `path`: its plain values as written, its tensors as float32 arrays."""
    with zipfile.ZipFile(path) as z:
        pickles = [n for n in z.namelist() if n.endswith("/data.pkl")]
        if len(pickles) != 1:
            raise UnsupportedModel(f"{path}: not a torch.save file (no single data.pkl)")
        root = pickles[0].removesuffix("/data.pkl")

        def rebuild(
            storage: tuple[Any, ...], offset: int, size: tuple[int, ...], stride: tuple[int, ...], *_: Any
        ) -> Any:
            _, kind, key, _, numel = storage
            raw = np.frombuffer(z.read(f"{root}/data/{key}"), dtype=STORAGES[kind])
            if len(raw) != numel:
                raise UnsupportedModel(f"{path}: storage {key} holds {len(raw)} values, not {numel}")
            count = int(np.prod(size)) if size else 1
            dense = tuple(int(np.prod(size[i + 1 :])) for i in range(len(size)))
            if tuple(stride) != dense:
                raise UnsupportedModel(f"{path}: a tensor of storage {key} is not stored densely")
            return raw[offset : offset + count].reshape(size).astype(np.float32)

        allowed: dict[tuple[str, str], Any] = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2"): rebuild,
            **{("torch", kind): _Storage(kind) for kind in STORAGES},
        }

        class Restricted(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                if (module, name) not in allowed:
                    raise UnsupportedModel(f"{path}: refused to load {module}.{name} from a pickle")
                return allowed[module, name]

            def persistent_load(self, pid: Any) -> Any:
                if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):
                    raise UnsupportedModel(f"{path}: unexpected persistent id {pid!r}")
                return pid

        value = Restricted(io.BytesIO(z.read(pickles[0]))).load()
    if not isinstance(value, dict):
        raise UnsupportedModel(f"{path}: holds a {type(value).__name__}, not a dict")
    return value
