"""media/resize.py against torchvision 0.29.1's own output (tests/data/media/make_resize_fixtures.py), bit for bit:
uint8 bilinear and bicubic with antialias, down, up, mixed and one axis only."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from ippon.media import resize

DATA = Path(__file__).resolve().parent / "data/media"
RESIZED = np.load(DATA / "resize.npz")
CASES = sorted(k for k in RESIZED.files if k.count(".") == 2)


@pytest.mark.parametrize("case", CASES)
def test_uint8_resizes_are_torchvisions(case: str) -> None:
    name, mode, size = case.split(".")
    height, width = map(int, size.split("x"))
    image = RESIZED[f"{name}.input"]
    got = resize.resize_frames(image[None], height, width, cast(resize.Mode, mode))[0]
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, RESIZED[case])


def test_the_same_size_is_a_copy() -> None:
    image = RESIZED["odd.input"]
    height, width = image.shape[:2]
    same = resize.resize_frames(image[None], height, width, "bilinear")[0]
    np.testing.assert_array_equal(same, image)
    assert not np.shares_memory(same, image)  # a caller may write into it


def test_wrong_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="uint8"):
        resize.resize_frames(np.zeros((1, 4, 4, 3), np.float32), 2, 2, "bilinear")
    with pytest.raises(ValueError, match="cannot resize"):
        resize.resize_frames(np.zeros((1, 4, 4, 3), np.uint8), 0, 2, "bicubic")
