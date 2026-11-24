"""Media placeholders (arch/common): a token that stands for a medium's row names it by digest and row, and a
placeholder whose medium was never encoded is refused rather than read as zeros."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from backbone_fixtures import LFM_CONV
from backbone_fixtures import lfm2_model as model
from ippon.arch import common


def test_placeholders_round_trip_and_stay_negative() -> None:
    for digest, row in ((0, 0), (2**48 - 1, 2**15 - 1), (123456789, 77)):
        token = common.media_id(digest, row)
        assert token < 0
        assert common.media_of(token) == (digest, row)
    with pytest.raises(ValueError, match="rows"):
        common.media_id(1, 2**15)


def test_a_placeholder_without_its_rows_is_refused(tmp_path: Path) -> None:
    from ippon.errors import UnsupportedModel

    m, _, _ = model(tmp_path, LFM_CONV)
    m.media = {1: mx.zeros((2, 32))}
    with pytest.raises(UnsupportedModel, match="no rows"):
        m.batch([[common.media_id(2, 0), 5, 6]], [[1]])
