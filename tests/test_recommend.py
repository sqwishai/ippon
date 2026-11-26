"""`ippon recommend`: from verified.py's builds alone, in a fixed order, a family's build for the task as
published; a smaller build is offered beside one that doesn't fit, never chosen in its place."""

from __future__ import annotations

import pytest

from ippon import names
from ippon.device import Device
from ippon.recommend import RANK, Pick, faster_here, recommend, report, start_with

MAC_8GB = Device("Apple M1", "applegpu_g13g", 8 << 30, int(5.3 * 2**30), (15, 0), True)
MAC_64GB = Device("Apple M5 Max", "applegpu_g17c", 64 << 30, 48 << 30, (26, 2), True)
MAC_M4 = Device("Apple M4 Pro", "applegpu_g16s", 48 << 30, 37 << 30, (26, 2), True)


@pytest.mark.parametrize("device", [MAC_8GB, MAC_64GB])
@pytest.mark.parametrize("task", ["routing", "images", "audio", "videos"])
def test_recommend_orders_each_familys_published_build_and_never_shrinks_one(task: str, device: Device) -> None:
    picks = recommend(task, device)
    kind = {"routing": "text"}.get(task, task)
    assert len({p.named.family for p in picks}) == len(picks)  # one build a family
    for p in picks:
        assert not any(n.low_bit for n in (p.named, *p.smaller))  # never below 8 bits
        assert kind == "text" or kind in p.named.media
        assert all(m.family == p.named.family and m.size < p.named.size for m in p.smaller)
        assert not p.smaller or p.fits is False  # offered beside a build that doesn't fit, and only then
    by_name = {p.named.name: p for p in picks}
    if task == "routing" and device is MAC_8GB:  # Kettei Core is listed, too big; d1-2B offers its 8-bit build beside
        assert (by_name["kettei-core:bf16"].fits, by_name["kettei-dot:bf16"].fits) == (False, True)
        assert [m.name for m in by_name["d1-2b:bf16"].smaller] == ["d1-2b:q8_0"]
    if device is MAC_64GB:
        assert all(p.fits and not p.smaller for p in picks)
    order = [(RANK[p.status], not p.fits, p.named.size) for p in picks]
    assert order == sorted(order)
    text = report(task, picks, device)
    usable = [p for p in picks if p.fits and p.status in ("verified", "prompts")]
    starter = [
        p for family in ("kettei-core", "kettei-dot") for p in usable if p.named.family == family
    ]  # Core, else Dot
    if usable:  # an 8 GB Mac holds no media model as published: none to start with, its smaller builds named
        first = (starter or usable)[0]
        assert f"Start with: ippon pull {(first.faster or first.named).name} " in text
    else:
        assert "Start with" not in text
        assert "smaller builds that do" in text
    if kind != "text":
        assert all(p.named.kind == "safetensors" for p in picks)  # the GGUF builds carry no media towers
    if task == "audio":
        # the family that takes clips, by its build with the speech encoder (not its q8_0, which has none)
        assert [p.named.name for p in picks] == ["d1-omni-2b:bf16"]


def test_an_8_bit_build_points_to_its_full_precision_one_where_that_is_faster() -> None:
    """Without neural accelerators an 8-bit build reads prompts 11-34% slower than the family's bf16 one: an M4 Pro is
    told so and started on bf16; an 8 GB M1, where bf16 swaps, and an M5, where both run the same, are not. No
    family's pick is such a build today (d1-2B's name loads its bf16 checkpoint), so d1-2B's Q8_0 file stands in."""
    q8, family = names.find("d1-2b:q8_0")[0], list(names.families()["d1-2b"])
    faster = faster_here(q8, family, MAC_M4)
    assert faster is not None
    assert faster.name == "d1-2b:bf16"
    pick = Pick(q8, "verified", True, (), faster)
    assert "d1-2b:bf16 (3.8 GB) fits and reads prompts faster than d1-2b:q8_0" in report("routing", [pick], MAC_M4)
    assert start_with([pick]) == faster  # pulled in its place where it leads the list
    for mac in (MAC_8GB, MAC_64GB):
        assert faster_here(q8, family, mac) is None


def test_the_build_to_start_with_is_one_this_mac_is_known_to_hold() -> None:
    """Kettei Core where it fits, Kettei Dot where it doesn't, and Dot too where this Mac didn't say what it gives a
    model: nothing unknown is taken to fit."""
    unknown = Device("Apple M2", "applegpu_g14g", None, None, (26, 2), True)
    starts = {d.name: start_with(recommend("text", d)) for d in (MAC_64GB, MAC_8GB, unknown)}
    assert {name: n.name if n else None for name, n in starts.items()} == {
        "Apple M5 Max": "kettei-core:bf16",
        "Apple M1": "kettei-dot:bf16",
        "Apple M2": "kettei-dot:bf16",
    }
