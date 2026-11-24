from __future__ import annotations

import pytest

from ippon.plan import Prompt, Readout, chunks, plan, shared_prefix
from ippon.runtime import PASS_TOKENS, pass_groups


def prompt(*tokens: int, at: int = -1, question: str = "q") -> Prompt:
    return Prompt(tokens, (Readout(question, at % len(tokens)),))


def test_prompts_share_their_common_prefix_short_of_their_own_last_token_and_readouts() -> None:
    """The prefix is what every prompt starts with, less what would leave a prompt no token of its own or a readout
    outside its tail (readout positions count from the tail); without sharing every prompt runs whole."""
    p = plan([prompt(1, 2, 3, 4, 9), prompt(1, 2, 3, 5, 9), prompt(1, 2, 3, 6, 7, 9)])
    assert p.prefix == (1, 2, 3)
    assert [t.tokens for t in p.tails] == [(4, 9), (5, 9), (6, 7, 9)]
    assert [t.readouts[0].position for t in p.tails] == [1, 1, 2]
    assert [t.tokens for t in plan([prompt(1, 2, 3), prompt(1, 2, 3)]).tails] == [(3,), (3,)]  # the same prompt twice
    assert plan([prompt(1, 2), prompt(7, 2)]).prefix == ()  # they differ at once
    joint = Prompt((1, 2, 3, 4, 5, 6), (Readout("a", 2), Readout("b", 5)))
    assert shared_prefix([joint]) == 2
    (tail,) = plan([joint]).tails
    assert tail.tokens == (3, 4, 5, 6)
    assert [(r.question, r.position) for r in tail.readouts] == [("a", 0), ("b", 3)]
    whole = plan([prompt(1, 2, 3), prompt(1, 2, 4)], share=False)
    assert whole.prefix == ()
    assert [t.tokens for t in whole.tails] == [(1, 2, 3), (1, 2, 4)]


@pytest.mark.parametrize(
    ("tokens", "readouts", "message"),
    [
        ((), (Readout("q", 0),), "at least one token"),
        ((1, 2), (), "at least one readout"),
        ((1, 2), (Readout("q", 2),), "outside"),
    ],
)
def test_prompts_are_checked(tokens: tuple[int, ...], readouts: tuple[Readout, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Prompt(tokens, readouts)


def test_chunks_cover_every_token_once() -> None:
    assert chunks(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert chunks(4, 4) == [(0, 4)]
    assert chunks(4156, 4096) == [(0, 4156)]  # a short last piece joins the one before: one pass, not two
    assert chunks(9000, 4096) == [(0, 4096), (4096, 9000)]
    assert chunks(9300, 4096) == [(0, 4096), (4096, 8192), (8192, 9300)]
    assert chunks(0, 3) == []
    with pytest.raises(ValueError, match="positive"):
        chunks(3, 0)


def test_prompts_are_batched_in_order_within_the_padded_budget() -> None:
    from ippon.compare_arms.harness import budget_groups

    assert budget_groups([4, 4, 4, 4], 8) == [[0, 1], [2, 3]]
    assert budget_groups([2, 6, 1, 1], 12) == [[0, 1], [2, 3]]  # 3 x 6 padded is past 12
    assert budget_groups([20, 3], 8) == [[0], [1]]  # a prompt longer than the budget alone
    assert budget_groups([], 8) == []


def test_whole_prompts_are_batched_where_padding_costs_less_than_another_pass() -> None:
    # d1-2B's 50 questions: 4 set prompts; padding the short one to the others' length cost 15% of the request
    assert pass_groups([1200, 2600, 2600, 2600], 16384) == [[0], [1, 2, 3]]
    assert pass_groups([100] * 64, 16384) == [list(range(64))]  # equal prompts share one pass
    assert pass_groups([100, 100 + PASS_TOKENS // 2], 16384) == [[0, 1]]  # a little padding beats a pass
    assert pass_groups([4, 4, 4, 4], 8, 0) == [[0], [1], [2], [3]]  # passes cost nothing: no padding, ever
    assert pass_groups([20, 3, 3], 8) == [[0], [1, 2]]  # a prompt longer than the budget alone
    assert pass_groups([], 8) == []
