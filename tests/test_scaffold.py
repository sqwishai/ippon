from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from ippon.cli import main
from ippon.scaffold import module_name, write


def test_the_package_it_writes_is_valid_python_with_an_entry_point(tmp_path: Path) -> None:
    files = write("acme/decider", "lfm2", tmp_path / "pkg")
    for path in files:
        if path.suffix == ".py":
            ast.parse(path.read_text(), str(path))
    text = (tmp_path / "pkg/pyproject.toml").read_text()
    assert '"acme/decider" = "acme_decider_ippon:Adapter"' in text
    sys.path.insert(0, str(tmp_path / "pkg/src"))
    try:
        import acme_decider_ippon  # type: ignore[import-not-found]

        adapter = acme_decider_ippon.Adapter()
        assert adapter.name == "acme/decider"
        assert "ctx.backbone(lfm2, model)" in (tmp_path / "pkg/src/acme_decider_ippon/__init__.py").read_text()
        renders_and_scores(acme_decider_ippon.Format)
    finally:
        sys.path.remove(str(tmp_path / "pkg/src"))
        sys.modules.pop("acme_decider_ippon", None)


def renders_and_scores(fmt_class: Any) -> None:
    """The generated format renders a request (a JSON state, marked where it ends) and scores it."""
    import mlx.core as mx
    from tokenizers import Tokenizer as Raw
    from tokenizers import models, pre_tokenizers

    from ippon.adapter import Hidden
    from ippon.schema import parse_request
    from ippon.tokenizer import Tokenizer

    words = ["[UNK]", "State", ":", "Question", "Answer", "{", "}", '"', "order", "1", "Refund", "?", "true", "false"]
    raw = Raw(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    weights = [[(4 * row + col) / 20 for col in range(4)] for row in range(len(words))]
    fmt = fmt_class(Tokenizer(raw), mx.array(weights))
    body = {"state": {"order": 1}, "questions": {"q": {"type": "noul", "instructions": "Refund?"}}}
    rendering = fmt.render(parse_request(body))
    (prompt,) = rendering.prompts
    assert prompt.marks
    assert 0 < prompt.marks[0] < len(prompt.tokens)
    hidden = [0.5, -0.25, 0.75, 1.0]
    scores = fmt.scores(rendering, Hidden({"q": mx.array([hidden])}))
    logits = [
        sum(a * b for a, b in zip(hidden, weights[words.index(option)], strict=True)) for option in ("true", "false")
    ]
    probabilities = [math.exp(value - max(logits)) for value in logits]
    assert scores["q"].tolist() == pytest.approx([p / sum(probabilities) for p in probabilities], abs=1e-6)


def test_it_never_overwrites_and_checks_its_arguments(tmp_path: Path) -> None:
    write("acme/x", "qwen35", tmp_path)
    with pytest.raises(ValueError, match="not overwriting"):
        write("acme/x", "qwen35", tmp_path)
    with pytest.raises(ValueError, match="arch"):
        write("acme/y", "llama", tmp_path / "y")
    with pytest.raises(ValueError, match="creator/model"):
        write("../evil", "qwen35", tmp_path / "z")
    assert module_name("Acme/Decider-2B") == "acme_decider_2b_ippon"
    assert module_name("2b/x") == "adapter_2b_x_ippon"


def test_the_command_writes_a_package_once_and_reports_a_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["new-adapter", "acme/cli", "--out", str(tmp_path / "cli")]) == 0
    assert main(["new-adapter", "acme/cli", "--out", str(tmp_path / "cli")]) == 3  # invalid_request
    assert "not overwriting" in capsys.readouterr().err
