"""Generic sequence-classifier and answer-token adapters against independent NumPy references."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

import deberta_numpy
import dense_numpy
import modernbert_numpy
from adapter_fixtures import softmax
from ippon import asks, registry
from ippon.adapter import LoadContext
from ippon.adapters.classifier import Classifier
from ippon.adapters.label_tokens import LabelTokens
from ippon.decider import Decider
from ippon.errors import ContextTooLong, InsufficientMemory, InvalidRequest, UnsupportedModel
from ippon.model_dir import ModelDir
from ippon.plan import plan
from ippon.schema import parse_request

WORDS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"] + [f"w{i}" for i in range(40)]
LABELS = {"0": "billing", "1": "sales", "2": "support"}


def write_tokenizer(path: Path) -> None:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tok.save(str(path / "tokenizer.json"))


def ids(text: str, pair: str | None = None) -> list[int]:
    words = [1] + [WORDS.index(w) for w in text.split()] + [2]
    return words + ([WORDS.index(w) for w in pair.split()] + [2] if pair is not None else [])


def qwen3_classifier(
    path: Path, labels: dict[str, str] = LABELS, settings: dict[str, Any] | None = None, **config: Any
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    c = (
        dense_numpy.tiny_config("qwen3")
        | {
            "architectures": ["Qwen3ForSequenceClassification"],
            "id2label": labels,
            "num_labels": len(labels),
        }
        | config
    )
    w = dense_numpy.tiny_weights(c)
    del w["lm_head.weight"]
    w["score.weight"] = (np.random.default_rng(7).standard_normal((len(labels), c["hidden_size"])) * 0.5).astype(
        np.float32
    )
    save(path, c, w, settings)
    return c, w


def save(path: Path, config: dict[str, Any], weights: dict[str, np.ndarray], settings: dict[str, Any] | None) -> None:
    (path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / "model.safetensors"), {k: mx.array(v) for k, v in weights.items()})
    write_tokenizer(path)
    if settings is not None:
        (path / "ippon.json").write_text(json.dumps(settings))


def decider(path: Path) -> Decider:
    """The classifier as Decider.load makes it: its adapter's questions are the ones it asks itself."""
    model = ModelDir.open(path)
    d = Decider(Classifier().load(model, LoadContext(dtype=mx.float32)))
    d.own_questions = asks.own(Classifier(), model)
    return d


def logits(c: dict[str, Any], w: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    return np.asarray(dense_numpy.forward(c, w, tokens)[-1] @ w["score.weight"].T)


def test_a_sequence_classifier_is_found_with_no_adapter_of_its_own(tmp_path: Path) -> None:
    qwen3_classifier(tmp_path)
    assert registry.select(ModelDir.open(tmp_path)).name == "generic/classifier"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}))
    assert not Classifier().matches(ModelDir.open(tmp_path))


@pytest.mark.parametrize(
    ("filename", "damaged"),
    [
        ("config.json", b"{"),
        ("config.json", b"[]"),
        ("config.json", b"\xff"),
        ("ippon.json", b"{"),
        ("tokenizer.json", b"{"),
        ("tokenizer.json", b"{}"),
        ("model.safetensors.index.json", b"{"),
        ("model.safetensors.index.json", b'{"weight_map":[]}'),
        ("model.safetensors.index.json", b'{"weight_map":{"w":"missing.safetensors"}}'),
        ("model.safetensors.index.json", b'{"weight_map":{"w":"../outside.safetensors"}}'),
        ("model.safetensors", b"broken"),
        ("model.safetensors", b"\xff" * 8),
        ("model.safetensors", None),
    ],
)
def test_damaged_checkpoints_name_the_file_and_load_after_repair(
    tmp_path: Path, filename: str, damaged: bytes | None, capsys: pytest.CaptureFixture[str]
) -> None:
    from ippon.cli import main

    config, weights = qwen3_classifier(tmp_path)
    path = tmp_path / filename
    original = path.read_bytes() if path.exists() else None
    path.write_bytes(damaged if damaged is not None else (original or b"")[:-10])
    with pytest.raises(UnsupportedModel, match=filename.replace(".", r"\.")):
        Decider.load(tmp_path, precision="fp32", keep_awake=None, pin_memory=False)
    request = json.dumps(
        {"state": "w1 w2", "questions": {"team": {"type": "choice", "criteria": list(LABELS.values())}}}
    )
    assert main(["decide", str(tmp_path), request, "--json"]) == 4
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "unsupported_model"
    if original is None:
        path.unlink()
    else:
        path.write_bytes(original)
    with Decider.load(tmp_path, precision="fp32", keep_awake=None, pin_memory=False) as model:
        result = model.classify("w1 w2")
    assert isinstance(result, dict)
    np.testing.assert_allclose(list(result.values()), softmax(logits(config, weights, ids("w1 w2"))), rtol=1e-4)


@pytest.mark.parametrize("failure", [MemoryError("allocation failed"), RuntimeError("[metal::malloc] out of memory")])
def test_allocator_failures_are_typed_and_requests_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: Exception
) -> None:
    from ippon.cli import main

    config, weights = qwen3_classifier(tmp_path)
    request = json.dumps(
        {"state": "w1 w2", "questions": {"team": {"type": "choice", "criteria": list(LABELS.values())}}}
    )

    def exhausted(*args: Any, **kwargs: Any) -> None:
        raise type(failure)(str(failure))

    with Decider.load(tmp_path, precision="fp32", keep_awake=None, pin_memory=False) as model:
        with monkeypatch.context() as fault:
            fault.setattr(mx, "eval", exhausted)
            with pytest.raises(InsufficientMemory, match="retry"):
                Decider.load(tmp_path, precision="fp32", keep_awake=None, pin_memory=False)
            with pytest.raises(InsufficientMemory, match="retry"):
                model.classify("w1 w2")
            assert main(["decide", str(tmp_path), request, "--json", "--precision", "fp32"]) == 6
            assert json.loads(capsys.readouterr().out)["error"]["type"] == "insufficient_memory"
        result = model.classify("w1 w2")
    assert isinstance(result, dict)
    np.testing.assert_allclose(list(result.values()), softmax(logits(config, weights, ids("w1 w2"))), rtol=1e-4)


def test_every_kind_of_question_reads_the_labels_probabilities(tmp_path: Path) -> None:
    """A choice gets its labels' probabilities renormalized over those asked, a yes/no the probability of the label
    its instructions name, a score the labels as ordered levels; a yes/no naming no label, or a choice of labels the
    model doesn't have, is refused."""
    c, w = qwen3_classifier(tmp_path)
    d = decider(tmp_path)
    questions = {
        "c": {"type": "choice", "criteria": ["support", "billing"]},
        "n": {"type": "noul", "instructions": "sales"},
        "s": {"type": "score", "criteria": ["bad", "ok", "good"]},
    }
    got = d.systemone({"state": "w1 w2 w3", "questions": questions})["answers"]
    z = logits(c, w, ids("w1 w2 w3"))
    want = softmax(z[[2, 0]])
    np.testing.assert_allclose(
        [got["c"]["probabilities"][k] for k in ("support", "billing")], want, rtol=1e-4, atol=1e-5
    )
    assert got["c"]["choice"] == ("support" if want[0] > want[1] else "billing")
    p = softmax(z)
    assert got["n"]["noul"] == pytest.approx(p[1], rel=1e-4)
    assert got["s"]["score"] == pytest.approx(float(p[1] + 2 * p[2]), rel=1e-4)
    with pytest.raises(InvalidRequest, match="names one of the model's labels"):
        d.systemone({"state": "w9", "questions": {"q": {"type": "noul", "instructions": "Is it about money?"}}})
    with pytest.raises(InvalidRequest, match="not labels of this model: refunds"):
        d.systemone({"state": "w9", "questions": {"q": {"type": "choice", "criteria": ["refunds", "sales"]}}})


def test_classify_gives_every_labels_probability(tmp_path: Path) -> None:
    """For one text or several, and only on a classifier."""
    from ippon.adapter import Loaded
    from toy import ToyBackbone, ToyFormat

    c, w = qwen3_classifier(tmp_path)
    d = decider(tmp_path)
    one = d.classify("w5 w6")
    assert isinstance(one, dict)
    np.testing.assert_allclose(list(one.values()), softmax(logits(c, w, ids("w5 w6"))), rtol=1e-4, atol=1e-5)
    many = d.classify(["w5 w6", "w7"])
    assert isinstance(many, list)
    assert list(many[0]) == ["billing", "sales", "support"]
    np.testing.assert_allclose(list(many[1].values()), softmax(logits(c, w, ids("w7"))), rtol=1e-4, atol=1e-5)
    backbone = ToyBackbone()
    with pytest.raises(UnsupportedModel, match="classify\\(\\) is for classifiers"):
        Decider(Loaded(backbone, ToyFormat(backbone))).classify("text")


def test_multi_label_and_one_logit_models_use_each_logits_sigmoid(tmp_path: Path) -> None:
    c, w = qwen3_classifier(tmp_path, problem_type="multi_label_classification")
    d = decider(tmp_path)
    got = d.classify("w3")
    assert isinstance(got, dict)
    np.testing.assert_allclose(list(got.values()), 1 / (1 + np.exp(-logits(c, w, ids("w3")))), rtol=1e-4)
    with pytest.raises(InvalidRequest, match="a yes/no question per label"):
        d.systemone({"state": "w3", "questions": {"q": {"type": "choice", "criteria": ["sales", "support"]}}})
    one = tmp_path / "one"
    one.mkdir()
    c1, w1 = qwen3_classifier(one, labels={"0": "relevant"})
    assert decider(one).classify("w4")["relevant"] == pytest.approx(  # type: ignore[call-overload]
        1 / (1 + math.exp(-float(logits(c1, w1, ids("w4"))[0]))), rel=1e-4
    )


def test_settings_set_the_input_templates_and_text_pairs(tmp_path: Path) -> None:
    """A cross-encoder: the question's instructions and the state as a text pair."""
    c, w = qwen3_classifier(tmp_path, settings={"text": "{instructions}", "pair": "{state}"})
    d = decider(tmp_path)
    got = d.systemone(
        {
            "state": "w20 w21",
            "questions": {
                "a": {"type": "choice", "instructions": "w1", "criteria": ["billing", "sales"]},
                "b": {"type": "choice", "instructions": "w2 w3", "criteria": ["billing", "sales"]},
            },
        }
    )
    for q, text in (("a", "w1"), ("b", "w2 w3")):
        want = softmax(logits(c, w, ids(text, "w20 w21"))[[0, 1]])
        p = got["answers"][q]["probabilities"]
        np.testing.assert_allclose([p["billing"], p["sales"]], want, rtol=1e-4, atol=1e-5)
    qwen3_classifier(tmp_path, settings={"true_label": "nope"})
    with pytest.raises(UnsupportedModel, match="'nope' is not one of the model's labels"):
        decider(tmp_path)


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_a_modernbert_classifier_runs_its_head_over_the_pooled_tokens(tmp_path: Path, pooling: str) -> None:
    c = modernbert_numpy.tiny_config(["full_attention", "sliding_attention", "full_attention"]) | {
        "architectures": ["ModernBertForSequenceClassification"],
        "id2label": LABELS,
        "num_labels": 3,
        "classifier_pooling": pooling,
        "classifier_activation": "gelu",
    }
    w = modernbert_numpy.tiny_weights(c)
    rng = np.random.default_rng(3)
    head = {
        "head.dense.weight": (rng.standard_normal((32, 32)) * 0.3).astype(np.float32),
        "head.norm.weight": (1 + rng.standard_normal(32) * 0.1).astype(np.float32),
        "classifier.weight": (rng.standard_normal((3, 32)) * 0.5).astype(np.float32),
        "classifier.bias": (rng.standard_normal(3) * 0.5).astype(np.float32),
    }
    save(tmp_path, c, {"model." + k: v for k, v in w.items()} | head, None)
    got = decider(tmp_path).classify("w1 w2 w3 w4 w5 w6 w7 w8")
    h = modernbert_numpy.forward(c, w, ids("w1 w2 w3 w4 w5 w6 w7 w8"))
    x = h[0] if pooling == "cls" else h.mean(0)
    x = x @ head["head.dense.weight"].T
    x = 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    x = modernbert_numpy.ln(x, head["head.norm.weight"], 1e-5)
    want = softmax(x @ head["classifier.weight"].T + head["classifier.bias"])
    assert isinstance(got, dict)
    np.testing.assert_allclose(list(got.values()), want, rtol=1e-4, atol=1e-5)


BERT_HEADS: dict[str, tuple[str, str, Callable[[np.ndarray], np.ndarray]]] = {  # family: dense, classifier, activation
    "bert": ("bert.pooler.dense", "classifier", np.tanh),
    "roberta": ("classifier.dense", "classifier.out_proj", np.tanh),
    "xlm-roberta": ("classifier.dense", "classifier.out_proj", np.tanh),
    "distilbert": ("pre_classifier", "classifier", lambda x: np.maximum(x, 0)),
    "deberta-v2": ("pooler.dense", "classifier", deberta_numpy.gelu),
}


@pytest.mark.parametrize("family", list(BERT_HEADS))
def test_bert_family_classifiers_run_their_own_heads(tmp_path: Path, family: str) -> None:
    """Each encoder's first-token pooler against its independent NumPy path, including BERT token types."""
    import bert_numpy

    deberta = family == "deberta-v2"
    labels = {"0": "SAFE", "1": "INJECTION"} if deberta else LABELS
    c = (deberta_numpy.tiny_config() if deberta else bert_numpy.tiny_config(family)) | {
        "architectures": [f"{family.title().replace('-', '')}ForSequenceClassification"],
        "id2label": labels,
        "num_labels": len(labels),
    }
    w = deberta_numpy.tiny_weights(c) if deberta else bert_numpy.tiny_weights(c)
    first, second, act = BERT_HEADS[family]
    rng = np.random.default_rng(5 if deberta else 4)
    w |= {
        first + ".weight": (rng.standard_normal((32, 32)) * 0.3).astype(np.float32),
        first + ".bias": (rng.standard_normal(32) * 0.1).astype(np.float32),
        second + ".weight": (rng.standard_normal((len(labels), 32)) * 0.5).astype(np.float32),
        second + ".bias": (rng.standard_normal(len(labels)) * 0.1).astype(np.float32),
    }
    pair = family == "bert"
    text = "w3 w4 w5 w6" if deberta else "w4 w5 w6"
    save(tmp_path, c, w, {"text": "{instructions}", "pair": "{state}"} if pair else None)
    d = decider(tmp_path)
    question = (
        {"type": "noul", "instructions": "INJECTION"}
        if deberta
        else {"type": "choice", "instructions": "w1 w2", "criteria": list(labels.values())}
    )
    got = d.systemone({"state": text, "questions": {"q": question}})["answers"]["q"]
    tokens = ids("w1 w2", text) if pair else ids(text)
    h = (
        deberta_numpy.forward(c, w, tokens)
        if deberta
        else bert_numpy.forward(c, w, tokens, [0] * 4 + [1] * 4 if pair else None)
    )[0]
    x = act(h @ w[first + ".weight"].T + w[first + ".bias"])
    want = softmax(x @ w[second + ".weight"].T + w[second + ".bias"])
    if deberta:
        assert got["noul"] == pytest.approx(float(want[1]), rel=1e-4)
    else:
        np.testing.assert_allclose(list(got["probabilities"].values()), want, rtol=1e-4, atol=1e-5)
        if not pair:
            np.testing.assert_allclose(list(d.classify(text).values()), want, rtol=1e-4, atol=1e-5)  # type: ignore[union-attr]
        with pytest.raises(ContextTooLong, match="more than the model's 40"):
            d.classify(" ".join(["w1"] * 45))


def test_warm_up_asks_the_models_own_questions(tmp_path: Path) -> None:
    """The standard warm-up questions aren't about a classifier's labels, so it refuses them all: it is warmed up
    with its own, so auto precision measures its activations."""
    qwen3_classifier(tmp_path, problem_type="multi_label_classification")
    assert decider(tmp_path).warm_up_measuring_peak() > 0


def test_a_gemma3_classifier_scores_its_last_token(tmp_path: Path) -> None:
    import gemma3_numpy

    c = gemma3_numpy.tiny_config() | {
        "architectures": ["Gemma3TextForSequenceClassification"],
        "id2label": LABELS,
        "num_labels": 3,
    }
    w = gemma3_numpy.tiny_weights(c)
    w["score.weight"] = (np.random.default_rng(9).standard_normal((3, 32)) * 0.5).astype(np.float32)
    save(tmp_path, c, w, None)
    text = "w1 w2 w3 w4 w5 w6 w7 w8 w9"  # longer than the sliding window (5)
    got = decider(tmp_path).classify(text)
    want = softmax(gemma3_numpy.forward(c, w, ids(text))[-1] @ w["score.weight"].T)
    assert isinstance(got, dict)
    np.testing.assert_allclose(list(got.values()), want, rtol=1e-4, atol=1e-5)


TOKEN_WORDS = ["[PAD]", "[BOS]", "[UNK]"] + [f"w{i}" for i in range(61)]  # 64: the tiny model's vocabulary
TOKEN_PROMPT = "w1 w2 {state} w3 {instructions} w4"


def write_token_model(path: Path, settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(TOKEN_WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 1)])
    tok.save(str(path / "tokenizer.json"))
    c = dense_numpy.tiny_config("llama") | {"architectures": ["LlamaForCausalLM"]}
    w = dense_numpy.tiny_weights(c)
    (path / "config.json").write_text(json.dumps(c))
    mx.save_safetensors(str(path / "model.safetensors"), {k: mx.array(v) for k, v in w.items()})
    (path / "ippon.json").write_text(json.dumps({"format": "label_tokens"} | settings))
    return c, w


def token_ids(text: str) -> list[int]:
    return [1] + [TOKEN_WORDS.index(t) for t in text.split()]


def token_logits(c: dict[str, Any], w: dict[str, np.ndarray], text: str) -> np.ndarray:
    return np.asarray(dense_numpy.forward(c, w, token_ids(text))[-1] @ w["lm_head.weight"].T)


def token_decider(path: Path) -> Decider:
    return Decider(LabelTokens().load(ModelDir.open(path), LoadContext(dtype=mx.float32)))


def test_a_yes_no_question_reads_its_two_answer_tokens(tmp_path: Path) -> None:
    c, w = write_token_model(tmp_path, {"prompt": TOKEN_PROMPT, "noul": {"true": "w30", "false": "w31"}})
    got = token_decider(tmp_path).systemone(
        {"state": "w7 w8 w9", "questions": {"q": {"type": "noul", "instructions": "w5"}}}
    )
    z = token_logits(c, w, "w1 w2 w7 w8 w9 w3 w5 w4")
    assert got["answers"]["q"]["noul"] == pytest.approx(softmax(z[[30 + 3, 31 + 3]])[0], rel=1e-4)


def test_over_the_vocabulary_one_tokens_probability_is_the_answer(tmp_path: Path) -> None:
    """An end-of-turn detector: P(the end token) among every token; false is the rest."""
    c, w = write_token_model(tmp_path, {"prompt": "{state}", "noul": {"true": "w40"}, "over": "vocabulary"})
    got = token_decider(tmp_path).systemone({"state": "w10 w11", "questions": {"done": {"type": "noul"}}})
    assert got["answers"]["done"]["noul"] == pytest.approx(softmax(token_logits(c, w, "w10 w11"))[40 + 3], rel=1e-4)


def test_a_choice_and_a_score_read_their_options_tokens(tmp_path: Path) -> None:
    settings = {
        "prompt": "w1 {state} w2 {options} w3",
        "noul": {"true": "w30", "false": "w31"},
        "choice": "{label}",
        "score": "w5{index}",
        "option": "{label}",
    }
    c, w = write_token_model(tmp_path, settings)
    got = token_decider(tmp_path).systemone(
        {
            "state": "w9",
            "questions": {
                "c": {"type": "choice", "criteria": ["w20", "w21", "w22"]},
                "s": {"type": "score", "criteria": ["w12", "w13"]},
            },
        }
    )
    zc = token_logits(c, w, "w1 w9 w2 w20\nw21\nw22 w3")
    want = softmax(zc[[20 + 3, 21 + 3, 22 + 3]])
    np.testing.assert_allclose(list(got["answers"]["c"]["probabilities"].values()), want, rtol=1e-4, atol=1e-6)
    zs = token_logits(c, w, "w1 w9 w2 w12\nw13 w3")
    p = softmax(zs[[50 + 3, 51 + 3]])
    assert got["answers"]["s"]["score"] == pytest.approx(float(p[1]), rel=1e-4)


def test_questions_about_one_state_share_its_reading_and_mark_its_end(tmp_path: Path) -> None:
    """Every question's prompt starts with the prompt's text up to its instructions, read once for all of them, and
    marks a position near the state's end that the next request about a grown state still shares."""
    write_token_model(tmp_path, {"prompt": TOKEN_PROMPT, "noul": {"true": "w30", "false": "w31"}})
    d = token_decider(tmp_path)
    body = {
        "state": "w7 w8 w9 w10",
        "questions": {f"q{i}": {"type": "noul", "instructions": f"w{5 + i}"} for i in range(3)},
    }
    rendering = d.render(parse_request(body))
    shared = plan(rendering.prompts)
    assert list(shared.prefix) == token_ids("w1 w2 w7 w8 w9 w10 w3")
    (mark,) = shared.marks
    grown = d.render(parse_request({**body, "state": "w7 w8 w9 w10 w11"})).prompts[0].tokens
    assert rendering.prompts[0].tokens[:mark] == grown[:mark]


def test_answers_that_one_token_cannot_tell_apart_are_refused(tmp_path: Path) -> None:
    write_token_model(tmp_path, {"prompt": TOKEN_PROMPT, "noul": {"true": "w30 w1", "false": "w30 w2"}})
    with pytest.raises(InvalidRequest, match="start with the same token"):
        token_decider(tmp_path).systemone({"state": "w7", "questions": {"q": {"type": "noul"}}})
    write_token_model(tmp_path, {"prompt": TOKEN_PROMPT, "noul": {"true": "w30"}})
    with pytest.raises(UnsupportedModel, match='needs a "false" answer'):
        token_decider(tmp_path)
    for prompt in ("w1 {state} w2 {state}", "w1 {state!r}", "w1 {"):
        write_token_model(tmp_path, {"prompt": prompt, "noul": {"true": "w30", "false": "w31"}})
        with pytest.raises(UnsupportedModel, match="at most once"):
            token_decider(tmp_path)
    for prompt in ("w1 {{state}} w2 {state} w3", "w1 {{state}} w3"):  # escaped braces are text, not the field
        write_token_model(tmp_path, {"prompt": prompt, "noul": {"true": "w30", "false": "w31"}})
        token_decider(tmp_path).systemone({"state": "w7", "questions": {"q": {"type": "noul"}}})


def test_found_only_with_its_settings_in_the_file_or_given_in_code(tmp_path: Path) -> None:
    """An ippon.json of its format, or one given in code (Decider.load(model, settings=...): a Hub model,
    untouched), which loads and answers as the file would."""
    settings = {"format": "label_tokens", "prompt": TOKEN_PROMPT, "noul": {"true": "w30", "false": "w31"}}
    write_token_model(tmp_path, settings)
    assert registry.select(ModelDir.open(tmp_path)).name == "generic/label-tokens"
    body = {"state": "w7", "questions": {"q": {"type": "noul", "instructions": "w5"}}}
    from_file = token_decider(tmp_path).systemone(body)["answers"]
    (tmp_path / "ippon.json").write_text(json.dumps({"format": "classifier"}))
    assert not LabelTokens().matches(ModelDir.open(tmp_path))
    (tmp_path / "ippon.json").unlink()
    assert not LabelTokens().matches(ModelDir.open(tmp_path))
    model = ModelDir.open(tmp_path, {"ippon.json": settings})
    assert registry.select(model).name == "generic/label-tokens"
    assert Decider(LabelTokens().load(model, LoadContext(dtype=mx.float32))).systemone(body)["answers"] == from_file


def test_a_special_tokens_string_in_the_state_reads_as_text_with_literal_state(tmp_path: Path) -> None:
    from tokenizers import Tokenizer as Raw

    write_token_model(tmp_path, {"prompt": TOKEN_PROMPT, "noul": {"true": "w30", "false": "w31"}})
    raw = Raw.from_file(str(tmp_path / "tokenizer.json"))
    raw.add_special_tokens(["[BOS]"])
    raw.save(str(tmp_path / "tokenizer.json"))
    loaded = LabelTokens().load(ModelDir.open(tmp_path), LoadContext(dtype=mx.float32))
    body = {"state": "w5 [BOS] w6", "questions": {"q": {"type": "noul", "instructions": "w7"}}}
    plain = Decider(loaded).render(parse_request(body)).prompts[0].tokens
    safe = Decider(loaded, literal_state=True).render(parse_request(body))
    assert plain.count(1) == 2  # the prompt's own [BOS] and the state's
    assert safe.prompts[0].tokens.count(1) == 1  # the state's is text now
    assert safe.request.state == "w5 [BOS] w6"  # the answer is about the state as sent
    clean = {"state": "w5 w6", "questions": {"q": {"type": "noul", "instructions": "w7"}}}
    assert (
        Decider(loaded, literal_state=True).render(parse_request(clean)).prompts
        == Decider(loaded).render(parse_request(clean)).prompts
    )
