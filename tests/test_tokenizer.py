from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer as Backend
from tokenizers import models, pre_tokenizers

from ippon.tokenizer import Tokenizer


def test_a_tokenizer_json_with_padding_and_truncation_on_encodes_every_text_exactly() -> None:
    backend = Backend(models.WordLevel({"a": 0, "b": 1, "c": 2, "<pad>": 3, "<unk>": 4}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.enable_padding(pad_id=3, pad_token="<pad>")
    backend.enable_truncation(max_length=2)
    tok = Tokenizer(backend)
    assert tok.encode_batch(["a", "a b c"]) == [[0], [0, 1, 2]]
    assert [ids for ids, _ in tok.encode_offsets(["a", "a b c"])] == [[0], [0, 1, 2]]
    assert tok.encode("a b c") == [0, 1, 2]


def test_a_wordpiece_vocab_builds_berts_fast_tokenizer(tmp_path: Path) -> None:
    """A BERT checkpoint with only vocab.txt: lower-cased, accents stripped, [CLS] a [SEP] b [SEP] with ## pieces."""
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "cafe", "is", "open", "##s", "now", "?"]
    (tmp_path / "vocab.txt").write_text("\n".join(words) + "\n")
    tok = Tokenizer.from_wordpiece(tmp_path / "vocab.txt", {"do_lower_case": True})
    assert tok.encode_special("Café IS opens") == [2, 4, 5, 6, 7, 3]
    assert tok.encode_pair("now?", "zebra") == [2, 8, 9, 3, 1, 3]
    cased = Tokenizer.from_wordpiece(tmp_path / "vocab.txt", {"do_lower_case": False})
    assert cased.encode_special("Café") == [2, 1, 3]


def test_a_placeholder_is_encoded_as_its_strings_ordinary_tokens() -> None:
    from tokenizers import Tokenizer as Raw
    from tokenizers import models, pre_tokenizers

    from ippon.tokenizer import Tokenizer, literal, placeholder

    vocab = {w: i for i, w in enumerate(["[UNK]", "a", "b", "<", "|", ">", "end"])}
    raw = Raw(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Split(pattern=" ", behavior="removed")
    raw.add_special_tokens(["<|end|>"])
    tok = Tokenizer(raw)
    special = tok.token_id("<|end|>")
    assert tok.special_strings == ["<|end|>"]
    assert special in tok.encode("a <|end|> b")  # recognised as the token, as Hugging Face does
    p = placeholder(0)
    with literal({p: "<|end|>"}):
        ids = tok.encode(f"a {p} b <|end|>")
        assert ids.count(special) == 1  # only the template's own, written as itself
        assert tok.encode("a b") == [1, 2]  # a text without placeholders: as always
        (found,) = tok.encode_offsets([f"a {p} b"])
        assert special not in found[0]
    assert tok.encode("a b") == [1, 2]
