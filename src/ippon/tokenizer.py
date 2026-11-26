"""Tokenization with Hugging Face's `tokenizers` (Rust): the tokenizer.json a model ships, or, for a BERT-family
checkpoint that ships only its WordPiece `vocab.txt`, the fast tokenizer transformers converts it to
(`from_wordpiece`, as convert_slow_tokenizer.BertConverter builds it).

Formats write the creator's prompt strings themselves (no chat-template engine) and encode them here; `verify`
checks the resulting token ids against the creator's code exactly. Text never gets special tokens added, but
special tokens written in it (`<|im_start|>`) are recognised, as Hugging Face's tokenizers do by default.

Within `literal(mapping)`, placeholders (`placeholder(i)`) stand for strings that are to be read as text: a
request's state that contains a special token's string (`Decider.load(literal_state=True)`) has each one replaced
by a placeholder before its prompts are written, and here each placeholder is encoded as that string's ordinary
tokens, as if special tokens were not special, the text around it encoded on its own. A text without placeholders
is encoded as always.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from tokenizers import AddedToken, Encoding, Regex, decoders, models, normalizers, pre_tokenizers, processors
from tokenizers import Tokenizer as _Tokenizer

from .checkpoint import invalid_file
from .errors import UnsupportedModel, as_decision_error, out_of_memory

_LITERAL: ContextVar[Mapping[str, str] | None] = ContextVar("literal", default=None)
# Texts of at most this many characters in all are encoded one by one: a batch's call to the tokenizer's thread pool
# costs about 70 us (M5 Pro: a question's instructions and three options, 85 us as a batch, 13 us one by one; three
# 600-character texts take the same either way, longer ones less as a batch).
SERIAL_CHARS = 1500
# transformers 5 builds a checkpoint's tokenizer as the class its tokenizer_config.json names, whose own normalizer and
# pre-tokenizer replace tokenizer.json's. Qwen2Tokenizer splits words by this pattern, whose letters are \p{L} only,
# where the tokenizer.json of Qwen3.5 checkpoints has [\p{L}\p{M}]+: d3's, Torchcast's and H2O's creators name
# Qwen2Tokenizer, so their code splits text at each combining mark (Hindi, Thai, Arabic, Tamil: 4 of 17 probe texts
# encoded otherwise with tokenizer.json alone, 28 Oct 2026).
QWEN2_SPLIT = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"""
    r"""|\s+(?!\S)|\s+"""
)
QWEN2_CLASSES = ("Qwen2Tokenizer", "Qwen2TokenizerFast")


def placeholder(i: int) -> str:
    """The i-th placeholder: private-use characters no prompt holds."""
    return f"\ue000{chr(0xE100 + i)}\ue001"


@contextmanager
def literal(mapping: Mapping[str, str]) -> Iterator[None]:
    """Encode, within the block, each placeholder in `mapping` as its string's ordinary tokens."""
    token = _LITERAL.set(dict(mapping) or None)
    try:
        yield
    finally:
        _LITERAL.reset(token)


@dataclass(frozen=True)
class Tokenizer:
    backend: _Tokenizer
    _plain: list[_Tokenizer] = field(default_factory=list, compare=False, repr=False)  # special tokens as text

    def __post_init__(self) -> None:
        # A tokenizer.json can switch on padding and truncation (one may pad every batch to its longest text);
        # prompts are exact token ids, so neither is ever applied.
        self.backend.no_padding()
        self.backend.no_truncation()

    @property
    def special_strings(self) -> list[str]:
        """The strings of the tokenizer's special tokens, longest first."""
        found = {a.content for a in self.backend.get_added_tokens_decoder().values() if a.special}
        return sorted(found, key=len, reverse=True)

    @property
    def special_ids(self) -> set[int]:
        """The ids of the tokenizer's special tokens."""
        return {i for i, a in self.backend.get_added_tokens_decoder().items() if a.special}

    def _as_text(self) -> _Tokenizer:
        if not self._plain:
            plain = _Tokenizer.from_str(self.backend.to_str())
            plain.no_padding()
            plain.no_truncation()
            plain.encode_special_tokens = True
            self._plain.append(plain)
        return self._plain[0]

    def _encoding(self, text: str) -> Encoding:
        """`text` encoded without special tokens added; its placeholders (in a `literal` block) as their strings'
        ordinary tokens, each piece encoded on its own and the pieces merged."""
        mapping = _LITERAL.get()
        if not mapping or not any(p in text for p in mapping):
            return self.backend.encode(text, add_special_tokens=False)
        parts = re.split("(" + "|".join(re.escape(p) for p in mapping) + ")", text)
        pieces = []
        for part in parts:
            if not part:
                continue
            if part in mapping:
                pieces.append(self._as_text().encode(mapping[part], add_special_tokens=False))
            else:
                pieces.append(self.backend.encode(part, add_special_tokens=False))
        return Encoding.merge(pieces, growing_offsets=True)

    def _with_specials(self, text: str, pair: str | None = None) -> Encoding:
        first = self._encoding(text)
        second = self._encoding(pair) if pair is not None else None
        return self.backend.post_process(first, second, add_special_tokens=True)

    @classmethod
    def from_file(cls, path: str | Path, config: Mapping[str, object] | None = None) -> Tokenizer:
        """A tokenizer.json, as transformers 5 loads it with its tokenizer_config.json (`config`): the added tokens
        the config lists and the file lacks, at their ids (Qwen3.5 checkpoints list seven audio and speech tokens only
        there, `<|audio_pad|>` among them); for a checkpoint that names Qwen2Tokenizer, that class's normalizer, split
        and byte-level steps (tokenization_qwen2.py)."""
        path = Path(path)
        if not path.is_file():
            raise UnsupportedModel(f"{path}: no tokenizer.json")
        try:
            backend = _Tokenizer.from_file(str(path))
        except Exception as error:  # tokenizers reports malformed files as a plain Exception
            if out_of_memory(error):
                raise as_decision_error(error) from None
            raise invalid_file(path, f"cannot load tokenizer ({error})") from None
        if config:
            _add_listed_tokens(backend, config, path)
        if config and config.get("tokenizer_class") in QWEN2_CLASSES:
            backend.normalizer = normalizers.NFC()
            backend.pre_tokenizer = pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Split(Regex(QWEN2_SPLIT), behavior="isolated", invert=False),
                    pre_tokenizers.ByteLevel(add_prefix_space=bool(config.get("add_prefix_space")), use_regex=False),
                ]
            )
            backend.decoder = decoders.ByteLevel()
        return cls(backend)

    @classmethod
    def from_wordpiece(cls, vocab: str | Path, config: dict[str, object] | None = None) -> Tokenizer:
        """A BERT WordPiece vocabulary (`vocab.txt`, one token a line) with its tokenizer_config.json's settings, as
        transformers' BertConverter makes it: BertNormalizer (lower-casing, accents, Chinese characters), the BERT
        pre-tokenizer, "[CLS] $A [SEP]" and "[CLS] $A [SEP] $B:1 [SEP]:1"."""
        c = config or {}
        try:
            tokens = Path(vocab).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise invalid_file(Path(vocab), f"cannot read vocabulary ({error})") from None
        ids = {t: i for i, t in enumerate(tokens)}
        unk, cls_, sep = (
            str(c.get(k, d)) for k, d in (("unk_token", "[UNK]"), ("cls_token", "[CLS]"), ("sep_token", "[SEP]"))
        )
        if cls_ not in ids or sep not in ids or unk not in ids:
            raise UnsupportedModel(f"{vocab}: a WordPiece vocabulary without {cls_}, {sep} or {unk}")
        backend = _Tokenizer(models.WordPiece(ids, unk_token=unk))
        accents = c.get("strip_accents")
        backend.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=bool(c.get("tokenize_chinese_chars", True)),
            strip_accents=None if accents is None else bool(accents),
            lowercase=bool(c.get("do_lower_case", True)),
        )
        backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        backend.post_processor = processors.TemplateProcessing(
            single=f"{cls_}:0 $A:0 {sep}:0",
            pair=f"{cls_}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[(cls_, ids[cls_]), (sep, ids[sep])],
        )
        backend.decoder = decoders.WordPiece(prefix="##")
        return cls(backend)

    def _literal(self, *texts: str) -> bool:
        mapping = _LITERAL.get()
        return bool(mapping) and any(p in t for t in texts for p in mapping or ())

    def encode(self, text: str) -> list[int]:
        return list(self._encoding(text).ids)

    def encode_literal(self, text: str) -> list[int]:
        """`text` encoded as ordinary tokens, with special-token matching off (no added specials)."""
        return list(self._as_text().encode(text, add_special_tokens=False).ids)

    def encode_offsets(
        self, texts: Sequence[str], special: bool = False
    ) -> list[tuple[list[int], list[tuple[int, int]]]]:
        """Token ids and each token's (start, end) character offsets, for several texts; with `special`, the
        tokenizer's own special tokens added as `encode_special` adds them."""
        if self._literal(*texts):
            found = [self._with_specials(t) if special else self._encoding(t) for t in texts]
        else:
            found = self._many(texts, special)
        return [(list(e.ids), [(int(a), int(b)) for a, b in e.offsets]) for e in found]

    def encode_special(self, text: str) -> list[int]:
        """With the tokenizer's own special tokens added (what `tok(text)` gives in transformers)."""
        if self._literal(text):
            return list(self._with_specials(text).ids)
        return list(self.backend.encode(text, add_special_tokens=True).ids)

    def encode_pair(self, text: str, pair: str) -> list[int]:
        """Two texts as one input, with the tokenizer's own special tokens (`tok(text, pair)` in transformers)."""
        if self._literal(text, pair):
            return list(self._with_specials(text, pair).ids)
        return list(self.backend.encode(text, pair, add_special_tokens=True).ids)

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Several texts at once (on the tokenizer's threads when they are long)."""
        if self._literal(*texts):
            return [list(self._encoding(t).ids) for t in texts]
        return [list(e.ids) for e in self._many(texts, False)]

    def _many(self, texts: Sequence[str], special: bool) -> list[Encoding]:
        """Each text's encoding: one by one when they are short (SERIAL_CHARS), else as a batch on the tokenizer's
        threads. The same encodings either way."""
        if len(texts) == 1 or sum(len(t) for t in texts) <= SERIAL_CHARS:
            return [self.backend.encode(t, add_special_tokens=special) for t in texts]
        return list(self.backend.encode_batch(list(texts), add_special_tokens=special))

    def decode(self, ids: Sequence[int]) -> str:
        """The text of token ids, special tokens included (what the model reads, for `Decider.inspect`)."""
        return str(self.backend.decode(list(ids), skip_special_tokens=False))

    def token_id(self, token: str) -> int | None:
        return self.backend.token_to_id(token)


def _add_listed_tokens(backend: _Tokenizer, config: Mapping[str, object], path: Path) -> None:
    """The added tokens tokenizer_config.json's added_tokens_decoder lists that the tokenizer lacks, added in id order
    as transformers adds them; UnsupportedModel when one would not land on its listed id."""
    listed = config.get("added_tokens_decoder")
    if not isinstance(listed, Mapping):
        return
    have = {a.content for a in backend.get_added_tokens_decoder().values()}
    for key, entry in sorted(listed.items(), key=lambda item: int(item[0])):
        if not isinstance(entry, Mapping) or entry.get("content") in have:
            continue
        token = AddedToken(
            str(entry["content"]),
            single_word=bool(entry.get("single_word", False)),
            lstrip=bool(entry.get("lstrip", False)),
            rstrip=bool(entry.get("rstrip", False)),
            normalized=bool(entry.get("normalized", False)),
            special=bool(entry.get("special", False)),
        )
        if token.special:
            backend.add_special_tokens([token])
        else:
            backend.add_tokens([token])
        if backend.token_to_id(token.content) != int(key):
            raise UnsupportedModel(f"{path}: added token {token.content!r} would not be id {key}")
