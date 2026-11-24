"""Liquid AI's d1 decision models on LFM3 (LFM Open License v1.0): d1-2B and d1-Omni-2B, read at their answer slots
under "set" attention (arch/lfm3.py). A reimplementation of the text path of their own code (prompt.py, attention.py,
runner.py, api.py, calibration.py; its defaults: no system turn, states as JSON, options by description), checked
token for token against it:

- the prompt is "<|startoftext|><|im_start|>user\\n<state>\\n\\n\\nQUESTION:\\n<question><|im_end|>\\n<|im_start|>
  assistant\\n<think></think>", the state as given when it is text, else JSON indented by 2 (the state and
  "QUESTION:" left out when there is none), and the answer read at its last token;
- up to 16 questions share one prompt: the first tokenized with the state as one string, each further one as
  "QUESTION:\\n<question>..." of its own, appended. The state's tokens are the prompt's but the first question's own
  (its tokens counted alone), and the model's full attention layers read it with every question while each question
  reads the state and itself (`plan.Prompt.groups`);
- the caller's text (the state and each question) has a zero-width space after the first character of every
  special token's string in it, so it never reads as that token;
- a choice lists "<code> <description, or the label with _ as spaces>" under "Options:", then "Reply with the
  option code only."; the codes are the labels themselves when they are single letters, else A..Z, else A..Z, AA,
  AB, ..., each replaced by the first free single-token code of a fixed pool when it isn't one token;
- a noul is the instructions (then "Yes: <true>" and "No: <false>" when it has criteria), then "Reply with yes
  or no only."; a score lists "<i> <level>", then "Reply with a single digit 0-<n-1> only.";
- each option is scored by the best of its single-token forms at its slot (yes, Yes, YES; no, No, NO; the code
  and " <code>"; the digit), divided by calibration.json's temperature, and softmaxed; a score answers its median
  level, the first whose cumulative probability reaches one half.

Instructions and option descriptions are written as Liquid's current prompt.py writes them (both checkpoints'
revisions use it; the token checks of such requests are against d1-2B's 10e7671): text as it is, any other JSON value
with Python's str(), and a description that is falsy (null, "", 0, false, [], {}) as its label. Instructions that are
missing, blank, or an object Liquid's code reads as media ("type" image, audio or video) are refused. Liquid's code
needs a map for a choice's criteria and text for a score's levels. Two extensions: a choice given as a list of labels
is accepted, shown as the labels without descriptions, and levels that are not text are written with str() too.

Pictures (both models): each at the head of the user turn, in order, as the chat template writes it,
"<|vision_start|><|image|><|vision_end|>", the image token expanded to the picture's rows (one placeholder each,
arch/common.media_id), which are state tokens like the text after them; the picture turned by its EXIF orientation
(the card's load_image) and capped at the checkpoint's pixel budget before Qwen2-VL's resize (media/lfm3.py).

An audio clip (d1-Omni-2B): after the pictures, "<|vision_start|><|audio|><|vision_end|>" with the audio token
expanded to the clip's rows, about 12.5 a second and no cut at any length (media/lfm3_audio.py). A spoken question
(its instructions a clip): that span where its text would be, after the escaping, its rows among its question's
tokens.

Videos (both models): after the pictures and before the audio clip, each video's span contains timestamped
frame-pair spans. Each pair is one temporal patch through the picture tower, with its own spatial attention and
convolutions. The decision API does not read the video's audio track (media/lfm3_video.py).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import mlx.core as mx
import numpy as np

from ..adapter import AdapterBase, FormatBase, Hidden, LoadContext, Loaded, Rendering
from ..arch import lfm3
from ..arch.common import MEDIA_ROW_BITS, Kept, expand, media_digest, media_id
from ..errors import InvalidRequest, UnsupportedModel
from ..formulas import median_level
from ..gguf.header import Header, close
from ..gguf.names import Reads, Rule, lfm3_media_rules, lfm3_rules
from ..gguf.source import lfm3_differences
from ..media import lfm3_audio
from ..media.decode import Media
from ..media.lfm3 import TOWER, Lfm3Vision, VisionConfig, capped, picture_rows
from ..media.lfm3_audio import clip_rows, samples
from ..media.lfm3_video import load_frames
from ..media.paths import Sampling, Seen, config_token, encode_kept, encoder_loader
from ..media.qwen import ImageConfig, QwenImages, QwenVideos, VideoConfig, frame_pair_times, preprocess
from ..media.video import Clip
from ..model_dir import API_VERSION, ModelDir
from ..plan import Prompt, Readout
from ..quant import Table
from ..readout import LabelSet, label_logits, option_probabilities
from ..schema import Question, QuestionType, Request, systemone_answer
from ..tokenizer import Tokenizer

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
LEAD = "<think></think>"  # what the template writes after the assistant header with thinking off: the answer slot
QUESTION = "QUESTION:\n"
BREAK = "\u200b"  # after a special token's first character in the caller's text
YES_FORMS, NO_FORMS = ("yes", "Yes", "YES"), ("no", "No", "NO")
LETTERS = tuple(chr(ord("A") + i) for i in range(26))
POOL = (
    [*LETTERS]
    + [f"{i:02d}" for i in range(100)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [f"#{i}" for i in range(200)]
    + [a + b for a in LETTERS for b in LETTERS]
)
SET_SIZE = 16  # questions a prompt holds at most (config.json's d1_set_size)
VISION = "<|vision_start|><|image|><|vision_end|>"  # one picture, as the chat template writes it
AUDIO = "<|vision_start|><|audio|><|vision_end|>"  # one clip, as the Omni's chat template writes it
SPOKEN = "\ue000"  # a spoken question's place in its block until its clip's span takes it (Liquid's media.SPOKEN)
LIQUID_MEDIA = ("image", "audio", "video")  # an instructions object of these types is media to Liquid (media.KINDS)
MAX_PIXELS = 1_638_400  # a picture's pixel budget when config.json names no d1_vision_max_pixels


@dataclass
class Pictures(QwenImages):
    """LFM3-VL's pictures: Qwen2-VL's processor (media/qwen.QwenImages), each picture first capped at the
    checkpoint's pixel budget (media/lfm3.capped), its rows from media/lfm3.Lfm3Vision."""

    max_pixels: int = MAX_PIXELS

    def count_size(self, width: int, height: int) -> int:
        return picture_rows(width, height, self.config, self.max_pixels)

    def prepare(self, rgb: np.ndarray) -> np.ndarray:
        return super().prepare(capped(rgb, self.max_pixels))

    def patches(self, rgb: np.ndarray) -> Any:
        return preprocess(capped(rgb, self.max_pixels), self.config)


@dataclass
class Videos(QwenVideos):
    """Liquid's video frames, sharing the picture tower and Qwen's temporal patch processor; each video's frame pairs
    inside one more vision span of their own."""

    def text(self, seen: Seen) -> str:
        return "<|vision_start|>" + super().text(seen) + "<|vision_end|>"


def omni_times(clip: Clip, temporal: int) -> tuple[float, ...]:
    """d1-Omni-2B's frame pair times: (first + last source index) / (2 fps) for each pair, the indices padded to a
    multiple of `temporal` with the last. The same times as Qwen's mean of the two frames' times (`timestamps`, which
    d1-2B's prompts carry), in another order of operations whose last bit can differ; kept as the Omni's prompts were
    checked token for token against Liquid's (12 of 12 video prompts)."""
    indices = list(clip.indices)
    indices += [indices[-1]] * (-len(indices) % temporal)
    return tuple((indices[i] + indices[i + temporal - 1]) / (2 * clip.fps) for i in range(0, len(indices), temporal))


def option_codes(labels: Sequence[str], scheme: int) -> list[str]:
    """Each option's code: the labels when they are single letters, else A..Z, else scheme 1's 00, 01, ... or
    scheme 2's A..Z, AA, AB, ... (config.json's d1_codes)."""
    labs = [label.strip() for label in labels]
    if labs and all(len(k) == 1 and k.isalpha() for k in labs):
        return labs
    if len(labs) <= 26:
        return list(LETTERS[: len(labs)])
    if scheme == 2:
        return ([*LETTERS] + [a + b for a in LETTERS for b in LETTERS])[: len(labs)]
    return [f"{i:02d}" for i in range(len(labs))]


def state_block(state: Any) -> str:
    return f"{state}\n\n" if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"


@dataclass
class Clips:
    """d1-Omni-2B's audio clips: each clip's placeholders (one per row of media/lfm3_audio.Lfm3Audio, which `loader`
    builds on the first clip) and its rows, the last few kept by digest."""

    token: int  # <|audio|>
    loader: Callable[[], Any]
    _kept: Kept = field(default_factory=Kept, init=False, repr=False)
    _counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)  # each clip's rows, by sha256

    def rows(self, clip: Media, waves: dict[int, np.ndarray | Media]) -> tuple[int, ...]:
        """The clip's placeholders; into `waves` by digest, its 16 kHz samples, decoded to count them, or the clip
        itself when it was counted before, so a new question about it decodes it only if its rows are not kept.
        InvalidRequest for a clip longer than its placeholders can address (about 43 minutes)."""
        digest = media_digest(clip.sha256)
        n = self._counts.get(clip.sha256)
        if n is None:
            wave = samples(clip)
            n = clip_rows(len(wave))
            waves[digest] = wave
            if len(self._counts) >= 4096:
                self._counts.clear()  # counts are cheap to redo; this only bounds the memory
            self._counts[clip.sha256] = n
        waves.setdefault(digest, clip)
        most = 1 << MEDIA_ROW_BITS
        if n > most:
            minutes = most * lfm3_audio.HOP * lfm3_audio.STRIDE // lfm3_audio.SAMPLE_RATE // 60
            raise InvalidRequest(
                f"an audio clip takes {n:,} input tokens, over the {most:,} one clip may take (about {minutes} "
                "minutes); send it in parts"
            )
        return tuple(media_id(digest, row) for row in range(n))

    def encode(self, renderings: Sequence[Rendering]) -> dict[int, mx.array]:
        """Every clip of the renderings' requests (the state's and the spoken questions'), by digest: its rows from
        the encoder, unless kept from before."""
        needed: dict[int, np.ndarray | Media] = {}
        for rendering in renderings:
            for digest, wave in rendering.meta["waves"].items():
                needed.setdefault(digest, wave)
        return encode_kept(self._kept, needed, self._through_encoder)

    def _through_encoder(self, new: dict[int, np.ndarray | Media]) -> dict[int, dict[int, mx.array]]:
        encoder = self.loader()
        return {
            digest: {digest: encoder(wave if isinstance(wave, np.ndarray) else samples(wave))}
            for digest, wave in new.items()
        }


def bucket(q: Question) -> str:
    """The calibration bucket of `q` (calibration.py's): its type, and for a choice its size."""
    if q.type in ("noul", "score"):
        return "noul" if q.type == "noul" else f"score:{len(q.options)}"
    k = len(q.options)
    return "choice:" + ("small" if k <= 3 else "mid" if k <= 8 else "large" if k <= 26 else "inventory")


@dataclass
class D1Format(FormatBase):
    tokenizer: Tokenizer
    bos: str
    output: Table  # the output matrix [vocabulary, hidden] (the tied embedding), maybe quantized
    temperature_default: float
    temperatures: Mapping[str, float] = field(default_factory=dict)  # by `bucket`
    codes: int = 2  # config.json's d1_codes
    set_size: int = SET_SIZE
    pictures: Pictures | None = None  # None for a build without the vision tower
    clips: Clips | None = None  # None for a build without the speech encoder (d1-2B)
    videos: Videos | None = None
    _aliases: dict[tuple[str, ...], list[tuple[str, int]]] = field(default_factory=dict, repr=False)

    def most_options(self, kind: QuestionType) -> int | None:
        if kind == "score":
            return 10  # "Reply with a single digit"
        return 26 + 26 * 26 if self.codes == 2 else 100  # A..ZZ, or 00..99

    @property
    def media(self) -> frozenset[str]:
        kinds = (("images", self.pictures), ("audio", self.clips), ("spoken", self.clips), ("videos", self.videos))
        return frozenset(kind for kind, path in kinds if path is not None)

    @cached_property
    def _specials(self) -> re.Pattern[str] | None:
        names = [s for s in self.tokenizer.special_strings if len(s) > 1 and BREAK not in s]
        return re.compile("|".join(map(re.escape, names))) if names else None

    def plain(self, text: str) -> str:
        """The caller's text with every special token's string broken by a zero-width space after its first
        character, again until none is left (one broken may make another)."""
        if self._specials is None:
            return text
        n = 1
        while n:
            text, n = self._specials.subn(lambda m: m[0][0] + BREAK + m[0][1:], text)
        return text

    def single(self, texts: Sequence[str]) -> list[int]:
        """The texts that are one token each, as ids, without repeats."""
        out: list[int] = []
        for ids in self.tokenizer.encode_batch(list(texts)):
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
        return out

    def aliases(self, labels: Sequence[str]) -> list[tuple[str, int]]:
        """Every option's code and its token: distinct single tokens, a code that isn't one replaced from POOL."""
        codes = tuple(option_codes(labels, self.codes))
        if codes not in self._aliases:
            used: set[int] = set()
            out: list[tuple[str, int]] = []

            def take(raw: str) -> bool:
                ids = self.tokenizer.encode(raw)
                if len(ids) != 1 or ids[0] in used:
                    return False
                out.append((raw, ids[0]))
                used.add(ids[0])
                return True

            for code in codes:
                if not take(code) and not any(take(raw) for raw in POOL):
                    raise InvalidRequest(f"no single-token code left for {len(codes)} options")
            self._aliases[codes] = out
        return self._aliases[codes]

    def label_tokens(self, q: Question) -> tuple[tuple[int, ...], ...]:
        """The tokens each option is scored by, max-pooled."""
        if q.type == "noul":
            return tuple(self.single(YES_FORMS)), tuple(self.single(NO_FORMS))
        if q.type == "score":
            return tuple(tuple(self.single([str(i)])) for i in range(len(q.criteria)))
        return tuple(
            (token, *(i for i in self.single([f" {code}"]) if i != token)) for code, token in self.aliases(q.options)
        )

    def question_block(self, q: Question) -> str:
        said = SPOKEN if q.spoken is not None else q.instructions
        if q.type == "choice":
            codes = self.aliases(q.options)
            lines = "\n".join(
                f"{code} {description or label.replace('_', ' ')}"
                for (code, _), (label, description) in zip(codes, q.criteria, strict=True)
            )
            return f"{said}\n\nOptions:\n{lines}\n\nReply with the option code only."
        if q.type == "noul":
            given = dict(q.criteria)
            extra = f"\nYes: {given.get('true')}\nNo: {given.get('false')}" if given else ""
            return f"{said}{extra}\n\nReply with yes or no only."
        legend = "\n".join(f"{i} {level}" for i, (_, level) in enumerate(q.criteria))
        return f"{said}\n\n{legend}\n\nReply with a single digit 0-{len(q.criteria) - 1} only."

    def prefix(self, state: Any, pictures: int = 0, clips: int = 0, videos: str = "") -> str:
        body = "" if state is None else f"{self.plain(state_block(state))}\n{QUESTION}"
        return f"{self.bos}{IM_START}user\n{VISION * pictures}{videos}{AUDIO * clips}{body}"

    def suffix(self, q: Question) -> str:
        block = self.plain(self.question_block(q))
        if q.spoken is not None:  # the clip's span in its place, after the escaping (Liquid's Reader.place)
            block = block.replace(SPOKEN, AUDIO, 1)
        return f"{block}{IM_END}\n{IM_START}assistant\n{LEAD}"

    def render(self, request: Request) -> Rendering:
        for q in request.questions:
            said = q.instructions
            if said is None or (isinstance(said, str) and not said.strip()):
                raise InvalidRequest(f"{q.id}: this model needs instructions, as text or other JSON, or spoken")
            if isinstance(said, Mapping) and said.get("type") in LIQUID_MEDIA:
                raise InvalidRequest(
                    f"{q.id}: Liquid's code reads instructions of type {said['type']!r} as media; a spoken question "
                    'is {"type": "input_audio", "input_audio": {"data": ...}}'
                )
        pictures: list[tuple[int, ...]] = []
        pixels: dict[int, np.ndarray] = {}
        if request.images:
            assert self.pictures is not None, "the decider refuses media a format does not take"
            pictures, pixels = self.pictures.rows(request.images)
        waves: dict[int, np.ndarray | Media] = {}
        heard: dict[str, tuple[tuple[int, ...], ...]] = {q.id: () for q in request.questions}  # each question's clip
        state_clips: tuple[tuple[int, ...], ...] = ()
        if request.audio or any(q.spoken is not None for q in request.questions):
            assert self.clips is not None, "the decider refuses media a format does not take"
            state_clips = tuple(self.clips.rows(clip, waves) for clip in request.audio)
            heard |= {q.id: (self.clips.rows(q.spoken, waves),) for q in request.questions if q.spoken is not None}
        video_text = ""
        video_rows: list[tuple[int, ...]] = []
        clips: dict[str, Clip] = {}
        video_info: tuple[dict[str, Any], ...] = ()
        if request.videos:
            assert self.videos is not None, "the decider refuses media a format does not take"
            seen, clips = self.videos.seen(request.videos)
            video_info = self.videos.summary(seen)
            video_text = "".join(self.videos.text(s) for s in seen)
            video_rows = [p for v, s in zip(request.videos, seen, strict=True) for p in self.videos.rows(v, s)]
        prefix = self.prefix(request.state, len(pictures), len(request.audio), video_text)
        header = "" if request.state is None else QUESTION
        prompts = []
        for start in range(0, len(request.questions), self.set_size):
            row = request.questions[start : start + self.set_size]
            suffixes = [self.suffix(q) for q in row]
            first, alone, *rest = self.tokenizer.encode_batch(
                [prefix + suffixes[0], suffixes[0], *(header + s for s in suffixes[1:])]
            )
            if pictures:  # each image token as its picture's rows
                assert self.pictures is not None
                first = self.pictures.expand(first, pictures)
            if video_rows:
                assert self.videos is not None
                first = expand(first, self.videos.token, video_rows)
            if self.clips is not None:  # each audio token as its clip's rows: the state's, then a spoken question's
                first = expand(first, self.clips.token, [*state_clips, *heard[row[0].id]])
                alone = expand(alone, self.clips.token, heard[row[0].id])
                rest = [expand(own, self.clips.token, heard[q.id]) for q, own in zip(row[1:], rest, strict=True)]
            state = len(first) - len(alone)  # the first question's own tokens, counted alone, are the row's last
            ids, groups = list(first), [-1] * state + [1] * (len(first) - state)
            readouts = [Readout(row[0].id, len(ids) - 1)]
            for k, (q, own) in enumerate(zip(row[1:], rest, strict=True), 2):
                ids += own
                groups += [k] * len(own)
                readouts.append(Readout(q.id, len(ids) - 1))
            prompts.append(Prompt(tuple(ids), tuple(readouts), groups=tuple(groups)))
        labels = {q.id: self.label_tokens(q) for q in request.questions}
        return Rendering(
            request, tuple(prompts), {"labels": labels, "waves": waves}, pixels=pixels, clips=clips, videos=video_info
        )

    def encode_media(self, renderings: Sequence[Rendering]) -> dict[int, mx.array]:
        """Every picture's and clip's rows by digest (the last few of each kept)."""
        out = self.pictures.encode(renderings) if self.pictures is not None else {}
        if self.videos is not None:
            out |= self.videos.encode(renderings)
        return out | (self.clips.encode(renderings) if self.clips is not None else {})

    def temperature(self, q: Question) -> float:
        return float(self.temperatures.get(bucket(q), self.temperature_default))

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        out = {}
        for q in rendering.request.questions:  # each option's forms, read by the largest of their logits
            labels = LabelSet(rendering.meta["labels"][q.id])
            logits = label_logits(hidden.rows(q.id)[-1:], self.output, labels.flat)
            out[q.id] = option_probabilities(logits, labels, self.temperature(q), pool=strongest)[0]
        return out

    def answer(self, question: Question, scores: Any) -> tuple[tuple[float, ...], dict[str, Any]]:
        p = tuple(float(x) for x in scores)
        shaped = systemone_answer(question, p)
        if question.type == "score":
            shaped["score"] = float(median_level(p))
        return p, shaped


def strongest(grouped: mx.array) -> mx.array:
    """Each option's label logits [rows, options, tokens] combined as d1 reads its label variants: the largest."""
    return grouped.max(axis=-1)


def gguf_tensors(header: Header) -> tuple[list[Rule], int]:
    """A GGUF build's LFM3 text model, and its blocks."""
    return lfm3_rules(), int(header.get("block_count"))


class D1(AdapterBase):
    api = API_VERSION
    name = "liquid/d1"
    gguf_rules = Reads(gguf_tensors, lfm3_media_rules)
    files = (
        "config.json",
        "calibration.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "model*.safetensors",
        "*.safetensors.index.json",
    )

    def matches(self, model: ModelDir) -> bool:
        config = model.config
        return (
            isinstance(config, dict)
            and config.get("model_type") in ("lfm3_vl", "lfm3_omni")
            and config.get("architectures") == ["D1Model"]
        )

    def gguf_check(self, header: Header, model: ModelDir) -> list[str]:
        """A GGUF build's LFM3 against the config, its attention (set, both ways), and its decision type and
        temperatures against the checkpoint's calibration.json."""
        differ = lfm3_differences(header, model.config)
        if header.get("attention.causal") is not False:
            differ.append("its attention is not d1's set attention")
        decision = {"lfm3_vl": "lfm3-d1", "lfm3_omni": "lfm3-d1-omni"}.get(str(model.config.get("model_type")))
        if header.get("decision.type") != decision:
            differ.append(f"decision type {header.get('decision.type')!r}")
        t = float((model.json("calibration.json") or {}).get("default", 1.0))
        for kind in ("noul", "choice", "score"):
            if not close(header.get(f"decision.temperature.{kind}", 0.0), t):
                differ.append(f"{kind} temperature {header.get(f'decision.temperature.{kind}')} against {t}")
        return differ

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        config = model.config
        if config.get("d1_attention") != "set":
            raise UnsupportedModel(f"{model.path}: d1 attention {config.get('d1_attention')!r}; ippon reads 'set'")
        weights = ctx.weights(model)
        backbone = ctx.backbone(lfm3, model, weights=weights)
        calibration = model.json("calibration.json")
        bos = (model.json("tokenizer_config.json") or {}).get("bos_token") or ""
        tokenizer = ctx.tokenizer(model)
        picture_path = pictures(model, ctx, backbone, weights, tokenizer)
        fmt = D1Format(
            tokenizer,
            bos if isinstance(bos, str) else "",
            backbone.output_matrix,
            float(calibration.get("default", 1.0)),
            {k: float(t) for k, t in (calibration.get("temperatures") or {}).items()},
            int(config.get("d1_codes", 1)),
            int(config.get("d1_set_size", SET_SIZE)),
            picture_path,
            clips(model, ctx, backbone, weights, tokenizer),
            videos(model, picture_path, tokenizer) if picture_path is not None else None,
        )
        details = {"architecture": "lfm3 (set attention)", "temperature": fmt.temperature_default}
        return Loaded(backbone, fmt, details)


def videos(model: ModelDir, pictures: Pictures, tokenizer: Tokenizer) -> Videos:
    """The checkpoint's video path, using its processor settings and its picture tower."""
    p = (model.json("processor_config.json") or {}).get("video_processor", {})
    if p.get("video_processor_type") != "Lfm3VlVideoProcessor" or not p.get("do_sample_frames", True):
        raise UnsupportedModel("Liquid's video processor must be Lfm3VlVideoProcessor with frame sampling")
    if p.get("num_frames") is not None:
        raise UnsupportedModel("Liquid's video processor with a fixed num_frames is not supported")
    image = ImageConfig.from_hf({**p, "image_processor_type": "Lfm3VlImageProcessor"})
    VisionConfig.from_hf(model.config).check(image)
    config = VideoConfig(image, int(p.get("max_video_tokens", 768)) if p.get("cap_pixels_per_frame") else None)
    token = config_token(model, tokenizer, "<|video|>", "video_token_id")
    sampling = Sampling(float(p.get("fps", 2.0)), int(p.get("min_frames", 4)), int(p.get("max_frames", 768)))
    times = omni_times if model.config["model_type"] == "lfm3_omni" else frame_pair_times
    return Videos(config, sampling, token, pictures, decoder=load_frames, pad="<|video|>", times=times)


def pictures(model: ModelDir, ctx: LoadContext, backbone: Any, weights: Any, tokenizer: Tokenizer) -> Pictures | None:
    """The checkpoint's picture path, or None for a build without one (no processor_config.json or no vision tower
    among its weights)."""
    config = model.config
    if not model.has("processor_config.json") or not ctx.has_media(model, weights, TOWER + "patchifier.weight"):
        return None
    image = ImageConfig.from_hf(model.json("processor_config.json"))
    VisionConfig.from_hf(config).check(image)
    token = config_token(model, tokenizer, "<|image|>", "image_token_id")
    tower = encoder_loader(Lfm3Vision.load, model, ctx, backbone)
    return Pictures(image, True, token, tower, int(config.get("d1_vision_max_pixels") or MAX_PIXELS))


def clips(model: ModelDir, ctx: LoadContext, backbone: Any, weights: Any, tokenizer: Tokenizer) -> Clips | None:
    """The checkpoint's audio path, or None for a build without one (d1-2B: no speech encoder among its weights)."""
    if not ctx.has_media(model, weights, lfm3_audio.ENCODER + "downsample.conv_in.weight"):
        return None
    lfm3_audio.check_front_end(model.json("processor_config.json"))
    token = config_token(model, tokenizer, "<|audio|>", "audio_token_id")
    return Clips(token, encoder_loader(lfm3_audio.Lfm3Audio.load, model, ctx, backbone))
