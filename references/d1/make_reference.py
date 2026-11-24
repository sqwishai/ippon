"""Liquid AI d1 reference answers on LFM3 (d1-2B, d1-Omni-2B), from the code each checkpoint ships, never inside ippon:

    python -I make_reference.py --model ~/models/d1-2B --repo LiquidAI/d1-2B --revision SHA \\
        --requests ../requests/standard.json requests.json --out d1-2B.json.gz [--tokens-only] [--media ../media] \\
        [--threads 8] [--only NAME ...] [--no-bf16] [--bf16-device mps]

Run it in an environment with the card's requirements (transformers >= 5.19, torch, torchvision, pillow); the
versions are recorded. The fp32 run is on the CPU (CUDA hidden); the bf16 run, for the bound, on `--bf16-device`
(MPS by default, as the card runs the model on a Mac).

Everything comes from the checkpoint's own code, imported the way transformers imports remote code from the local
folder (`trust_remote_code`):

- `--tokens-only` needs no weights: the creator's SystemOne state is built without a model (`bare_reader`), and each
  request goes through its own validation (`SystemOneApi.named`, which refuses what `system_one` refuses) and its own
  rendering and `set` packing (`SlotReader._sequences`): every row's token ids, its state length, each token's question
  (-1 on the state; None outside `set`), each question's answer slots, and each question's label tokens
  (`prompt.readout_ids`). A request with media also records what the processor made of it (grids, frames).
- Without it, the model is loaded as its card says (`AutoModel`, `trust_remote_code`), in fp32 on the CPU, and
  `model.engine.run` gives each question's probabilities in the request's option order (a noul as true, false: the
  creator's [P(yes), P(no)]) and the tokens it read, which must equal the rows' total; `engine.answer` shapes them as
  `system_one` does (a score's median level, the confidences, calibration tables for typed answers). The projectors'
  rows (pictures, clips) are kept, the first FIRST values of each. The model then runs again in bf16, and the bound
  is the larger of 0.01 and that run's distance from fp32 plus 0.0005, rounded to four decimals.

Requests follow references/requests/*.json: `state`, `questions`, `images` (file names), `audio` (one file name),
and a spoken question's `instructions` as {"type": "input_audio", "input_audio": {"file": NAME}} (ippon's form, its
clip named instead of embedded). Media names are looked up in `--media`,
then its `images/`, `audio/` and `videos/`. Pictures use transformers.image_utils.load_image, as the card does (EXIF
orientation applied, RGB); a clip goes in as its path, which the creator reads itself (16-bit PCM WAV only, resampled to
16 kHz, channels averaged). Media go in the order images, videos, then the clip. Videos go to the processor as paths,
decoded by TorchCodec in the creator environment, with no audio extraction. make_video_fixtures.py records its sampled
RGB frames, indices and patches for comparison with ippon's PyAV path. On M5 Pro, d1-2B and d1-Omni-2B's shared
processor fixtures (98 frames) match in all 120,342,816 RGB channels and every timestamp; OpenCV differs by up to
3 levels and uses a different clock on the variable-rate clip (README.md records the comparison).
A choice given as a list of labels, which the
creator refuses ("a list of levels is a score"), is asked as the map of those labels without descriptions, as ippon
reads it. A request the creator refuses is recorded with its message (`error`) and nothing else.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

FIRST = 8  # values kept from the start of every projector row
SCHEMA = "ippon-reference/1"


def maker_package(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint's own modules, imported as transformers imports remote code from a local folder (no weights
    read): its AutoModel class, and prompt, runner, media, calibration, attention and api from its package."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module(config["auto_map"]["AutoModel"], str(root))
    package = cls.__module__.rsplit(".", 1)[0]
    names = ("prompt", "runner", "media", "calibration", "attention", "api")  # calibration is imported lazily
    return {"model_class": cls, **{name: importlib.import_module(f"{package}.{name}") for name in names}}


def bare_reader(pkg: dict[str, Any], root: Path, config: Any, tokenizer: Any) -> Any:
    """runner.SystemOne's state without a model, set as its __init__ sets it: what `named` and `_sequences` read."""
    runner, prompt, calibration = pkg["runner"], pkg["prompt"], pkg["calibration"]
    reader = object.__new__(runner.SystemOne)
    reader.tokenizer = tokenizer
    bos = getattr(tokenizer, "bos_token", None)
    reader.bos = bos if isinstance(bos, str) else ""
    reader.lead = prompt.default_lead(config.model_type, tokenizer)  # LEADS: "<think></think>" for lfm3 (lfm3.py)
    found = root / "calibration.json"
    reader.calibration = calibration.Calibration.load(found) if found.is_file() else None
    reader.state_style, reader.system, reader.option_style = prompt.DEFAULT_STATE_STYLE, prompt.DEFAULT_SYSTEM, "desc"
    reader._serve(config)  # d1_codes, d1_numbers, d1_attention, d1_set_size, d1_vision_max_pixels from config.json
    reader.processor, reader.revision, reader.model = None, None, None
    reader._load_processor = lambda: runner.auto_processor(str(root))  # the processor its folder holds
    return reader


def find(media: Path | None, name: str) -> Path:
    for folder in [media, media / "images", media / "audio", media / "videos"] if media is not None else []:
        if (folder / name).is_file():
            return folder / name
    raise FileNotFoundError(f"{name}: not in --media ({media}) or its images/, audio/ and videos/")


def float_wav(path: Path) -> tuple[Any, int] | None:
    """A float32 WAV's samples [channels, samples] and rate, or None for any other WAV: the creator's reader takes
    16-bit PCM files only, and a caller with float samples hands its API the waveform and its rate."""
    import struct

    import numpy as np

    data = path.read_bytes()
    chunks, at = {}, 12
    while at + 8 <= len(data):
        tag, size = data[at : at + 4], struct.unpack_from("<I", data, at + 4)[0]
        chunks[tag] = data[at + 8 : at + 8 + size]
        at += 8 + size + (size & 1)
    code, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", chunks[b"fmt "])
    if (code, bits) != (3, 32):
        return None
    return np.frombuffer(chunks[b"data"], "<f4").reshape(-1, channels).T.copy(), rate


def clip_item(path: Path) -> dict[str, Any]:
    """A clip as the creator's API takes it: a 16-bit PCM WAV by its path (its own reader), a float32 one as its
    samples and rate (`float_wav`)."""
    wave = float_wav(path)
    return (
        {"type": "audio", "path": str(path)}
        if wave is None
        else {"type": "audio", "audio": wave[0], "sampling_rate": wave[1]}
    )


def liquid_request(case: dict[str, Any], media: Path | None) -> tuple[Any, dict[str, Any], list[Any]]:
    """A reference request as the creator's API takes it: (state, {name: question}, media items)."""
    from transformers.image_utils import load_image

    request = case["request"]
    items: list[Any] = [load_image(str(find(media, name))) for name in request.get("images") or ()]
    items.extend({"type": "video", "path": str(find(media, name))} for name in request.get("videos") or ())
    if request.get("audio") is not None:
        items.append(clip_item(find(media, request["audio"])))
    questions = {}
    for name, q in request["questions"].items():
        q = dict(q)
        if q.get("type") == "choice" and isinstance(q.get("criteria"), list):  # ippon's reading of a list of labels
            q["criteria"] = dict.fromkeys(str(c) for c in q["criteria"])
        said = q.get("instructions")
        if isinstance(said, dict) and said.get("type") == "input_audio":  # a spoken question, its clip by file name
            q["instructions"] = clip_item(find(media, said["input_audio"]["file"]))
        questions[name] = q
    return request.get("state"), questions, items


def option_order(probs: Any) -> Any:
    """The creator's probabilities are in the request's option order already: a noul's [P(yes), P(no)] is ippon's
    (true, false), a choice's follows its criteria, a score's its levels. A typed answer's (a read's dict, a box's
    rows) is kept as the creator gives it."""
    if isinstance(probs, list) and all(isinstance(p, (int, float)) for p in probs):
        return [float(p) for p in probs]
    return probs


def package_of(reader: Any) -> dict[str, Any]:
    """The modules of the package `reader`'s class comes from: the ones it calls, whatever imported them first."""
    package = type(reader).__module__.rsplit(".", 1)[0]
    names = ("prompt", "runner", "media", "calibration", "attention", "api")
    return {name: sys.modules[f"{package}.{name}"] for name in names}


def media_record(inputs: dict[str, Any]) -> dict[str, Any]:
    """What the processor made of a row's media: picture and video grids (t, h, w in patches), each clip's frames."""
    out: dict[str, Any] = {}
    for key in ("image_grid_thw", "video_grid_thw"):
        if key in inputs:
            out[key] = [list(map(int, g)) for g in inputs[key].tolist()]
    if "feature_attention_mask" in inputs:
        out["audio_frames"] = [int(n) for n in inputs["feature_attention_mask"].sum(-1).tolist()]
    return out


def tokens_case(pkg: dict[str, Any], reader: Any, state: Any, questions: dict[str, Any], items: list[Any]) -> dict:
    """The creator's rows for one request: validation as system_one does it, then `_sequences` (the `set` packing)."""
    prompt, media = pkg["prompt"], pkg["media"]
    named = reader.named(questions)  # ValueError for what the creator refuses
    names, qs = list(named), list(named.values())
    rows = reader._sequences([(state, qs, media.media_list(items))])
    prompts, readouts, segments, states, slots, recorded = [], [], [], [], {}, []
    for row, (_, first, ids, state_len, question, ends, inputs) in enumerate(rows):
        prompts.append(list(map(int, ids)))
        readouts.append([int(e) for own in ends for e in own])
        segments.append(None if question is None else list(map(int, question)))
        states.append(None if state_len is None else int(state_len))
        for k, own in enumerate(ends):
            slots[names[first + k]] = {"prompt": row, "positions": [int(e) for e in own]}
        recorded.append(media_record(inputs) if inputs else {})
    groups = {
        n: (None if prompt.typed(q) else prompt.readout_ids(reader.tokenizer, q))
        for n, q in zip(names, qs, strict=True)
    }
    entry: dict[str, Any] = {
        "prompts": prompts,
        "readouts": readouts,
        "segments": segments,  # per row: each token's question (1..N within the row), -1 on the state
        "state_lengths": states,
        "slots": slots,  # per question: its row and its answer positions in it
        "input_tokens": sum(len(p) for p in prompts),  # the creator's usage: every row's tokens
        "groups": groups,  # per question: each option's label tokens, max-pooled (None for a typed answer)
    }
    if any(recorded):
        entry["media_inputs"] = recorded
    return entry


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--requests", required=True, nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--media", type=Path, help="where the requests' image and audio names are (or its images/, audio/)")
    p.add_argument("--tokens-only", action="store_true", help="the creator's rows without the weights")
    p.add_argument("--no-bf16", action="store_true", help="skip the bf16 run (the bound is then 0.01)")
    p.add_argument("--bf16-device", default="mps", help="where the bf16 run goes (mps, as the card runs it on a Mac)")
    p.add_argument("--threads", type=int, help="CPU threads for torch (default: torch's own)")
    p.add_argument("--only", action="append", help="run only the requests with this name (repeatable)")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the CPU only: a GPU may be timing something else
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the local folder, never the Hub
    # runner.py reads this at import and lets it override the checkpoint's picture cap (d1_vision_max_pixels)
    if os.environ.pop("D1_VISION_MAX_PIXELS", None) is not None:
        print("D1_VISION_MAX_PIXELS was set and is ignored: the checkpoint's own cap applies", file=sys.stderr)

    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    if args.threads:
        torch.set_num_threads(args.threads)
    versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    from importlib.metadata import PackageNotFoundError, version

    with contextlib.suppress(PackageNotFoundError):
        versions["torchcodec"] = version("torchcodec")
    try:
        import torchvision

        versions["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) < (5, 19):
        print(f"transformers {transformers.__version__}: the card asks for >= 5.19", file=sys.stderr)

    root = args.model.expanduser()
    raw = json.loads((root / "config.json").read_text())
    if raw.get("model_type") not in ("lfm3_vl", "lfm3_omni") or raw.get("architectures") != ["D1Model"]:
        raise SystemExit(f"{root}: not a d1 checkpoint on LFM3 (model_type {raw.get('model_type')!r})")
    pkg = maker_package(root, raw)
    config = AutoConfig.from_pretrained(str(root), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(root))

    model = None
    if args.tokens_only:
        reader = bare_reader(pkg, root, config, tokenizer)
    else:
        if not any(root.glob("*.safetensors")):
            raise SystemExit(f"{root}: no weights; download model.safetensors at the pinned revision first")
        from transformers import AutoModel

        model = AutoModel.from_pretrained(str(root), trust_remote_code=True, dtype=torch.float32).to("cpu").eval()
        assert all(t.device.type == "cpu" for t in [*model.parameters(), *model.buffers()])
        reader = model.engine  # runner.SystemOne over the model, with the folder's calibration.json
    pkg = package_of(reader)  # the modules the reader calls (its Media class, its prompt.py)

    # the projectors' rows, kept as each forward makes them (fp32 run only)
    captured: list[dict[str, Any]] = []
    hooks = []
    if model is not None:
        body = model.model
        for name in ("multi_modal_projector", "audio_projector"):
            part = getattr(body, name, None)
            if part is not None:

                def keep(module: Any, inputs: Any, output: Any, name: str = name) -> None:
                    rows = output.detach().float().cpu()
                    captured.append({"part": name, "rows": int(rows.shape[0]), "first": rows[:, :FIRST].tolist()})

                hooks.append(part.register_forward_hook(keep))

    cases = [c for f in args.requests for c in json.loads(f.read_text()) if not args.only or c["name"] in args.only]
    out: list[dict[str, Any]] = []
    for case in cases:
        state, questions, items = liquid_request(case, args.media.expanduser() if args.media else None)
        try:
            entry = tokens_case(pkg, reader, state, questions, items)
        except ValueError as e:  # the creator's own refusal (check_question, a box without one picture, ...)
            out.append({"name": case["name"], "request": case["request"], "error": str(e)})
            print(f"{case['name']}: refused ({e})", file=sys.stderr)
            continue
        entry = {
            "name": case["name"],
            "request": case["request"],
            **({"media": case["media"]} if "media" in case else {}),
            **entry,
        }
        if model is not None:
            captured.clear()
            named = reader.named(questions)
            qs = list(named.values())
            with torch.inference_mode():
                ((probs, read),) = reader.run([(state, qs, items)])  # run turns the items into its own Media
            assert read == entry["input_tokens"], (case["name"], read, entry["input_tokens"])
            entry["probabilities"] = {n: option_order(pr) for n, pr in zip(named, probs, strict=True)}
            entry["answers"] = {n: reader.answer(state, q, pr) for (n, q), pr in zip(named.items(), probs, strict=True)}
            if captured:
                entry["projector_rows"] = list(captured)
        out.append(entry)
        print(f"{case['name']}: {len(entry['prompts'])} row(s), {entry['input_tokens']} tokens", file=sys.stderr)
    for h in hooks:
        h.remove()

    bound = 0.01
    if model is not None and not args.no_bf16:  # the model's own bf16, for the bound
        model.to(device=args.bf16_device, dtype=torch.bfloat16)
        reader.device = torch.device(args.bf16_device)  # the runner puts its inputs where it was made (runner.py)
        distance = 0.0
        for entry in (e for e in out if "error" not in e):
            state, questions, items = liquid_request(entry, args.media.expanduser() if args.media else None)
            named = reader.named(questions)
            with torch.inference_mode():
                ((probs, _),) = reader.run([(state, list(named.values()), items)])
            entry["probabilities_bf16"] = {n: option_order(pr) for n, pr in zip(named, probs, strict=True)}
            for n, got in entry["probabilities_bf16"].items():
                if isinstance(got, list) and isinstance(entry["probabilities"][n], list):
                    pairs = zip(got, entry["probabilities"][n], strict=True)
                    distance = max(distance, *(abs(a - b) for a, b in pairs))
        bound = max(0.01, round(distance + 5e-4, 4))

    calibration = getattr(reader, "calibration", None)
    reference = {
        "schema": SCHEMA,
        "adapter": "liquid/d1",
        "model": {"repo": args.repo, "revision": args.revision, "model_type": raw["model_type"]},
        "maker": {
            "code": "the checkpoint's own (trust_remote_code, runner.SystemOne)",
            "device": "cpu",
            "dtype": "float32",
            "bf16_device": None if args.tokens_only or args.no_bf16 else args.bf16_device,
            "versions": versions,
        },
        "layout": {  # how the rows are read: `set` attention, up to set_size questions after their state
            "attention": reader.attention,
            "set_size": reader.set_size,
            "codes": getattr(reader.tokenizer, "d1_codes", 1),
            "lead": reader.lead,
            "max_pixels": reader.max_pixels,
        },
        "temperature": None if calibration is None else calibration.default,
        "bound": bound,
        "tokens_only": args.tokens_only,
        "requests": out,
    }
    if any(
        "media" in e for e in out
    ):  # where verify finds the files, and the tokens the creator's prompts hold for rows
        names = {m["name"] for e in out for m in e.get("media", ())}
        paths = [find(args.media.expanduser(), name) for name in names]
        relative = all((args.media.expanduser() / name).is_file() for name in names)
        folders = {path.parent.name for path in paths}
        if not relative and len(folders) != 1:
            raise SystemExit(f"the requests' media are in {sorted(folders)}: use paths relative to --media")
        reference["media_format"] = {
            "directory": "references/media" if relative else f"references/media/{folders.pop()}",
            "decode": "pictures: transformers.image_utils.load_image, as the card loads them (EXIF orientation "
            "applied, RGB); clips: the creator's reader for 16-bit PCM WAV, a float32 WAV's samples handed over with "
            "their rate; videos: the checkpoint's processor, TorchCodec exact-seek CPU decoding; no soundtrack",
            "image_token": config.image_token_id,
            **({"audio_token": config.audio_token_id} if hasattr(config, "audio_token_id") else {}),
            **({"video_token": config.video_token_id} if hasattr(config, "video_token_id") else {}),
        }
    data = json.dumps(reference, ensure_ascii=False, default=float).encode()
    args.out.write_bytes(gzip.compress(data) if args.out.suffix == ".gz" else data)
    refused = sum("error" in e for e in out)
    print(
        f"{len(out)} requests, {refused} refused -> {args.out} (bound {bound}"
        f"{', tokens only' if args.tokens_only else ''})"
    )


if __name__ == "__main__":
    main()
