"""Torchcast Decision's reference answers, from the creator's own serving package (torchcast_decision, shipped in the
model repository, Apache-2.0), never inside ippon:

    python -I make_reference.py --model path/to/torchcast-decision-4b --requests requests.json \\
        --repo torchcast-ai/torchcast-decision-4b --revision SHA --out torchcast-decision-4b.json.gz [--tokens-only]

Run it in an environment with the creator's pins (torch 2.9.1, transformers 5.8.1). It loads the checkpoint with
`TorchcastDecision(root, device="cpu", graphs=False, images=False)`, takes every question's prompt tokens from the
package's `render.prompt_ids` (through `TorchcastDecision._prompt`), and scores them with its `probabilities` (the
letter rows of the output matrix at the last token, softmaxed at the question type's temperature), with the backbone
in fp32 and the package's bf16 autocast off; then again as the package runs on a Mac, its backbone in bf16 under bf16
autocast on `--bf16-device`, for the bound. Probabilities are the model's, before the server's yes/no floor (a noul as
true, false): the floor jumps at 0.5, so a near tie's would part fp32 from bf16 by 0.6. A request the package answers
without a model pass (a choice of one option) or in rounds (a choice of more than 26 options) is left out, with its
reason: ippon reads one pass of at most 26 options.

With --media DIR (the folder the requests' image names are in), the requests' images are read as the package's
`decide(state, questions, images)` reads them, its vision tower on (`images=True`, the server's --max-pixels default
of 1,048,576 pixels): `_embed_images` (read_image's decode: no EXIF turn, alpha dropped; the checkpoint's processor
with the package's pixel range), `_state_with_images` (each "<image>" mark of the state, else the images ahead of it),
then each question's prompt from `_image_prompt` (the chat template's image parts, each image token expanded to its
rows, with get_rope_index's multimodal positions, which are recorded) and its `probabilities`; fp32 with the package's
bf16 autocast off (its vision tower too), then bf16 under its autocast on --bf16-device, for the bound. A request the
package refuses (marks that do not match the images) is recorded with its error. With --tokens-only the model is
built on the meta device (no weights read) and only the prompts are recorded:

    python -I make_reference.py --model DIR --requests ../requests/media-torchcast.json --media ../media/images \\
        --repo torchcast-ai/torchcast-decision-4b --revision SHA --out torchcast-decision-4b-media.json.gz
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from runpy import run_path
from typing import Any

# Load just the stdlib serializer without importing ippon or changing runtime module discovery.
dumps = run_path(str(Path(__file__).resolve().parents[2] / "src/ippon/_portable.py"))["dumps"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, type=Path, help="a downloaded torchcast-decision checkpoint")
    p.add_argument("--requests", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--repo", required=True, help="the checkpoint's Hub repository, for the tests' tokenizer")
    p.add_argument("--revision", required=True, help="the commit the checkpoint was downloaded at")
    p.add_argument("--tokens-only", action="store_true")
    p.add_argument("--bf16-device", default="mps", help="where the bf16 run for the bound happens")
    p.add_argument("--only", nargs="*", help="request names to run")
    p.add_argument("--media", type=Path, help="the folder of the image files the requests name")
    args = p.parse_args()
    root = args.model.expanduser().resolve()
    sys.path.insert(0, str(root))
    import torch
    import transformers
    from torchcast_decision import model as maker
    from torchcast_decision import render

    if args.media is not None:
        media_reference(args, root)
        return
    tc = maker.TorchcastDecision(str(root), device="cpu", graphs=False, images=False)
    cases = json.loads(args.requests.read_text())
    if args.only:
        if missing := set(args.only) - {c["name"] for c in cases}:
            p.error(f"unknown request names: {sorted(missing)}")
        cases = [c for c in cases if c["name"] in args.only]

    out, left_out = [], []
    for case in cases:
        request = case["request"]
        try:
            settled, pending, _ = tc._sort(request.get("state"), request["questions"])
        except ValueError as e:  # a request the package refuses is not a reference case
            left_out.append({"name": case["name"], "why": str(e)})
            continue
        if settled:
            left_out.append({"name": case["name"], "why": "answered without one pass of at most 26 options"})
            continue
        prompts = [tc._prompt(q)[0] for _, q in pending]
        out.append(
            {
                "name": case["name"],
                "request": request,
                "keys": [key for key, _ in pending],
                "prompts": prompts,
                "input_tokens": sum(len(ids) for ids in prompts),
            }
        )
    if not out:
        p.error("the package refused every request")

    def score(key: str) -> None:
        for entry in out:
            request = entry["request"]
            _, pending, _ = tc._sort(request.get("state"), request["questions"])
            probabilities, used = tc.probabilities([q for _, q in pending])
            assert used == entry["input_tokens"], entry["name"]
            entry[key] = {k: [float(x) for x in ps] for (k, _), ps in zip(pending, probabilities, strict=True)}
            print(f"{key} {entry['name']}: {used} tokens", flush=True)

    if not args.tokens_only:
        autocast = torch.autocast
        torch.autocast = lambda *a, **k: contextlib.nullcontext()  # fp32 throughout: the package's autocast off
        try:
            tc.body.float()
            with torch.inference_mode():
                score("probabilities")
        finally:
            torch.autocast = autocast
        tc.body.to(torch.bfloat16)  # the published bf16 weights again, exactly
        tc.device = torch.device(args.bf16_device)
        tc.body.to(tc.device)
        tc.letter_rows = tc.letter_rows.to(tc.device)
        with torch.inference_mode():
            score("probabilities_bf16")
    bound = 0.01
    if not args.tokens_only:
        distance = max(
            abs(a - b)
            for e in out
            for q in e["probabilities"]
            for a, b in zip(e["probabilities"][q], e["probabilities_bf16"][q], strict=True)
        )
        bound = max(0.01, round(distance + 5e-4, 4))
    if not math.isfinite(bound):
        raise SystemExit("non-finite probabilities")
    config = json.loads((root / "decision_config.json").read_text())
    reference: dict[str, Any] = {
        "schema": "ippon-reference/1",
        "adapter": "torchcast/decision",
        "model": {"repo": args.repo, "revision": args.revision, "model_name": config.get("model_name")},
        "maker": {
            "code": "torchcast_decision (render.prompt_ids, TorchcastDecision.probabilities)",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "render": render.VERSION,
            "temperature": tc.temperature,
            "noul_floor": tc.noul_floor,
            "left_out": left_out,
            "device": "cpu",
            "dtype": "float32",
            "bf16_device": args.bf16_device,
        },
        "bound": bound,
        "tokens_only": args.tokens_only,
        "requests": out,
    }
    data = dumps(reference, ensure_ascii=False).encode()
    args.out.write_bytes(gzip.compress(data) if args.out.suffix == ".gz" else data)
    print(f"{len(out)} requests -> {args.out} (bound {bound}{', tokens only' if args.tokens_only else ''})")


def media_reference(args: Any, root: Path) -> None:
    """The image requests (--media): see the module's docstring."""
    import torch
    import transformers
    from torchcast_decision import model as maker
    from torchcast_decision import render

    if args.tokens_only:  # the package's own code, its model on the meta device: no weights read
        built = getattr(transformers, json.loads((root / "config.json").read_text())["architectures"][0])

        def on_meta(path: str, *_: Any, **__: Any) -> Any:
            with torch.device("meta"):
                return built(transformers.AutoConfig.from_pretrained(path))

        built.from_pretrained = on_meta
    tc = maker.TorchcastDecision(str(root), device="cpu", graphs=False, images=True)
    if tc.vision is None:
        raise SystemExit(f"no image path: {tc.vision_note}")
    cases = json.loads(args.requests.read_text())
    if args.only:
        cases = [c for c in cases if c["name"] in args.only]

    def files(case: dict[str, Any]) -> list[bytes]:
        named = {m["name"]: m["sha256"] for m in case.get("media", [])}
        out = []
        for name in case["request"]["images"]:
            raw = (args.media / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != named[name]:
                raise SystemExit(f"{args.media / name}: not the file the request names (sha256)")
            out.append(raw)
        return out

    def embed(raw: list[bytes]) -> dict[str, Any]:
        if not args.tokens_only:
            return dict(tc._embed_images(raw))
        # _embed_images without its tower pass (the features are not needed for the prompts)
        grid = tc.image_processor(images=[maker.read_image(x) for x in raw], return_tensors="pt")["image_grid_thw"]
        return {
            "features": None,
            "grid": grid,
            "tokens": (grid.prod(-1) // tc.vision.visual.spatial_merge_size**2).tolist(),
        }

    def asked(case: dict[str, Any]) -> tuple[list[tuple[str, Any]], list[Any]]:
        """decide()'s steps for one request: (its questions, their prompts), with tc._images set until scored."""
        raw = files(case)
        tc._images = embed(raw)
        state = tc._state_with_images(case["request"].get("state"), len(raw))
        settled, pending, _ = tc._sort(state, case["request"]["questions"])
        if settled:
            raise LookupError("answered without one pass of at most 26 options")
        return pending, [tc._prompt(q) for _, q in pending]

    out, left_out = [], []
    for case in cases:
        entry: dict[str, Any] = {"name": case["name"], "request": case["request"], "media": case["media"]}
        try:
            pending, prompts = asked(case)
        except ValueError as e:  # the package refuses it: ippon must too
            entry["error"] = str(e)
            out.append(entry)
            print(f"{case['name']}: refused ({e})", flush=True)
            continue
        except LookupError as e:
            left_out.append({"name": case["name"], "why": str(e)})
            continue
        finally:
            tc._images = None
        pixels = tc.image_processor(images=[maker.read_image(x) for x in files(case)], return_tensors="pt")
        entry.update(
            keys=[key for key, _ in pending],
            prompts=[list(ids) for ids, _, _ in prompts],
            positions=[positions.tolist() for _, positions, _ in prompts],
            image_grids=pixels["image_grid_thw"].tolist(),
            pixel_values_sha256=hashlib.sha256(
                pixels["pixel_values"].float().numpy().astype("<f4").tobytes()
            ).hexdigest(),
            input_tokens=sum(len(ids) for ids, _, _ in prompts),
        )
        out.append(entry)
        print(f"{case['name']}: {entry['input_tokens']} tokens", flush=True)

    def score(key: str) -> None:
        for entry in out:
            if "error" in entry:
                continue
            case = next(c for c in cases if c["name"] == entry["name"])
            try:
                pending, _ = asked(case)
                probabilities, used = tc.probabilities([q for _, q in pending])
            finally:
                tc._images = None
            assert used == entry["input_tokens"], entry["name"]
            entry[key] = {k: [float(x) for x in ps] for (k, _), ps in zip(pending, probabilities, strict=True)}
            print(f"{key} {entry['name']}: {used} tokens", flush=True)

    def write(bound: float, distance: float | None) -> None:
        config = json.loads((root / "decision_config.json").read_text())
        reference: dict[str, Any] = {
            "schema": "ippon-reference/1",
            "adapter": "torchcast/decision",
            "model": {"repo": args.repo, "revision": args.revision, "model_name": config.get("model_name")},
            "maker": {
                "code": "torchcast_decision (TorchcastDecision.decide's image path: _embed_images, "
                "_state_with_images, _image_prompt, probabilities)",
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "render": render.VERSION,
                "temperature": tc.temperature,
                "noul_floor": tc.noul_floor,
                "image_processor": type(tc.image_processor).__name__,
                "image_size": dict(tc.image_processor.size),
                "left_out": left_out,
                "device": "cpu",
                "dtype": "float32",
                "bf16_device": args.bf16_device,
                "bf16_distance": distance,
            },
            "bound": bound,
            "tokens_only": args.tokens_only,
            "media_format": {
                "directory": "references/media/images",
                "decode": "model.read_image: PIL.Image.open, convert('RGB'), no EXIF turn",
                "image_token": int(tc.image_token),
                "prompts": "the package's prompt ids, each image token expanded to its image's rows (ippon's "
                "placeholders, arch/common.media_id, stand where these ids are image_token)",
                "positions": "per prompt [3][tokens]: the package's get_rope_index (t, h, w) position of every id",
                "images": "per request, its image_grid_thw and the sha256 of its pixel_values rows (float32, "
                "little-endian)",
            },
            "requests": out,
        }
        data = dumps(reference, ensure_ascii=False).encode()
        args.out.write_bytes(gzip.compress(data) if args.out.suffix == ".gz" else data)
        print(f"{len(out)} requests -> {args.out} (bound {bound})", flush=True)

    if args.tokens_only:
        write(0.01, None)
        return
    autocast = torch.autocast
    torch.autocast = lambda *a, **k: contextlib.nullcontext()  # fp32 throughout: the package's autocast off
    try:
        tc.vision.float()  # the vision tower and the decoder (tc.body is its language model)
        with torch.inference_mode():
            score("probabilities")
    finally:
        torch.autocast = autocast
    write(0.01, None)
    tc.vision.to(torch.bfloat16)  # the published bf16 weights again, exactly
    tc.device = torch.device(args.bf16_device)
    tc.vision.to(tc.device)
    tc.letter_rows = tc.letter_rows.to(tc.device)
    with torch.inference_mode():
        score("probabilities_bf16")
    distance = max(
        abs(a - b)
        for e in out
        if "probabilities" in e
        for q in e["probabilities"]
        for a, b in zip(e["probabilities"][q], e["probabilities_bf16"][q], strict=True)
    )
    if not math.isfinite(distance):
        raise SystemExit("non-finite probabilities")
    write(max(0.01, round(distance + 5e-4, 4)), distance)


if __name__ == "__main__":
    main()
