"""A creator's own transformers path on MPS: AutoModel with the repo's remote code, model.system_one(state, questions).

    envs/d1/bin/python -I adapters/tf_systemone.py --engine d1-2B-tf --family d1 --repo LiquidAI/d1-2B \
        --model ~/models/d1-2B --dtype bfloat16 --device-map none --max-input 32768
    envs/vllmsr/bin/python -I adapters/tf_systemone.py --engine d3-mini-tf --family d3 \
        --model ~/models/d3-mini --max-input 32768

d1 (LiquidAI d1-2B, d1-Omni-2B): as their card runs them on a Mac, AutoModel.from_pretrained(dir,
trust_remote_code=True, dtype=bfloat16).to("mps"); system_one(state, questions, media=[pictures..., clip]), a clip
as its float samples and their rate (the stdlib's wave reader, PCM16 over 32,768), as their API takes a waveform.
d3 (vllm-sr d3-mini): AutoModel.from_pretrained(dir, trust_remote_code=True, device="mps"); its own
runtime defaults to CUDA else CPU, so the device is named. Numerics are fixed (bf16 backbone, fp32 readout).
trust_remote_code is used only for the official LiquidAI and vllm-sr checkpoints (the directories hold their
Hub downloads).
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common, gen_requests
import common

OFFICIAL = ("LiquidAI/", "vllm-sr/")


def main() -> None:
    p = common.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--model", required=True, help="a local download of an official LiquidAI or vllm-sr repo")
    p.add_argument("--repo", required=True, help="its Hub id (LiquidAI/... or vllm-sr/...), recorded and checked")
    p.add_argument("--family", required=True, choices=("d1", "d3"))
    p.add_argument("--dtype", default="bfloat16", help="d1 only: bfloat16, float16 or float32")
    p.add_argument("--device-map", default="auto", help="d1 only: passed to from_pretrained ('none' to omit)")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, help="d3 only: questions per forward pass (default: the runtime's, 8)")
    args = p.parse_args()
    if not args.repo.startswith(OFFICIAL):
        sys.exit(f"trust_remote_code is allowed only for {OFFICIAL}")

    import torch
    import transformers
    from transformers import AutoModel

    model_dir = str(Path(args.model).expanduser())
    if args.family == "d1":
        kwargs = {"trust_remote_code": True, "dtype": args.dtype}
        if args.device_map != "none":
            kwargs["device_map"] = args.device_map
        model = AutoModel.from_pretrained(model_dir, **kwargs)
        placed = str(next(model.parameters()).device)
        if not placed.startswith(args.device):
            model = model.to(args.device)
        model.eval()

        def call(state, questions, images, videos, audio):
            clip = [{"type": "audio", "audio": audio[0], "sampling_rate": audio[1]}] if audio is not None else []
            films = [{"type": "video", "path": path} for path in videos]  # Liquid's media.Reader takes video paths
            return model.system_one(state, questions, media=[*(images or ()), *films, *clip] or None)
    else:
        kwargs = {"trust_remote_code": True, "device": args.device}
        if args.batch_size:
            kwargs["batch_size"] = args.batch_size
        model = AutoModel.from_pretrained(model_dir, **kwargs)
        placed = str(model.runtime.device)

        def call(state, questions, images, videos, audio):
            if images or videos or audio is not None:
                raise ValueError("d3 reads text only")
            return model.system_one(state=state, questions=questions)

    device = str(next(model.parameters()).device) if args.family != "d3" else str(model.runtime.device)

    def media(body, folder):
        """A request's files as the creator's code takes them: PIL images, video files written to `folder`, or a
        clip's float samples and their rate."""
        images, videos, audio = [], [], None
        for url in body.get("files", []):
            head, data = url.split(",", 1)
            raw = io.BytesIO(base64.b64decode(data))
            if head.startswith("data:video/"):
                path = os.path.join(folder, f"video-{len(videos)}.mp4")
                with open(path, "wb") as out:
                    out.write(raw.getvalue())
                videos.append(path)
            elif head.startswith("data:image/"):
                from PIL import Image

                image = Image.open(raw)
                image.load()
                images.append(image)
            else:
                import wave

                import numpy as np

                with wave.open(raw, "rb") as clip:
                    pcm = np.frombuffer(clip.readframes(clip.getnframes()), "<i2").reshape(-1, clip.getnchannels())
                    audio = (pcm.T.astype(np.float32) / 32768, clip.getframerate())
        return images or None, videos, audio

    def decide(req):
        body = req["systemone"]
        with tempfile.TemporaryDirectory() as folder:  # its videos as files, removed once it is answered
            out = call(body["state"], body["questions"], *media(body, folder))
        errors = {k: a for k, a in out["answers"].items() if "error" in a}
        if errors:
            raise RuntimeError(f"answer errors: {errors}")
        return common.systemone_answers(out), out["usage"]["input_tokens"], None

    def memory():
        if device.startswith("mps"):
            return {
                "mps_driver_allocated": torch.mps.driver_allocated_memory(),
                "mps_current_allocated": torch.mps.current_allocated_memory(),
            }
        return None

    meta = {
        "runtime": f"transformers {transformers.__version__} + torch {torch.__version__} ({device})",
        "family": args.family,
        "repo": args.repo,
        "model_dir": model_dir,
        "revision": common.repo_revision(model_dir),
        "device": device,
        "placed_by_loader": placed,
        "dtype": args.dtype if args.family != "d3" else "bfloat16 backbone, float32 readout (fixed)",
        "device_map": args.device_map if args.family != "d3" else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_high_watermark_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
    }
    if args.family == "d3":
        rt = model.runtime
        meta.update(
            {
                "batch_size": rt.batch_size,
                "kernels": rt.kernels,
                "max_length": rt.max_length,
                "attention_mode": getattr(rt, "attention_mode", None),
            }
        )
    meta["question_method"] = (
        f"one system_one call; the creator batches up to {model.runtime.batch_size} questions per forward pass"
        if args.family == "d3"
        else "one system_one call containing all questions"
    )
    with torch.inference_mode():
        common.run(args, decide, meta, memory=memory)


if __name__ == "__main__":
    main()
