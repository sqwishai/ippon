"""Record Liquid's own video decoder and processor, without a tower, in its environment.

python references/d1/make_video_fixtures.py --model ~/models/d1-2B
The stored RGB frames and patches come only from the creator's processor (TorchCodec); tests compare ippon's decoder,
clock, resize and temporal patches with them. This script executes the checkpoint's own Python code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from make_reference import bare_reader, maker_package
from transformers import AutoConfig, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "video-processor")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    model = a.model.expanduser()
    raw = json.loads((model / "config.json").read_text())
    reader = bare_reader(
        maker_package(model, raw),
        model,
        AutoConfig.from_pretrained(str(model), trust_remote_code=True),
        AutoTokenizer.from_pretrained(str(model)),
    )
    vp = reader.media.processor.video_processor
    cases = json.loads((root / "references/requests/media-d1-video.json").read_text())
    names = sorted({name for c in cases for name in c["request"]["videos"]})
    names.append("videos/bar-h264-1280x720-29.97fps-20s.mp4")
    arrays, records = {}, []
    for i, name in enumerate(names):
        path = root / "references/media" / name
        frames, metadata = vp.fetch_videos(str(path), sample_indices_fn=vp.sample_frames)
        rgb = frames.permute(0, 2, 3, 1).numpy()
        out = vp(videos=[str(path)], return_metadata=True, return_tensors="pt")
        arrays[f"frames_{i}"] = rgb
        arrays[f"patches_{i}"] = out["pixel_values_videos"].numpy()
        records.append(
            {
                "name": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "indices": metadata.frames_indices.tolist(),
                "fps": metadata.fps,
                "total": metadata.total_num_frames,
                "grid": out["video_grid_thw"][0].tolist(),
            }
        )
        print(name, rgb.shape, records[-1]["grid"], flush=True)
    # The sampler's floor and cap, without decoding minutes of synthetic footage.
    from transformers.video_utils import VideoMetadata

    samples = []
    for total, fps in [(1, 24.0), (2, 24.0), (5, 5.0), (25, 10.0), (10000, 24.0)]:
        indices = vp.sample_frames(VideoMetadata(total_num_frames=total, fps=fps))
        samples.append({"total": total, "fps": fps, "indices": indices.tolist()})
    a.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "versions": {k: version(k) for k in ("torch", "torchcodec", "transformers", "torchvision")},
                "sources": {
                    name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                    for name in (
                        "media.py",
                        "video_processing_lfm3_vl.py",
                        "processing_lfm3_vl.py",
                        "processor_config.json",
                    )
                },
                "processor": json.loads((model / "processor_config.json").read_text())["video_processor"],
                "clips": records,
                "sampling": samples,
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(a.out.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
