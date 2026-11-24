"""Cloudflare Clef-flash in MLX, the community build (mlx-community/clef-flash-8bit), through the
repo's own clef_mlx.py as its card documents: model = clef_mlx.load(path); model.systemone(request).
Not Cloudflare's: listed as a community build.

    envs/mlxvlm/bin/python -I adapters/clef_mlx_community.py --engine clef-flash-mlx-8bit \
        --repo mlx-community/clef-flash-8bit --model ~/models/clef-flash-mlx-8bit --max-input 16384
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common, gen_requests
import common

OFFICIAL = ("mlx-community/clef-flash-",)


def main() -> None:
    p = common.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--model", required=True)
    p.add_argument("--repo", required=True)
    args = p.parse_args()
    if not args.repo.startswith(OFFICIAL):
        sys.exit(f"only the {OFFICIAL} builds")

    import mlx.core as mx
    import mlx_vlm

    model_dir = str(Path(args.model).expanduser())
    sys.path.insert(0, model_dir)
    import clef_mlx

    model = clef_mlx.load(model_dir)

    def decide(req):
        out = model.systemone({"model": "clef-flash", **req["systemone"]})
        return common.systemone_answers(out), out.get("usage", {}).get("input_tokens"), None

    def memory():
        return {"mlx_peak": mx.get_peak_memory(), "mlx_active": mx.get_active_memory()}

    meta = {
        "runtime": f"mlx-vlm {mlx_vlm.__version__} + mlx {mx.__version__}, the repo's clef_mlx.py",
        "repo": args.repo,
        "model_dir": model_dir,
        "revision": common.repo_revision(model_dir),
        "build": "community",
    }
    meta["question_method"] = "one systemone call containing all questions"
    common.run(args, decide, meta, memory=memory)


if __name__ == "__main__":
    main()
