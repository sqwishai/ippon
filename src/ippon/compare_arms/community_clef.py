"""A `ippon bench --compare` arm: the community MLX build of Cloudflare's Clef (mlx-community/clef-flash-8bit)
through the repo's own loader, as its card documents: model = clef_mlx.load(path); model.systemone(request).
This runs the clef_mlx.py the model directory came with: read it before running it.

    PYTHON -I community_clef.py MODEL_DIR REQUESTS.json OUT.json      (PYTHON: an environment with mlx-vlm installed)

IPPON_ARM_DTYPE=float32 computes the build's own weights in fp32 (its head through load's head_dtype, its
backbone recast with MLX's set_dtype), the reference `ippon verify --against` checks the build against.
"""

import json
import os
import sys
import time
from importlib.metadata import version


def main() -> None:
    model_dir, requests, out = sys.argv[1:4]

    sys.path.insert(0, model_dir)
    import clef_mlx  # the model repository's own loader

    dtype = os.environ.get("IPPON_ARM_DTYPE")
    if dtype:
        import mlx.core as mx

        model = clef_mlx.load(model_dir, head_dtype=getattr(mx, dtype))
        model.backbone.set_dtype(getattr(mx, dtype))
    else:
        model = clef_mlx.load(model_dir)
    results = []
    with open(requests, encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        if item["gap"]:
            time.sleep(item["gap"])
        body = item["body"]
        # the port takes a choice's criteria only as {label: description}; a list's labels have none, which it
        # renders as ippon does (no "description" key)
        questions = {
            k: {**q, "criteria": dict.fromkeys(map(str, q["criteria"]))}
            if q.get("type") == "choice" and isinstance(q.get("criteria"), list)
            else q
            for k, q in body["questions"].items()
        }
        start = time.perf_counter()
        response = model.systemone({"model": "clef-flash", **body, "questions": questions})
        ms = (time.perf_counter() - start) * 1000
        answers = {}
        for name, a in response["answers"].items():
            if a.get("type") == "noul" or "noul" in a:
                answers[name] = {"yes": float(a["noul"]), "no": 1 - float(a["noul"])}
            else:
                answers[name] = {str(k): float(v) for k, v in a["probabilities"].items()}
        results.append({"ms": ms, "answers": answers, "input_tokens": response.get("usage", {}).get("input_tokens")})
    with open(out, "w", encoding="utf-8") as f:
        runtime = f"the repository's clef_mlx.py + mlx {version('mlx')}" + (f" ({dtype})" if dtype else "")
        json.dump({"runtime": runtime, "dtype": dtype, "results": results}, f)


if __name__ == "__main__":
    main()
