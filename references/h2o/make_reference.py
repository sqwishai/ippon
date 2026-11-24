"""H2O-Lightning-4B reference answers from the creator's own shim, with a stand-in for its vLLM backend, never inside
ippon.

    VENV=path/to/an/environment/with/vllm's/pins
    $VENV/bin/python -I h2o_reference.py --model ~/models/h2o-lightning-4b \\
        --requests references/requests/standard.json \\
        --out h2o-lightning-4b.json.gz --repo h2oai/h2o-lightning-4b \\
        --revision d7f1b05aa0b4e789b3896116f309ef842440e260 [--mode bidirectional|causal] [--media DIR] \\
        [--tokens-only] [--bf16] [--max-model-len 40960] [--threads N]

Image requests (../requests/media-h2o.json) take --media ../media/images, the folder their file names are in.

The pplx venv has torch 2.14.0, torchvision 0.29.0, transformers 5.17.0 and pillow 12.3.0. 5.17.0 is the transformers
release tokenizer_contract.json was written with. The reference runs on the CPU (CUDA hidden) in fp32: about 18 GB of
weights; --bf16 then reads again in bf16 on the Mac's GPU (--bf16-device, MPS), the output rows kept in fp32, for the
bound.

What runs:

- The shim itself, unmodified, loaded by path from the checkpoint (h2o_lightning_shim.py; it loads its sha256-pinned
  siblings selector_v2.py and box.py). It is built from the shipped serve_config.json with no SHIM_* or LIGHTNING_*
  override but --max-images (the shim's SHIM_MAX_IMAGES, recorded on the requests it answers), and every request goes
  through `Shim.decide(body)`. So the following are all the shim's own: the prompt
  text, the option lines, the image lifting, the long-record fit (selector v2 at --max-model-len, serve.sh's MAX_LEN
  of 40,960), the temperatures, the yes/no floor, the confidences and the usage.
- `Engine` stands in for the vLLM the shim talks to: the methods of h2o_lightning_shim.VLLM that the shim calls.
  - Tokenizer: vLLM's tokenizer as serve.sh serves it, AutoTokenizer on <model>/tokenizer, which loads as
    Qwen2Tokenizer.
  - /tokenize and /detokenize.
  - The label log-probabilities at the answer slot, as vLLM's `logprob_token_ids` returns them. Each comes from
    Qwen3_5ForConditionalGeneration in fp32: the final, normed hidden state at the last prompt position, times the
    tied output matrix in fp32 (config.json head_dtype float32), log-softmax over the whole vocabulary.
  - A read flagged `attention_mode: bidirectional` (serve.sh's default one engine flags every two-way read) gives the
    full-attention layers a mask that keeps every key. The Gated DeltaNet layers stay causal, as the card describes
    the h2oai/vllm build.
- The image path, for images given in the request:
  - The chat messages the shim builds are rendered by the tokenizer's chat template with continue_final_message and
    thinking off, as vLLM's chat completions renders them.
  - Each <|image_pad|> is expanded to its image's rows.
  - Pixels come from the checkpoint's Qwen2VLImageProcessor, built with min 65,536 pixels (preprocessor_config.json)
    and the max the shim sends (1,048,576).
  - The 3-D RoPE positions come from Qwen3_5Model.
  - Each image is decoded as the h2oai/vllm build serve.sh pins (b05cc29ab813895947954c268c4ec41ca0da1306, vLLM
    0.30's vllm/multimodal/media/image.py ImageMediaIO.load_bytes and vllm/multimodal/image.py): PIL.Image.open, turned
    by its EXIF orientation (normalize_image), loaded, then RGB as it is, RGBA composited on white, any other mode with
    transparency (LA, PA, a palette or grayscale with a transparent colour) converted to RGBA and composited on white,
    and the rest converted to RGB.

The shim checks this stand-in before its first two-way answer (Shim.check_two_way_flag). The flagged readouts of its
three fixed probes must be within 0.05 of the release's recorded two-way answers (serve_config.json
two_way.probe_reference). They must also differ from the unflagged readouts by at least 0.05. Before the first answer
of any kind, it also checks that the tokenizer reproduces all 24 tokenizer_contract.json probes and that each of the
255 labels is one token at the answer slot. A failed check stops the run.

Recorded for each request:

- For each question: its prompt token ids as the engine read them (images expanded), the label token ids, and its
  probabilities before the yes/no floor: softmax(logprob / T) in the request's option order (a noul as true, false;
  a choice in the request's order; a score in level order). The floor jumps at 0.5, so a near tie's floored answer
  would part fp32 from bf16 by 0.6; the shim's floored answers are recorded beside them.
- The shim's response: its answers (after the floor), its usage (input_tokens counts the shared head once), its
  attention_mode, and the state selection, if any.

A choice given as a list of labels, which the shim refuses (it takes a map), is asked as the map of those labels
without descriptions, as ippon reads it. A request the shim refuses is recorded with its status (400 or 422) and
detail. A request with a `box` question is
left out: neither ippon nor llama.cpp has that type. --tokens-only loads no weights and records only the prompts. The
prompts do not depend on the attention mode, so it reads them one-way and skips the two-way probe check.

"""

from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from runpy import run_path
from typing import Any

# Load just the stdlib serializer without importing ippon or changing runtime module discovery.
dumps = run_path(str(Path(__file__).resolve().parents[2] / "src/ippon/_portable.py"))["dumps"]
DEFAULT_MAX_PIXELS = 1_048_576  # serve.sh's --mm-processor-kwargs and serve_config.json image.max_pixels
MIN_BOUND = 0.01


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_environment() -> None:
    """The shim reads SHIM_* and LIGHTNING_* settings from the environment, some of them at import: none may leak
    in. Then set serve.sh's default deployment: one engine, each two-way read flagged."""
    for key in [k for k in os.environ if k.startswith(("SHIM_", "LIGHTNING_"))]:
        del os.environ[key]
    os.environ["SHIM_TWO_WAY_FLAG"] = "1"  # serve.sh, ONE_ENGINE=1 (h2o_lightning_shim.py:1379)
    os.environ["SHIM_QUESTION_THREADS"] = "1"  # one model behind the stand-in: questions are read one at a time
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_shim(root: Path) -> Any:
    """h2o_lightning_shim.py from the checkpoint, by path (its siblings are found beside it and checked by sha256)."""
    spec = importlib.util.spec_from_file_location("h2o_lightning_shim", root / "h2o_lightning_shim.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["h2o_lightning_shim"] = module
    spec.loader.exec_module(module)
    return module


def two_way_mask(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """A flagged (two-way) read: the full-attention layers see every key of the prompt, with no causal mask, and the
    linear-attention (Gated DeltaNet) layers stay causal (no padding mask: one unpadded sequence)."""
    import torch

    if args or kwargs.get("past_key_values") is not None:
        raise RuntimeError("the two-way mask expects keyword inputs and no cache")
    x = kwargs.get("inputs_embeds")
    x = kwargs["input_ids"] if x is None else x
    keys = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)[:, None, None, :]
    kwargs["attention_mask"] = {"full_attention": keys, "linear_attention": None}
    return args, kwargs


class Engine:
    """The vLLM 0.30 server serve.sh starts, as the shim calls it (the methods of h2o_lightning_shim.VLLM it uses),
    on transformers in fp32 on the CPU."""

    def __init__(self, shim: Any, root: Path, served: str, max_model_len: int, weights: bool):
        import torch
        from transformers import AutoTokenizer

        self.maker, self.torch = shim, torch
        self.base = "transformers stand-in"  # VLLM.base, quoted in the shim's messages
        self.root, self.served, self.max_model_len = root, served, max_model_len
        self.tok = AutoTokenizer.from_pretrained(
            str(root / "tokenizer")
        )  # serve.sh: vllm --tokenizer <model>/tokenizer
        if type(self.tok).__name__ != "Qwen2Tokenizer":
            raise SystemExit(f"<model>/tokenizer loads as {type(self.tok).__name__}, not serve.sh's Qwen2Tokenizer")
        config = json.loads((root / "config.json").read_text())
        self.image_token = int(config["image_token_id"])
        pre = json.loads((root / "preprocessor_config.json").read_text())
        self.min_pixels = int(pre["size"]["shortest_edge"])  # 65,536
        self.merge = int(pre["merge_size"])
        self.processors: dict[int, Any] = {}
        self.lock = threading.Lock()
        self.model: Any = None
        self.device = torch.device("cpu")
        self.head: Any = None  # the output rows the logits are read with, fp32
        self.vocab = len(self.tok)
        if weights:
            self.load()

    def load(self) -> None:
        from transformers import Qwen3_5ForConditionalGeneration

        torch = self.torch
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(self.root), dtype=torch.float32, attn_implementation="sdpa"
        )  # on the CPU (no device_map, CUDA hidden)
        model.eval()
        text = model.model.language_model
        if model.lm_head.weight.data_ptr() != text.embed_tokens.weight.data_ptr():
            raise SystemExit("expected the output matrix tied to the input embedding (tie_word_embeddings)")
        self.model, self.head = model, model.lm_head.weight
        self.vocab = int(self.head.shape[0])

    def bf16(self, device: str) -> None:
        """The backbone and vision tower in bf16 on `device`, the output rows kept in fp32 on the CPU: vLLM's bf16
        weights with config.json's head_dtype float32 (the bf16 checkpoint's rows, upcast exactly)."""
        torch = self.torch
        self.head = self.head.detach().clone()
        self.model.to(device=device, dtype=torch.bfloat16)
        self.device = torch.device(device)

    # -- the vLLM endpoints the shim calls --------------------------------------------------------------------------
    def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        if method == "GET" and path == "/v1/models":
            return 200, {"object": "list", "data": [{"id": self.served, "max_model_len": self.max_model_len}]}
        raise RuntimeError(f"the stand-in does not serve {method} {path}")

    def tokenize(self, text: str) -> list[int]:
        """vLLM's /tokenize and its completion prompt: add_special_tokens false; special token text is still read as
        those tokens (split_special_tokens false), as Hugging Face tokenizers do."""
        return list(self.tok.encode(text, add_special_tokens=False))

    def detokenize(self, ids: list[int]) -> str:
        return self.tok.decode(list(ids))

    def chat_text(self, messages: list[dict[str, Any]]) -> str:
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, continue_final_message=True, enable_thinking=False
        )

    def tokenize_chat(self, messages: list[dict[str, Any]]) -> list[int]:
        return self.tokenize(self.chat_text(messages))

    def label_logprobs(self, prompt: str, ids: list[int], attention_mode: Any = None, cache_write: Any = None):
        tokens = self.tokenize(prompt)
        self.fits(tokens)
        lp = self.readout(tokens, attention_mode, None)
        return {i: float(lp[i]) for i in ids}, {"prompt_tokens": len(tokens), "prompt_token_ids": tokens}

    def label_logprobs_chat(self, messages, ids, max_pixels=0, attention_mode=None, cache_write=None):
        urls = [
            part["image_url"]["url"]
            for message in messages
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]
        vision = self.images(urls, max_pixels)
        tokens = self.expand(self.tokenize(self.chat_text(messages)), vision["rows"])
        self.fits(tokens)
        lp = self.readout(tokens, attention_mode, vision)
        usage = {
            "prompt_tokens": len(tokens),
            "prompt_token_ids": tokens,
            "image_grids": vision["grids"],
            "pixel_values_sha256": vision["pixels"],
        }
        return {i: float(lp[i]) for i in ids}, usage

    # -- the model --------------------------------------------------------------------------------------------------
    def fits(self, tokens: list[int]) -> None:
        """vLLM refuses a prompt over --max-model-len (the two-way build: over --max-num-batched-tokens, which serve.sh
        sets to the same value); the shim reads that refusal as a 422."""
        if len(tokens) > self.max_model_len:
            raise self.maker.Unprocessable(f"over the server's context limit: {len(tokens)} > {self.max_model_len}")

    def processor(self, max_pixels: int) -> Any:
        from transformers import AutoImageProcessor

        mp = int(max_pixels) or DEFAULT_MAX_PIXELS
        if mp not in self.processors:  # given at construction: Qwen2VLImageProcessor.__init__ honours max_pixels alone
            self.processors[mp] = AutoImageProcessor.from_pretrained(
                str(self.root), min_pixels=self.min_pixels, max_pixels=mp
            )
        return self.processors[mp]

    def images(self, urls: list[str], max_pixels: int) -> dict[str, Any]:
        pictures, digests = [], []
        for url in urls:
            raw = base64.b64decode(url.partition(",")[2])
            digests.append(hashlib.sha256(raw).hexdigest())
            pictures.append(vllm_image(raw))
        out = self.processor(max_pixels)(images=pictures, return_tensors="pt")
        grids = [[int(x) for x in g] for g in out["image_grid_thw"].tolist()]
        rows = [t * h * w // (self.merge * self.merge) for t, h, w in grids]
        return {
            "pixel_values": out["pixel_values"],
            "grid": out["image_grid_thw"],
            "grids": grids,
            "rows": rows,
            "sha256": digests,
            "pixels": hashlib.sha256(out["pixel_values"].float().numpy().astype("<f4").tobytes()).hexdigest(),
        }

    def expand(self, tokens: list[int], rows: list[int]) -> list[int]:
        """Each image token expanded to its image's rows, in order (what the processor does to the prompt)."""
        at = [i for i, t in enumerate(tokens) if t == self.image_token]
        if len(at) != len(rows):
            raise RuntimeError(f"{len(at)} image tokens in the prompt for {len(rows)} images")
        out, last = [], 0
        for i, n in zip(at, rows, strict=True):
            out += tokens[last:i] + [self.image_token] * n
            last = i + 1
        return out + tokens[last:]

    def readout(self, tokens: list[int], attention_mode: Any, vision: dict[str, Any] | None) -> Any:
        """log-softmax over the whole vocabulary of the logits at the last prompt position, fp32 (vLLM's logprobs)."""
        torch = self.torch
        if self.model is None:  # --tokens-only: every label alike
            return torch.zeros(self.vocab)
        with self.lock, torch.inference_mode():
            ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            kwargs: dict[str, Any] = {"input_ids": ids, "use_cache": False}
            if vision is not None:
                kwargs["pixel_values"] = vision["pixel_values"].to(self.device, self.model.dtype)
                kwargs["image_grid_thw"] = vision["grid"].to(self.device)
                kwargs["mm_token_type_ids"] = (ids == self.image_token).to(torch.int32)  # 1 = image, 0 = text
            hook = None
            if attention_mode == "bidirectional":
                hook = self.model.model.language_model.register_forward_pre_hook(two_way_mask, with_kwargs=True)
            try:
                hidden = self.model.model(**kwargs).last_hidden_state[0, -1]  # after the final norm
            finally:
                if hook is not None:
                    hook.remove()
            logits = self.head @ hidden.float().cpu()
            return torch.log_softmax(logits, dim=-1)


def vllm_image(raw: bytes) -> Any:
    """An image's bytes as the h2oai/vllm build loads them (ImageMediaIO.load_bytes with image_mode RGB and the white
    default background; normalize_image, rgba_to_rgb and convert_image_mode of vllm/multimodal/image.py)."""
    import contextlib

    from PIL import Image, ImageOps

    image = Image.open(io.BytesIO(raw))
    with contextlib.suppress(Exception):
        image = ImageOps.exif_transpose(image)
    image.load()
    if image.mode == "RGB":
        return image
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in getattr(image, "info", {}):
        return image.convert("RGB")
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    flat = Image.new("RGB", image.size, (255, 255, 255))
    flat.paste(image, mask=image.split()[3])
    return flat


def capture_responds(maker: Any) -> list[tuple[Any, Any, Any]]:
    """Shim.respond's inputs (each question's work item and read, and the temperatures), recorded per call."""
    seen: list[tuple[Any, Any, Any]] = []
    original = maker.Shim.respond

    def respond(self, work, reads, shared, asked, t0, temperature, *args, **kwargs):
        seen.append((work, reads, temperature))
        return original(self, work, reads, shared, asked, t0, temperature, *args, **kwargs)

    maker.Shim.respond = respond
    return seen


def as_maps(request: dict[str, Any]) -> dict[str, Any]:
    """A choice given as a list of labels asked as the map of those labels without descriptions, as ippon reads it:
    the shim takes a map only."""
    questions = {
        k: dict(q, criteria=dict.fromkeys(str(c) for c in q["criteria"]))
        if isinstance(q, dict) and q.get("type", "choice") == "choice" and isinstance(q.get("criteria"), list)
        else q
        for k, q in (request.get("questions") or {}).items()
    }
    return dict(request, questions=questions)


def with_media(case: dict[str, Any], media: Path | None, maker: Any) -> dict[str, Any]:
    """The request with the media files it names (ippon's media request files: `images` by file name, `media` with
    each one's sha256) as data URIs; a choice's list of labels as a map (`as_maps`)."""
    request = as_maps(copy.deepcopy(case["request"]))
    named = {m["name"]: m["sha256"] for m in case.get("media", [])}
    if not named:
        return request
    if media is None:
        raise SystemExit(f"{case['name']}: --media is needed for its images")
    out = []
    for name in request.get("images", []):
        if name not in named:
            out.append(name)
            continue
        raw = (media / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != named[name]:
            raise SystemExit(f"{media / name}: not the file the request names (sha256)")
        mime = maker.sniff_image(raw) or "image/png"
        out.append(f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}")
    request["images"] = out
    return request


def ask(maker: Any, shim: Any, seen: list[Any], request: dict[str, Any], mode: str) -> dict[str, Any]:
    body = copy.deepcopy(request)
    body["attention_mode"] = mode
    seen.clear()
    try:
        response = shim.decide(body)
    except maker.BadRequest as e:
        return {"error": {"status": 400, "detail": str(e)}}
    except maker.Unprocessable as e:
        return {"error": {"status": 422, "detail": str(e)}}
    work, reads, temperature = seen[-1]
    by_type = temperature[0] if isinstance(temperature, tuple) else temperature
    questions = []
    for (qid, typ, names, _descs, _text, labels, _threshold), (lp, usage) in zip(work, reads, strict=True):
        t = by_type[typ] if isinstance(by_type, dict) else by_type
        p = [float(x) for x in maker.probabilities(lp, t)]  # the model's, before the yes/no floor
        questions.append(
            {
                "id": qid,
                "type": typ,
                "options": list(names),
                "labels": list(labels),
                "prompt": list(usage["prompt_token_ids"]),
                "image_grids": usage.get("image_grids"),
                "pixel_values_sha256": usage.get("pixel_values_sha256"),
                "temperature": float(t),
                "probabilities": p,
            }
        )
    response["usage"].pop("latency_ms", None)
    return {"questions": questions, "response": response}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, type=Path, help="the h2oai/h2o-lightning-4b checkpoint folder")
    p.add_argument("--requests", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--media", type=Path, help="the folder of the media files the requests name")
    p.add_argument(
        "--mode",
        choices=("bidirectional", "causal"),
        default="bidirectional",
        help="the attention mode every request is read in (serve.sh's default: bidirectional)",
    )
    p.add_argument("--max-model-len", type=int, default=40960, help="vLLM's --max-model-len (serve.sh's MAX_LEN)")
    p.add_argument("--tokens-only", action="store_true")
    p.add_argument("--bf16", action="store_true", help="read again in bf16, for the bound")
    p.add_argument("--bf16-device", default="mps", help="where the bf16 reading runs (the CPU's bf16 is far slower)")
    p.add_argument("--bound", type=float, default=MIN_BOUND, help="the bound without --bf16")
    p.add_argument("--threads", type=int)
    p.add_argument("--only", nargs="*", help="request names to run")
    p.add_argument(
        "--max-images",
        type=int,
        help="the shim's own SHIM_MAX_IMAGES override (its shipped 4 is the deployment's cap, not the model's), "
        "recorded on each request it answers",
    )
    a = p.parse_args()

    root = a.model.expanduser().resolve()
    clean_environment()
    maker = load_shim(root)
    seen = capture_responds(maker)
    import torch
    import transformers

    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = json.loads((root / "serve_config.json").read_text())
    engine = Engine(maker, root, cfg["model"], a.max_model_len, not a.tokens_only)
    overrides = {} if a.max_images is None else {"SHIM_MAX_IMAGES": str(a.max_images)}
    contract = maker.Contract(dict(cfg, _config_dir=str(root)), env=overrides)  # the shipped configuration
    shim = maker.Shim(contract, engine, 4096, None, None)
    mode = "causal" if a.tokens_only else a.mode

    cases = json.loads(a.requests.read_text())
    if a.only:
        if missing := set(a.only) - {c["name"] for c in cases}:
            p.error(f"unknown request names: {sorted(missing)}")
        cases = [c for c in cases if c["name"] in a.only]

    out: list[dict[str, Any]] = []
    for case in cases:
        request = with_media(case, a.media, maker)
        if any(isinstance(q, dict) and q.get("type") == "box" for q in (request.get("questions") or {}).values()):
            out.append({"name": case["name"], "request": case["request"], "left_out": "a box question"})
            continue
        t0 = time.time()
        found = ask(maker, shim, seen, request, mode)
        entry: dict[str, Any] = {"name": case["name"], "request": case["request"], "mode": mode}
        if case.get("media"):
            entry["media"] = case["media"]
        if overrides:
            entry["shim_env"] = overrides
        if "error" in found:
            entry["error"] = found["error"]
            print(f"{case['name']}: refused {found['error']['status']} ({found['error']['detail'][:120]})", flush=True)
        else:
            qs, response = found["questions"], found["response"]
            entry.update(
                {
                    "keys": [q["id"] for q in qs],
                    "prompts": [q["prompt"] for q in qs],
                    "labels": {q["id"]: q["labels"] for q in qs},
                    "image_grids": {q["id"]: q["image_grids"] for q in qs if q["image_grids"]},
                    "pixel_values_sha256": {q["id"]: q["pixel_values_sha256"] for q in qs if q["pixel_values_sha256"]},
                    "temperatures": {q["id"]: q["temperature"] for q in qs},
                    "input_tokens": response["usage"]["input_tokens"],
                    "usage": response["usage"],
                    "attention_mode": response["attention_mode"],
                    "state_selection": response["effort_used"].get("state_selection"),
                }
            )
            if not a.tokens_only:
                entry["probabilities"] = {q["id"]: q["probabilities"] for q in qs}
                entry["answers"] = response["answers"]
            print(f"{case['name']}: {entry['input_tokens']} tokens, {time.time() - t0:.1f} s", flush=True)
        out.append(entry)

    def write(bound: float, worst: float | None) -> None:
        reference = {
            "schema": "ippon-reference/1",
            "adapter": "h2oai/h2o-lightning",
            "model": {"repo": a.repo, "revision": a.revision},
            "maker": {
                "code": "h2o_lightning_shim.py (Contract, Shim.decide) with a transformers stand-in for vLLM",
                "shim_sha256": sha256_file(root / "h2o_lightning_shim.py"),
                "serve_config_sha256": sha256_file(root / "serve_config.json"),
                "tokenizer": "AutoTokenizer(<model>/tokenizer): Qwen2Tokenizer",
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "device": "cpu",
                "dtype": "float32",
                "attention_mode": mode,
                "max_model_len": a.max_model_len,
                "image_decode": "vLLM b05cc29 ImageMediaIO.load_bytes: EXIF turn, transparency composited on white",
                "image_pixels": {"min": engine.min_pixels, "max": shim.c.max_pixels or DEFAULT_MAX_PIXELS},
                "two_way_probe_distance": shim.two_way_anchor,
                "two_way_separation": shim.two_way_separation,
                "bf16_distance": worst,
                "bf16_device": a.bf16_device if a.bf16 else None,
            },
            "bound": bound,
            "tokens_only": a.tokens_only,
            "requests": out,
        }
        if a.media is not None:
            reference["media_format"] = {
                "directory": "references/media/images",
                "decode": reference["maker"]["image_decode"],
                "image_token": engine.image_token,
                "prompts": "the engine's input ids, each image's <|image_pad|> expanded to its rows (ippon's "
                "placeholders, arch/common.media_id, stand where these ids are image_token)",
                "images": "per question, its images' image_grid_thw and the sha256 of their pixel_values rows "
                "(float32, little-endian)",
            }
        data = dumps(reference, ensure_ascii=False, allow_nan=False).encode()
        a.out.write_bytes(gzip.compress(data) if a.out.suffix == ".gz" else data)
        print(f"{len(out)} requests -> {a.out} (mode {mode}, bound {bound})")

    write(a.bound, None)  # fp32 first: a failed bf16 run keeps it
    bound, worst = a.bound, None
    if a.bf16 and not a.tokens_only:
        engine.bf16(a.bf16_device)
        for entry in out:
            if "probabilities" not in entry:
                continue
            found = ask(
                maker,
                shim,
                seen,
                with_media(next(c for c in cases if c["name"] == entry["name"]), a.media, maker),
                mode,
            )
            entry["probabilities_bf16"] = {q["id"]: q["probabilities"] for q in found["questions"]}
            print(f"bf16 {entry['name']}", flush=True)
        worst = max(
            (
                abs(x - y)
                for e in out
                if "probabilities_bf16" in e
                for q in e["probabilities"]
                for x, y in zip(e["probabilities"][q], e["probabilities_bf16"][q], strict=True)
            ),
            default=0.0,
        )
        bound = round(max(MIN_BOUND, worst + 5e-4), 4)
    if not math.isfinite(bound):
        raise SystemExit("non-finite probabilities")

    write(bound, worst)


if __name__ == "__main__":
    main()
