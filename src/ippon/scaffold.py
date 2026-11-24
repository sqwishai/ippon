"""`ippon new-adapter`: a package that plugs a decision model into ippon, ready to fill in.

    ippon new-adapter acme/decider --arch qwen35 --out acme-ippon

writes a pyproject with the `ippon.adapters` entry point, an adapter and format skeleton on the chosen
architecture (the parts to write marked TODO), a reference script that runs the creator's own code, and a test that
compares prompt tokens with its output. docs/adding-a-model.md describes each part.
"""

from __future__ import annotations

import keyword
import re
from pathlib import Path

from .arch import DECODERS, ENCODERS
from .errors import InvalidRequest
from .verify import BOUND, REFERENCE_SCHEMA


def _archs() -> dict[str, str]:
    """Every architecture module an adapter can load (arch.DECODERS and ENCODERS), by name, with its title (its
    docstring's first words)."""
    modules = {m.__name__.rsplit(".", 1)[-1]: m for m in (*DECODERS.values(), *ENCODERS.values())}
    return {
        name: re.split(r":|,|\s\(|\.\s", (m.__doc__ or name).strip())[0].strip() for name, m in sorted(modules.items())
    }


ARCHS = _archs()

PYPROJECT = """[project]
name = "{dist}"
version = "0.1.0"
description = "ippon adapter for {name}"
requires-python = ">=3.11"
dependencies = ["ippon"]

[project.entry-points."ippon.adapters"]
"{name}" = "{module}:Adapter"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{module}"]
"""

ADAPTER = '''"""ippon adapter for {name} ({arch_title}): the creator's prompt and readout code, reimplemented.

Write down here what the creator's code does, step by step, as the built-in adapters do: the prompt strings, the
order criteria are shown in, how a missing state or description is handled, the limits, the readout and the
temperature. Everything that differs from the creator's code (a request it refuses that this accepts) is listed too.
"""

from __future__ import annotations

import json
from typing import Any

from ippon import API_VERSION, LoadContext, Loaded, ModelDir, Rendering
from ippon import quant
from ippon.adapter import AdapterBase, FormatBase, Hidden
from ippon.arch import {arch}
from ippon.errors import InvalidRequest
from ippon.plan import Prompt, Readout, state_end_marks
from ippon.readout import LabelSet, label_logits, option_probabilities
from ippon.schema import Request
from ippon.tokenizer import Tokenizer


class Format(FormatBase):
    def __init__(self, tokenizer: Tokenizer, output: quant.Table) -> None:
        self.tokenizer, self.output = tokenizer, output

    def render(self, request: Request) -> Rendering:
        """The creator's prompts for a request: one per question here, each starting with the state, so that the state
        is read once for all of them and kept for the next request about it."""
        state = request.state if isinstance(request.state, str) else json.dumps(request.state, ensure_ascii=False)
        head = f"State: {{state}}\\n"  # TODO: the creator's exact strings
        end = len("State: ") + len(state)  # where the state's own text ends in every prompt
        texts = []
        for q in request.questions:
            if q.instructions is None:
                raise InvalidRequest(f"{{q.id}}: this model needs instructions")
            texts.append(head + f"Question: {{q.instructions}}\\nAnswer:")
        encoded = self.tokenizer.encode_offsets(texts)
        prompts = [
            Prompt(tuple(ids), (Readout(q.id, len(ids) - 1),), state_end_marks(text, offsets, end))
            for q, text, (ids, offsets) in zip(request.questions, texts, encoded, strict=True)
        ]
        return Rendering(request, tuple(prompts))

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, Any]:
        """Each question's probabilities, in `question.options` order (a noul: true, then false)."""
        out = {{}}
        for q in rendering.request.questions:
            labels = LabelSet(tuple((self.tokenizer.encode(f" {{o}}")[0],) for o in q.options))  # TODO: the creator's
            logits = label_logits(hidden.rows(q.id)[-1:], self.output, labels.flat)
            out[q.id] = option_probabilities(logits, labels)[0]
        return out


class Adapter(AdapterBase):
    api = API_VERSION
    name = "{name}"
    files = ("config.json", "tokenizer.json", "*.safetensors", "model.safetensors.index.json")

    def matches(self, model: ModelDir) -> bool:
        return False  # TODO: recognise this model's files, e.g. model.json("{module}_config.json") is not None

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        backbone = ctx.backbone({arch}, model)  # in the load's dtype, quantized to its bits
        return Loaded(backbone, Format(ctx.tokenizer(model), backbone.output_matrix))
'''

REFERENCE = '''"""{name} reference answers from the creator's own code, run in the creator's environment:

    python -I make_reference.py --model path/to/checkpoint --requests standard.json --out reference.json.gz

The file lists, for each request the creator's code accepts, the token ids of every prompt it reads, the creator's
input-token count and (unless --tokens-only) each question's probabilities in the request's option order, from the
creator's code in fp32. `ippon verify` compares a model with it. Get standard.json from ippon's
references/requests/.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--requests", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tokens-only", action="store_true")
    args = p.parse_args()
    out = []
    for case in json.loads(args.requests.read_text()):
        try:
            prompts: list[list[int]] = []  # TODO: the creator's token ids for case["request"]
            input_tokens = sum(len(x) for x in prompts)
        except ValueError:  # a request the creator's code refuses is not a reference case
            continue
        entry: dict[str, Any] = {{"name": case["name"], "request": case["request"], "prompts": prompts,
                                  "input_tokens": input_tokens}}
        if not args.tokens_only:
            entry["probabilities"] = {{}}  # TODO: {{question id: [p per option]}} from the creator's code in fp32
        out.append(entry)
    reference = {{"schema": "{schema}", "adapter": "{name}", "bound": {bound},
                 "tokens_only": args.tokens_only, "requests": out}}
    data = json.dumps(reference, ensure_ascii=False).encode()
    args.out.write_bytes(gzip.compress(data) if args.out.suffix == ".gz" else data)


if __name__ == "__main__":
    main()
'''

TEST = '''"""{name}'s prompts against the creator's own token ids (references/reference.tokens.json.gz)."""

import gzip
import json
from pathlib import Path

import pytest

from ippon.schema import parse_request

REFERENCE = Path(__file__).resolve().parents[1] / "references" / "reference.tokens.json.gz"


@pytest.mark.skipif(not REFERENCE.exists(), reason="make the reference first (references/make_reference.py)")
def test_prompts_match_the_creators_code() -> None:
    from {module} import Format

    fmt = Format(...)  # TODO: the tokenizer (from the checkpoint's tokenizer.json) and what render() needs
    for case in json.loads(gzip.decompress(REFERENCE.read_bytes()))["requests"]:
        rendering = fmt.render(parse_request(case["request"]))
        assert [list(p.tokens) for p in rendering.prompts] == case["prompts"], case["name"]
        assert fmt.input_tokens(rendering) == case["input_tokens"], case["name"]
'''

README = """# {dist}

An ippon adapter for {name} ({arch_title}).

    pip install -e .
    ippon doctor                 # lists "{name}" among the adapters
    ippon verify path/to/checkpoint --reference references/reference.json.gz

Checklist (docs/adding-a-model.md in ippon): the format reimplements the creator's code; the reference comes
from the creator's code in fp32; `verify` passes at every precision offered; `ippon bench` on one Mac.
"""


def module_name(name: str) -> str:
    """A Python package name for an adapter name: "acme/decider" -> "acme_decider_ippon"."""
    base = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    if not base or base[0].isdigit() or keyword.iskeyword(base):
        base = "adapter_" + base
    return base + "_ippon"


def write(name: str, arch: str, out: Path) -> list[Path]:
    """Write the package; refuses to overwrite an existing file. The files written."""
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]*(/[\w.-]+)?", name):
        raise InvalidRequest("an adapter name looks like creator/model")
    if arch not in ARCHS:
        raise InvalidRequest(f"arch is one of {', '.join(ARCHS)}")
    module = module_name(name)
    values = {"name": name, "arch": arch, "arch_title": ARCHS[arch], "module": module, "dist": module.replace("_", "-")}
    values |= {"schema": REFERENCE_SCHEMA, "bound": str(BOUND)}
    files = {
        out / "pyproject.toml": PYPROJECT,
        out / "README.md": README,
        out / "src" / module / "__init__.py": ADAPTER,
        out / "references" / "make_reference.py": REFERENCE,
        out / "tests" / f"test_{module}.py": TEST,
    }
    existing = [p for p in files if p.exists()]
    if existing:
        raise InvalidRequest(f"not overwriting {', '.join(map(str, existing))}")
    for path, template in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.format(**values))
    return list(files)
