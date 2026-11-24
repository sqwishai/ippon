"""Print an Ollama manifest's weights fingerprint: the sha256 of its tensor and JSON layers' digests, sorted, one per
line. Its params, licence and the recorded manifest layer are left out, so a tag republished with new params keeps
its fingerprint and a change to any weight, tokenizer or config file changes it."""

import hashlib
import json
import sys
from pathlib import Path

WEIGHTS = ("tensor", "json")  # application/vnd.ollama.image.<kind>; "v2+json" is the manifest recorded, not a file

layers = json.loads(Path(sys.argv[1]).read_text())["layers"]
digests = sorted(layer["digest"] for layer in layers if layer["mediaType"].rsplit(".", 1)[-1] in WEIGHTS)
print(hashlib.sha256("\n".join(digests).encode()).hexdigest())
