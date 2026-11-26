"""ippon's one catalogue: every model and build it loads by name, what has been checked of each and how it came out,
a GGUF build's pinned file, and the facts setup and `ippon recommend` read of a model. docs/models.md's table
(`scripts/models_table.py`), `ippon check`, the short model names (names.py: a build's `name`, FAMILY:TAG, as `ippon
pull d1-2b:q8_0` takes it), the GGUF builds (gguf/builds.py), `ippon suite`'s models and the licences ippon names
(licences.py) all read it.

A build's `status`:

- "verified": its answers are within the bound of a reference (docs/accuracy.md): the creator's own code in fp32 on
  the creator's requests, in the creator's bf16 where its fp32 fits no 48 GB Mac (the note says so), or, for a
  quantized build, the code it was published for (`verify --against`);
- "prompts": its prompts are token for token the creator's on the reference requests, its answers not checked
  against an fp32 reference (no reference fits a 48 GB Mac, or none is published);
- "past bound": its answers are checked and move past the bound: `distance` says by how much, the note why;
- "experimental": compatibility implemented, with outstanding full-model quality or serving gates in its note.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# The licences the catalogue's models come under, as each names itself
APACHE, LFM, MIT, NONCOMMERCIAL = "Apache-2.0", "LFM Open License v1.0", "MIT", "CC BY-NC 4.0"
KETTEI = "Kettei Community Model License 1.0"
FASTINO = "Fastino Community License"
# Hugging Face's licence identifiers (a model card's `license`), as their licences name themselves
LICENCES = {
    "apache-2.0": APACHE,
    "mit": MIT,
    "cc-by-4.0": "CC BY 4.0",
    "cc-by-nc-4.0": NONCOMMERCIAL,
    "cc-by-sa-4.0": "CC BY-SA 4.0",
    "cc-by-nc-sa-4.0": "CC BY-NC-SA 4.0",
    "llama3.1": "Llama 3.1 Community License",
    "gemma": "Gemma Terms of Use",
}
# Open-source and ungated community licences whose weights download without a clickwrap prompt (matching mlx-lm,
# transformers, and ollama; licence name and URL are still shown on download, in Result.diagnostics, and /v1/models).
# Non-commercial (CC BY-NC 4.0) and other custom licences require explicit acceptance before download, Kettei's
# included: Sqwish asks every user to accept it once, here as in the Sqwish Labs SDK.
OPEN = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        LFM.casefold(),
        FASTINO.casefold(),
    }
)

# What the agent commands and `ippon start` load when neither setup nor the caller names a model:
# Kettei Core at 8 bits, which fits every Mac ippon runs on
DEFAULT_MODEL = "kettei-core:q8"

Status = Literal["verified", "prompts", "past bound", "experimental"]


def status_line(status: str, note: str) -> str:
    """A build's status with its note, in a line: "past bound (...)"; the status alone without one."""
    return f"{status}{' (' + note + ')' if note else ''}"


@dataclass(frozen=True)
class Gguf:
    """A GGUF build's pinned file: its repository's revision, the file, its sha256 and header digest, and its picture
    and audio parts, a file of their own beside it (llama.cpp's --mmproj), if any. The creator's small files (config,
    tokenizer, templates) it is read with come from its model's first build, at that build's pin unless
    `base_revision` names the one its answers were checked with."""

    revision: str
    file: str
    sha256: str
    header: str  # sha256 of the header's bytes (`gguf.header.Header.digest`)
    mmproj: str = ""
    mmproj_sha256: str = ""
    mmproj_header: str = ""
    base_revision: str = ""


@dataclass(frozen=True)
class Size:
    """What a build's weights take as published, in GB (10^9 bytes), and what they are where it says (bf16): the
    number setup, `pull` and `ippon recommend` judge a fit by, and the label the docs' table and messages show."""

    gigabytes: float
    form: str = ""

    def __str__(self) -> str:
        return f"{self.gigabytes:.1f} GB" + (f" ({self.form})" if self.form else "")


@dataclass(frozen=True)
class Build:
    # what loads it: a Hub id pinned to the revision its answers were verified at (org/name@commit, so a pull never
    # takes a later main), a GGUF build (repo:QUANT, pinned by `gguf`), or an Ollama build (ollama://name:tag)
    load: str
    weights: Size  # what its weights take, as published
    status: Status
    note: str = ""  # how it came out, for the table
    name: str = ""  # its short name, FAMILY:TAG (names.py: `ippon pull d1-2b:q8_0`); "" for one without
    # what a request may carry besides text, for this build (a GGUF build's media parts are in its mmproj file), and
    # how the answers about it came out, as `status` (`ippon recommend images`)
    media: tuple[tuple[str, Status], ...] = ()
    # the file under references/ its answers were checked against (the creator's, for text), shipped in the wheel so
    # `ippon verify NAME` checks it on any Mac without a clone; "" for a build with none
    reference: str = ""
    # a build past the bound: the largest distance of its answers from the creator's fp32 ones, as recorded (`ippon
    # suite` takes a verify within 10% of it as matching the record)
    distance: float | None = None
    gguf: Gguf | None = None  # a GGUF build's pinned file

    def __post_init__(self) -> None:
        if (self.status == "past bound") != (self.distance is not None):
            raise ValueError(f"{self.load}: a build past the bound, and only one, records its distance")
        if (self.gguf is not None) != (":" in self.load and not self.load.startswith("ollama://")):
            raise ValueError(f"{self.load}: a GGUF build, and only one, pins its file")

    @property
    def answers(self) -> str:
        """How its answers stand, in a line (`status_line`): the docs' table of builds."""
        return status_line(self.status, self.note)

    @property
    def gigabytes(self) -> float:
        """What its weights take, in GB (10^9 bytes)."""
        return self.weights.gigabytes


@dataclass(frozen=True)
class Model:
    """A model, its licence and its builds, the first its creator's checkpoint; and what setup and `ippon recommend`
    read of it: where setup lists it (`place`, from 0: Sqwish's own first, then the others from the better choice to
    the lighter one, roughly, a suggestion of where to start and never shown as a score), its rank among the models a
    new user starts with (`starter`, from 0; None for the others), and what it holds in memory at full precision
    where that is well under its download (`resident_gb`)."""

    name: str
    licence: str
    place: int
    builds: tuple[Build, ...]
    starter: int | None = None
    # Kettei reads its vocabulary tables from the file on the CPU (`ippon pull kettei-core --dry-run`, M5 Pro). Kettei
    # Core at 8 bits held 3.7 GiB on an 8 GB M1 Air, with answers 0.030 from the reference against a bound of 0.037
    # (benchmarks/results/m1-2026-11-05-kettei). Every other model holds about its download.
    resident_gb: float | None = None


MODELS = (
    Model(
        "Clef-Flash",
        APACHE,
        place=12,
        builds=(
            Build(
                "Cloudflare/clef-flash@8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
                Size(19.1, "bf16"),
                "prompts",
                "and its head on a random instance",
                name="clef-flash:bf16",
                reference="clef/clef-flash.tokens.json.gz",
                media=(("images", "prompts"),),
            ),
            Build(
                "mlx-community/clef-flash-8bit@5a195dc147926b63b1d139691ed4480ca9f92c0e",
                Size(10.7),
                "verified",
                "within the bound of the build's own code",
                name="clef-flash:8bit",
                reference="clef/clef-flash.tokens.json.gz",
                media=(("images", "prompts"),),
            ),
            Build(
                "ggml-org/Clef-Flash-GGUF:Q8_0",
                Size(9.7),
                "verified",
                "every tensor, and 0.031 from the full checkpoint",
                name="clef-flash:q8_0",
                reference="clef/clef-flash.tokens.json.gz",
                gguf=Gguf(
                    revision="8aa35d1cc7e141f3771e564eeb630a501597c681",
                    file="Clef-Flash-Q8_0.gguf",
                    sha256="87a232aa1f7495cfc90ce22f08bb3779655e1b05c991c84b21afa4c1c8c4e0f5",
                    header="93e7f47fa7de5ebe61c42fb0d7edaf8b9a64a3e671b096465a97030b4befd45c",
                ),
            ),
            Build(
                "ollama://clef-flash:9b-mxfp8",
                Size(12.4),
                "experimental",
                (
                    "0.00113 from the same weights in fp32. Ollama 0.40.13's own answers 0.378 from them, past 0.01. "
                    "Original fp32 quality unmeasured"
                ),
                name="clef-flash:mxfp8",
            ),
        ),
    ),
    Model(
        "d1-2B",
        LFM,
        place=15,
        builds=(
            Build(
                "LiquidAI/d1-2B@10e7671d44e4b37700020b1eed3f594c5f2af0b4",
                Size(3.8, "bf16"),
                "verified",
                "0.0012, bound 0.016. Images 0.0016, bound 0.01. Videos 0.0015, bound 0.01",
                name="d1-2b:bf16",
                reference="d1/d1-2B.json.gz",
                media=(("images", "verified"), ("videos", "verified")),
            ),
            Build(
                "LiquidAI/d1-2B-GGUF:Q8_0",
                Size(1.8),
                "verified",
                "0.010, bound 0.016. 0.0023 from llama.cpp b11960's on the same file",
                name="d1-2b:q8_0",
                reference="d1/d1-2B.json.gz",
                gguf=Gguf(
                    revision="e968f6575f0d4fad928312d7af15cad90fd365c0",
                    file="d1-2B-Q8_0.gguf",
                    sha256="073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c",
                    header="67b53db82607e9c7d532ff24b1e491bf99016024263ee4aa32060d7e301cf85a",
                    mmproj="mmproj-d1-2B-Q8_0.gguf",
                    mmproj_sha256="62cc6d74fd1e5b1d6b48518566bf4c0cfd2468079a0c096ea323b74eed01e0d1",
                    mmproj_header="8d36aa52511ff3a973f2b7cae988337f7695d3cc7f430ee474fbb86150559797",
                    # the revision its answers were checked with (references/d1/README.md); the model's pin, 10e7671,
                    # changed only Liquid's prompt code and lfm3.py, not the files a GGUF build reads
                    base_revision="cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41",
                ),
                # its mmproj file reads them, through the prompts the checkpoint's are verified with; its answers
                # about images are 0.0138 from Liquid's and about its 12 videos 0.019, past the bound of 0.01
                # (docs/accuracy.md)
                media=(("images", "past bound"), ("videos", "past bound")),
            ),
            Build(
                "ollama://d1:2b",
                Size(3.9),
                "verified",
                "Ollama's default d1 build, BF16, at 0.0012, bound 0.016. Ollama 0.40.13's own answers 0.0125",
            ),
            Build(
                "ollama://d1:2b-mxfp8",
                Size(2.4),
                "past bound",
                (
                    "0.10, bound 0.016, one top answer changed. Ollama 0.40.13's own answers 0.081 on the requests "
                    "it takes"
                ),
                distance=0.1,
            ),
        ),
    ),
    Model(
        "d1-Omni-2B",
        LFM,
        place=16,
        builds=(
            Build(
                "LiquidAI/d1-Omni-2B@c8d48c08dada0191c6b362820f055cfb49c99426",
                Size(4.3, "bf16"),
                "verified",
                (
                    "0.0018, bound 0.017. Images 0.0048, bound 0.019. Clips 0.0010, bound 0.014. Videos 0.0021, "
                    "bound 0.0145. Explicit video bf16 0.0156, past bound"
                ),
                name="d1-omni-2b:bf16",
                reference="d1/d1-omni-2B.json.gz",
                media=(("images", "verified"), ("audio", "verified"), ("videos", "verified")),
            ),
            Build(
                "LiquidAI/d1-Omni-2B-GGUF:Q8_0",
                Size(1.8),
                "past bound",
                (
                    "the build's own Q8_0 at 0.045 from the full checkpoint's fp32, bound 0.017. 0.0033 from "
                    "llama.cpp b11960's on the same file"
                ),
                name="d1-omni-2b:q8_0",
                reference="d1/d1-omni-2B.json.gz",
                distance=0.045,
                # its mmproj file reads them: 0.0085 on the 17 image requests and 0.0075 on the 18 audio ones, within
                # their bounds (docs/accuracy.md); its videos run but are not compared with Liquid's
                media=(("images", "verified"), ("audio", "verified"), ("videos", "experimental")),
                gguf=Gguf(
                    revision="c53ae0b1f92b17065e9158d34019c9ca1ea5bcf8",
                    file="d1-Omni-2B-Q8_0.gguf",
                    sha256="aedf1a7c341b4281979423dbc39e9c3d3ad1da8834156c8e75af5399ca3b0861",
                    header="cf7c51c2ea6e6699a459fc21a8202d3d44940dc2024a50df4ac183b41daac415",
                    mmproj="mmproj-d1-Omni-2B-Q8_0.gguf",
                    mmproj_sha256="31778160ac0ec70f9b59f296aa1c8e70cbd1eb7e3a3ff873d695aaa4c7b8d674",
                    mmproj_header="9b57cdca06a69254a060b5362af220523a15bffc62a9051c198cbb7529bbecab",
                ),
            ),
            Build(
                "ollama://d1-omni:2b",
                Size(4.4),
                "verified",
                (
                    "Ollama's default d1-omni build, BF16, at 0.0018, bound 0.017. Ollama 0.40.13's own answers "
                    "0.029, past it"
                ),
            ),
            Build(
                "ollama://d1-omni:2b-mxfp8",
                Size(2.9),
                "past bound",
                "0.15, bound 0.017. Ollama 0.40.13's own answers 0.15 on the requests it takes",
                distance=0.15,
            ),
        ),
    ),
    Model(
        "pplx-decider flash v1.3",
        APACHE,
        place=5,
        builds=(
            Build(
                "perplexity-ai/pplx-decider-flash-v1.3@d06fb3f2d97e0a52325a0d463c450cf0338f44fe",
                Size(9.1, "bf16"),
                "verified",
                "0.0023, bound 0.017. Images 0.0030, bound 0.033",
                name="pplx-flash:bf16",
                reference="perplexity/pplx-decider-flash-v1.3.json.gz",
                media=(("images", "verified"),),
            ),
        ),
    ),
    Model(
        "elephant-1",
        MIT,
        place=4,
        builds=(
            Build(
                "baselabs/elephant-1@b3f8a93462fb1051a9de031e94a7ff380e254072",
                Size(18.8, "bf16"),
                "verified",
                "0.0024, bound 0.025. Images 0.012, bound 0.020 (bf16 0.048, past it)",
                name="elephant:bf16",
                reference="elephant/elephant-1.json.gz",
                media=(("images", "verified"),),
            ),
        ),
    ),
    Model(
        "d3-mini",
        APACHE,
        place=8,
        builds=(
            Build(
                "vllm-sr/d3-mini@97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e",
                Size(9.1),
                "verified",
                "0.0085, bound 0.029. Images 0.0090, bound 0.019. Videos 0.020, bound 0.048",
                name="d3-mini:bf16",
                reference="d3/d3-mini.json.gz",
                media=(("images", "verified"), ("videos", "verified")),
            ),
            Build(
                "ollama://d3:4b-mxfp8",
                Size(5.6),
                "past bound",
                (
                    "0.062 from the creator's fp32, bound 0.029, and 0.0059 from the build's own weights in fp32. "
                    "Ollama 0.40.13's own answers 0.050 on the requests it takes"
                ),
                name="d3-mini:mxfp8",
                reference="batching/d3-mini-mxfp8-dequantized.json.gz",
                distance=0.062,
            ),
            Build(
                "ollama://d3:4b-nvfp4",
                Size(4.0),
                "experimental",
                (
                    "0.00895 from the same weights in fp32, bound 0.01. Up to 0.018 from Ollama's own answers. "
                    "Original fp32 quality unmeasured. Loads only by name, with a drift warning"
                ),
                name="d3-mini:nvfp4",
            ),
        ),
    ),
    Model(
        "d3-flash",
        APACHE,
        place=6,
        builds=(
            Build(
                "vllm-sr/d3-flash@13dc1cc8ef23caefa765907e1d764751e5629f7d",
                Size(16.8),
                "verified",
                "0.0017, bound 0.016. Images 0.012, bound 0.022. Videos 0.0022, bound 0.025",
                name="d3-flash:bf16",
                reference="d3/d3-flash.json.gz",
                media=(("images", "verified"), ("videos", "verified")),
            ),
            Build(
                "ollama://d3:9b-mxfp8",
                Size(10.1),
                "past bound",
                (
                    "0.044, bound 0.016, one top answer changed. Ollama 0.40.13's own answers 0.039 on the requests "
                    "it takes"
                ),
                distance=0.044,
            ),
        ),
    ),
    Model(
        "Torchcast Decision 4B",
        APACHE,
        place=11,
        builds=(
            Build(
                "torchcast-ai/torchcast-decision-4b@492bfb19662f3c6e601cc9c4bd2911ac95609ed4",
                Size(10.4, "bf16"),
                "verified",
                "0.0024, bound 0.043. Images 0.0080, bound 0.020 (bf16 0.031, past it)",
                name="torchcast:bf16",
                reference="torchcast/torchcast-decision-4b.json.gz",
                media=(("images", "verified"),),
            ),
        ),
    ),
    Model(
        "H2O-Lightning-4B",
        APACHE,
        place=9,
        builds=(
            Build(
                "h2oai/h2o-lightning-4b@4ab7b8d60408495cb2a8c14a9454a1ac30d8c12f",
                Size(9.3, "bf16"),
                "verified",
                "0.0040, bound 0.019. Images 0.0097, bound 0.079 (one way 0.0035, bound 0.030)",
                name="h2o:bf16",
                reference="h2o/h2o-lightning-4b.json.gz",
                media=(("images", "verified"),),
            ),
            Build(
                "h2oai/h2o-lightning-4b-gguf:Q8_0",
                Size(4.5),
                "past bound",
                (
                    "the build's own Q8_0, read one way, at 0.024, bound 0.0104. 0.0043 from llama.cpp's on the "
                    "same file"
                ),
                name="h2o:q8_0",
                reference="h2o/h2o-lightning-4b.causal.json.gz",
                distance=0.024,
                # its mmproj file reads them: 0.0071 on the 18 image requests, read one way, bound 0.0302
                # (docs/accuracy.md)
                media=(("images", "verified"),),
                gguf=Gguf(
                    revision="e19edebcafb33e1154e6260d695935ab0984b0e8",
                    file="h2o-lightning-4b-Q8_0.gguf",
                    sha256="4127b2ae7710c3864c21c3d946c316f28a654a53d3a4ba99f383bbcaf2768e81",
                    header="4c00f03ebd6426b14f1e6bf810ded33780c7a2f9e5fcecdb54d366e57eb79f1e",
                    mmproj="mmproj-lightning-f16.gguf",
                    mmproj_sha256="051f748171a801f8983d2872725146a50532955561d657effb7ff08902c25afc",
                    mmproj_header="aa3104766835b826547da86a02732def05ca41cf53db187f1c17a031006a6b37",
                    # the revision its answers were checked with (references/h2o/make_reference.py): the model's pin,
                    # 4ab7b8d, changed serve_config.json, which the H2O adapter reads from these small files
                    base_revision="d7f1b05aa0b4e789b3896116f309ef842440e260",
                ),
            ),
        ),
    ),
    Model(
        "Jeff-1.1-Medium",
        NONCOMMERCIAL,
        place=7,
        builds=(
            Build(
                "jgeuter/Jeff-1.1-Medium@540c40924dd67c715d3367309ba6744bac754b00",
                Size(25.4, "bf16"),
                "verified",
                "0.0474 from the creator's bf16 on the Mac's GPU, bound 0.057: its fp32 fits no 48 GB Mac",
                name="jeff:bf16",
                reference="jeff/jeff-1.1-medium.json.gz",
            ),
        ),
    ),
    Model(
        "Jeff-1.1-Qwen-4B",
        NONCOMMERCIAL,
        place=17,
        builds=(
            Build(
                "jgeuter/Jeff-1.1-Qwen-4B@5401940b5a8c959cba82187ebfa2dce5a21abc0a",
                Size(9.1, "bf16"),
                "verified",
                "0.0033, bound 0.01",
                name="jeff-qwen-4b:bf16",
                reference="jeff/jeff-1.1-qwen-4b.json.gz",
            ),
        ),
    ),
    Model(
        "Kev 0.8B",
        APACHE,
        place=18,
        builds=(
            Build(
                "jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f",
                Size(1.8, "bf16, its base included"),
                "verified",
                "0.0026, bound 0.020",
                name="kev-0.8b:bf16",
                reference="kev/kev-0.8b.json.gz",
            ),
        ),
    ),
    Model(
        "Kev 4B",
        APACHE,
        place=19,
        builds=(
            Build(
                "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52",
                Size(9.5, "bf16, its base included"),
                "verified",
                "0.0023, bound 0.010",
                name="kev-4b:bf16",
                reference="kev/kev-4b.json.gz",
            ),
        ),
    ),
    Model(
        "Kev 9B",
        APACHE,
        place=20,
        builds=(
            Build(
                "jaredpalmer/kev-9b@db029f08b290afd9fee4aa4bbcd9ae48602d1eb0",
                Size(19.5, "bf16, its base included"),
                "verified",
                "0.0027, bound 0.011",
                name="kev-9b:bf16",
                reference="kev/kev-9b.json.gz",
            ),
        ),
    ),
    Model(
        "GLiDE 2B",
        FASTINO,
        place=14,
        builds=(
            Build(
                "fastino/GLiDE-2B@00f122c120cdeed82284106cb77d9bab5a50ad16",
                Size(3.8, "bf16"),
                "verified",
                "0.0022, bound 0.016",
                name="glide-2b:bf16",
                reference="glide/glide-2b.json.gz",
            ),
        ),
    ),
    Model(
        "GLiDE 9B",
        FASTINO,
        place=2,
        builds=(
            Build(
                "fastino/GLiDE-9B@ab54df5acba0eebf7df45560ffdaacd64e32df3b",
                Size(15.9, "bf16"),
                "verified",
                "0.0063, bound 0.011, its recurrent depth included",
                name="glide-9b:bf16",
                reference="glide/glide-9b.json.gz",
            ),
        ),
    ),
    Model(
        "AutoTrust-Decider-Mini-9B",
        APACHE,
        place=3,
        builds=(
            Build(
                "autotrust/AutoTrust-Decider-Mini-9B@7ac669b8fb476ba926491e3d1c39f9330d68dbed",
                Size(18.1, "bf16"),
                "verified",
                "0.0019, bound 0.014",
                name="autotrust-mini-9b:bf16",
                reference="autotrust/decider-mini-9b.json.gz",
            ),
        ),
    ),
    Model(
        "Strands-Decider-E2B",
        APACHE,
        place=21,
        resident_gb=3.8,
        builds=(
            Build(
                "StrandsAgents/strands-decider-E2B-gemma4-v2-2611@eca68159a790d1e7ff93559d166dc3fbe24d5975",
                Size(10.3, "bf16"),
                "verified",
                "0.0031 mean TV (max 0.025, 180/180 top-1) against the creator's bf16 golden vectors; "
                "prompts verified (430 of 430)",
                name="strands-e2b:bf16",
            ),
        ),
    ),
    Model(
        "Strands-Decider-E4B",
        APACHE,
        place=13,
        resident_gb=8.0,
        builds=(
            Build(
                "StrandsAgents/strands-decider-E4B-gemma4-v2-2611@1717bca732ba4ef8c8c531fbc9875fafaf858811",
                Size(16.0, "bf16"),
                "verified",
                "0.0037 mean TV (max 0.024, 90/90 top-1) against the creator's bf16 golden vectors; "
                "prompts verified (430 of 430)",
                name="strands-e4b:bf16",
            ),
        ),
    ),
    Model(
        "Strands-Decider-12B",
        APACHE,
        place=10,
        resident_gb=21.8,
        builds=(
            Build(
                "StrandsAgents/strands-decider-12B-gemma4-v2-2611@41d5dc0d20a4deb2835690eb7529ee92463f6c9f",
                Size(23.9, "bf16"),
                "verified",
                "0.0039 mean TV (max 0.034, 89/90 top-1, up to 32.7K tokens) against the creator's bf16 "
                "golden vectors; prompts verified (430 of 430)",
                name="strands-12b:bf16",
            ),
        ),
    ),
    Model(
        "Kettei Core",
        KETTEI,
        place=0,
        starter=0,
        resident_gb=7.6,
        builds=(
            Build(
                "Sqwish/Kettei-Core@5a9821633c8bb15b92b677d368083ec04720d164",
                Size(10.1, "bf16"),
                "verified",
                "0.0067, bound 0.037 (bf16 0.045, past it)",
                name="kettei-core:bf16",
                reference="kettei/kettei-core.json.gz",
            ),
        ),
    ),
    Model(
        "Kettei Dot",
        KETTEI,
        place=1,
        starter=1,
        resident_gb=1.8,
        builds=(
            Build(
                "Sqwish/Kettei-Dot@a8ef982f8736cbac710899810f87bb53668be6d1",
                Size(2.2, "bf16"),
                "verified",
                "0.0026, bound 0.032",
                name="kettei-dot:bf16",
                reference="kettei/kettei-dot.json.gz",
            ),
        ),
    ),
)


BUILDS = tuple(b for m in MODELS for b in m.builds)  # every build, in the catalogue's order
