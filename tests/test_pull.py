"""Download refusals, shared licence acceptance and interruption recovery without network or real weights."""

from __future__ import annotations

import errno
import hashlib
import json
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import httpx2
import huggingface_hub
import pytest
from huggingface_hub.errors import RemoteEntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.repocard import metadata_load

from ippon import downloads, hub, licences, locate, pull
from ippon.cli import main
from ippon.device import Device
from ippon.errors import (
    DownloadFailed,
    InsufficientDisk,
    InsufficientMemory,
    InvalidRequest,
    LicenseNotAccepted,
    ModelNotFound,
    UnsupportedModel,
)
from ippon.safetensors import Tensor
from ippon.storage import cache_dir

# a 64 KiB safetensors file of one 2 x 2 float32 matrix (its header padded with spaces, as the format allows), the
# matrix the fake Hub's header read reports: a pull that finds it cached reads the same model from disk
_HEADER = json.dumps({"classifier.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}).encode()
WEIGHTS = (65536 - 8 - 16).to_bytes(8, "little") + _HEADER.ljust(65536 - 8 - 16) + bytes(16)


@pytest.fixture
def remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    import huggingface_hub as hf
    from huggingface_hub import constants

    cache = tmp_path / "hub"
    monkeypatch.setenv("IPPON_CONFIG", str(tmp_path / "config"))
    monkeypatch.setenv("IPPON_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    state = SimpleNamespace(
        interrupt=False, transferred=[], offsets=[], requested=[], sha="a" * 40, license="cc-by-nc-4.0"
    )  # a licence to accept: an open one asks nothing
    config = {
        "model_type": "bert",
        "architectures": ["BertForSequenceClassification"],
        "id2label": {"0": "no", "1": "yes"},
    }
    payload = {"config.json": json.dumps(config).encode(), "tokenizer.json": b"{}", "model.safetensors": WEIGHTS}

    def contents(repo: str, name: str) -> bytes:
        if repo == "org/unsupported" and name == "config.json":
            return b'{"model_type":"not-supported","architectures":["Unknown"]}'
        if name not in payload:  # as the Hub answers a file the repository doesn't have
            missing = httpx2.Response(404, request=httpx2.Request("GET", f"https://huggingface.co/{repo}/{name}"))
            raise RemoteEntryNotFoundError(f"{repo} has no {name}", response=missing)
        return payload[name]

    def target(repo: str, revision: str, name: str) -> Path:
        return cache / f"models--{repo.replace('/', '--')}" / "snapshots" / revision / name

    def cached(repo: str, name: str, *, revision: str) -> str | None:
        path = target(repo, revision, name)
        return str(path) if path.exists() else None

    def model_info(repo: str, **kwargs: Any) -> Any:
        return SimpleNamespace(
            id=repo,
            sha=state.sha,
            card_data=hf.ModelCardData(license=state.license),
            siblings=[SimpleNamespace(rfilename=name, size=len(contents(repo, name))) for name in payload],
        )

    def file_metadata(url: str) -> Any:
        org, name, _, _revision, file = urlsplit(url).path.lstrip("/").split("/", 4)
        data = contents(f"{org}/{name}", file)
        return SimpleNamespace(size=len(data), etag=hashlib.sha256(data).hexdigest())

    def download(repo: str, name: str, *, revision: str, tqdm_class: Any = None, **kwargs: Any) -> str:
        state.requested.append(name)
        destination = target(repo, revision, name)
        if destination.exists():
            return str(destination)
        data = contents(repo, name)
        blob = cache / f"models--{repo.replace('/', '--')}" / "blobs" / hashlib.sha256(data).hexdigest()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if blob.is_file():  # as the Hub cache does: a file another revision holds already is linked, not fetched
            destination.symlink_to(blob)
            return str(destination)
        temporary = blob.with_suffix(".incomplete")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        offset = temporary.stat().st_size if temporary.exists() else 0
        if name.endswith(".safetensors"):
            state.offsets.append(offset)
        bar = tqdm_class or downloads.quiet()
        with bar(total=len(data), initial=offset, desc=name) as progress, temporary.open("ab") as stream:
            for start in range(offset, len(data), 16384):
                chunk = data[start : start + 16384]
                stream.write(chunk)
                stream.flush()
                progress.update(len(chunk))
                if name.endswith(".safetensors"):
                    state.transferred.append(len(chunk))
                    if state.interrupt:
                        state.interrupt = False
                        raise KeyboardInterrupt
        temporary.replace(blob)
        destination.symlink_to(blob)
        return str(destination)

    def snapshot(repo: str, *, revision: str | None, allow_patterns: Any, ignore_patterns: Any, **kwargs: Any) -> str:
        for name in payload:
            if any(fnmatch(name, p) for p in allow_patterns) and not any(fnmatch(name, p) for p in ignore_patterns):
                download(repo, name, revision=revision or state.sha)
        return str(target(repo, revision or state.sha, "config.json").parent)

    monkeypatch.setattr(hf, "HfApi", lambda: SimpleNamespace(model_info=model_info))
    monkeypatch.setattr(hf, "try_to_load_from_cache", cached)
    monkeypatch.setattr(hf, "get_hf_file_metadata", file_metadata)
    monkeypatch.setattr(hf, "hf_hub_download", download)
    monkeypatch.setattr(hf, "snapshot_download", snapshot)
    monkeypatch.setattr(locate, "_remote", lambda *a: ([Tensor("classifier.weight", "F32", (2, 2))], 1, 65536))

    def transfer_file(item: downloads.File, progress: Any) -> Path:
        class Bar:
            def __init__(self, *args: Any, initial: int = 0, **kwargs: Any):
                self.n = initial
                progress(self.n)

            def __enter__(self) -> Bar:
                return self

            def __exit__(self, *args: Any) -> None:
                pass

            def update(self, n: int | float | None = 1) -> None:
                self.n += int(n or 0)
                progress(self.n)

        return Path(download(item.repo, item.name, revision=item.revision, tqdm_class=Bar))

    monkeypatch.setattr(downloads, "_download", transfer_file)
    return state


def test_pull_refuses_before_weights_and_remembers_a_licence_across_models(
    remote: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(UnsupportedModel, match="no installed adapter matches this model"):
        pull.prepare("org/unsupported")
    assert not remote.transferred
    assert "tokenizer.json" not in remote.requested  # compatibility looked only at config and asset metadata
    plan = pull.prepare("org/first")
    assert "generic path" in plan.report["support"]
    assert not remote.transferred
    with pytest.raises(LicenseNotAccepted, match="--accept-license"):
        plan.run()
    monkeypatch.setattr(downloads, "disk_space", lambda _: (65536, 1))
    with pytest.raises(InsufficientDisk, match=r"65,536.*1"):
        plan.run(accept_license=True)
    assert not remote.transferred
    assert not plan.licenses[0].accepted


def test_a_precision_a_name_carries_is_the_one_pull_checks_the_fit_of(
    remote: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup recommends kettei-core:q8 for a Mac where Kettei Core fits at 8 bits only, then pulls it: the fit is
    checked at the precision the name asks for (q8, or fp32 where only 16 bits fit), and a bare name's as published."""
    layers = [Tensor(f"encoder.layer.{i}.weight", "F32", (2048, 2048)) for i in range(10)]
    stored = [*layers, Tensor("classifier.weight", "F32", (2048, 4096))]  # about 101 MB at 16 bits, 61 MB at 8
    monkeypatch.setattr(locate, "_remote", lambda *a: (stored, 1, 65536))
    room = 80_445_440  # admission's budget of a 3.072 GB working set: between the two sizes
    mac = Device("Apple M1", "applegpu_g13g", 8 << 30, 3_072_000_000, (26, 1), True)
    monkeypatch.setattr(pull, "detect", lambda: mac)
    plan = pull.prepare("kettei-core:q8")
    assert plan.report["room_bytes"] == room
    with pytest.raises(InsufficientMemory, match="doesn't fit this Mac as published") as refused:
        pull.prepare("kettei-core")
    assert "At 8 bits it fits: ippon pull kettei-core:q8." in str(refused.value)  # the name to pull instead
    roomier = Device("Apple M1", "applegpu_g13g", 8 << 30, 3_200_000_000, (26, 1), True)  # 196 MB: 16 bits fit, 32 not
    monkeypatch.setattr(pull, "detect", lambda: roomier)
    pull.prepare("kettei-core")
    with pytest.raises(InsufficientMemory, match="doesn't fit this Mac at fp32"):  # never checked at bf16 instead
        pull.prepare("kettei-core:fp32")
    eight = Device("Apple M1", "applegpu_g13g", 8 << 30, 5_726_623_744, (26, 1), True)  # 5.33 GiB for its GPU
    big = [Tensor(f"encoder.layer.{i}.weight", "F32", (8192, 8192)) for i in range(40)]  # about 2.7 GB at 8 bits
    monkeypatch.setattr(locate, "_remote", lambda *a: (big, 1, 65536))
    monkeypatch.setattr(pull, "detect", lambda: eight)
    with pytest.raises(InsufficientMemory, match="at q8") as refused:  # nothing of Core fits: the model that does
        pull.prepare("kettei-core:q8")
    assert "ippon pull kettei-dot:bf16 fits this Mac, and `ippon recommend` lists what else does." in str(refused.value)
    assert not remote.transferred


def test_an_open_licence_asks_nothing_and_the_environment_can_accept(
    remote: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apache-2.0 or MIT weights download with no acceptance and record none; for another licence,
    IPPON_ACCEPT_LICENSE=1 accepts it as --accept-license does, as CI needs."""
    remote.license = "apache-2.0"
    plan = pull.prepare("org/first")
    plan.run()
    assert remote.transferred
    assert not plan.licenses[0].accepted  # nothing to remember
    # Ungated community models (Liquid d1) download smoothly without a clickwrap prompt while still preserving their
    # licence name and URL; Kettei's licence and non-commercial ones (CC BY-NC 4.0) require acceptance, with a
    # Python-first hint when called outside the CLI.
    kettei_lic = licences.resolve("Sqwish/Kettei-Core", "main", {}, ())
    assert not kettei_lic.open
    assert kettei_lic.name == "Kettei Community Model License 1.0"
    d1_lic = licences.resolve("LiquidAI/d1-2B", "main", {}, ())
    assert d1_lic.open
    assert d1_lic.name == "LFM Open License v1.0"
    remote.license = "cc-by-nc-4.0"
    other = pull.prepare("org/second")
    with pytest.raises(
        LicenseNotAccepted, match=r"Decider\.load\('org/second', accept_license=True\).*IPPON_ACCEPT_LICENSE=1"
    ):
        other.run()
    monkeypatch.setenv("IPPON_ACCEPT_LICENSE", "1")
    other.run()
    assert other.licenses[0].accepted


def test_cli_non_tty_refusal_and_two_models_share_one_acceptance(
    remote: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["pull", "org/first", "--no-sample"]) == 11
    assert "ippon pull org/first --no-sample --accept-license" in capsys.readouterr().err  # the command as typed
    assert not remote.transferred
    assert main(["pull", "org/first", "--no-sample", "--json", "--accept-license"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["path"]
    assert result["revision"] == remote.sha
    assert main(["pull", "org/second", "--no-sample", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["revision"] == remote.sha
    from ippon.storage import config_dir

    assert len(json.loads((config_dir() / "licenses.json").read_text())) == 1


def test_an_interrupted_pull_resumes_and_cached_weights_load_without_acceptance_or_network(
    remote: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = pull.prepare("org/first")
    events: list[downloads.Progress] = []
    remote.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        plan.run(accept_license=True, progress=events.append)
    assert events[-1].phase == "cancelled"
    assert not hub.downloaded_weights("org/first")
    plan = pull.prepare("org/first")
    plan.run(progress=events.append)
    assert events[-1].phase == "complete"  # resuming the exact byte range: test_transfer_preserves_partial_...
    assert events[-1].done == events[-1].total
    from ippon.storage import config_dir

    (config_dir() / "licenses.json").unlink()
    monkeypatch.setattr(downloads, "info", lambda *a: pytest.fail("cached weights must work offline"))
    path = hub.fetch("org/first", plan.report["downloads"][0]["patterns"])
    assert (path / "model.safetensors").read_bytes() == WEIGHTS


def test_direct_weight_loads_require_acceptance_including_gguf(remote: Any) -> None:
    from ippon.decider import Decider

    with pytest.raises(LicenseNotAccepted):
        Decider.load("org/first", warm_up=False)
    assert not remote.transferred
    with licences.accepting(True):
        path = hub.fetch("org/first", ("*.safetensors",))
    assert (path / "model.safetensors").exists()
    # A distinct licence must not inherit the earlier consent, including in the single-file entry point.
    from ippon.storage import config_dir

    (config_dir() / "licenses.json").unlink()
    with pytest.raises(LicenseNotAccepted):
        hub.pinned_file("org/second", remote.sha, "model.safetensors", "unused")
    assert remote.offsets == [0]

    # An interactive CLI approve callback passed through licences.accepting() lets any command (decide, serve, etc.)
    # prompt once on stderr, save the acceptance, and download seamlessly — while declining leaves nothing downloaded.
    asked: list[str] = []

    def _answer(allow: bool) -> Any:
        def _ask(lic: licences.Licence) -> bool:
            asked.append(lic.name)
            return allow

        return _ask

    with pytest.raises(LicenseNotAccepted), licences.accepting(approve=_answer(False)):
        hub.fetch("org/second", ("*.safetensors",))
    assert len(asked) == 1
    with licences.accepting(approve=_answer(True)):
        got = hub.fetch("org/second", ("*.safetensors",))
    assert (got / "model.safetensors").exists()
    assert len(asked) == 2
    # Once accepted, subsequent calls never prompt again.
    with licences.accepting(approve=lambda lic: pytest.fail("already accepted licence must not prompt again")):
        hub.fetch("org/second", ("*.safetensors",))


def test_cli_approve_licence_only_prompts_on_interactive_non_json_non_mcp_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse
    import io
    import sys

    from ippon.cli import _approve_licence

    class _TTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    licence = licences.Licence("Test License 1.0", "https://example.com/license", "k", "org/model")
    # Non-TTY, --json, and stdio mcp never prompt.
    assert _approve_licence(argparse.Namespace(json=False, command="decide")) is None
    monkeypatch.setattr(sys, "stdin", _TTY("y\n"))
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    assert _approve_licence(argparse.Namespace(json=True, command="decide")) is None
    assert _approve_licence(argparse.Namespace(json=False, command="mcp")) is None
    # Interactive non-json command prompts on stderr and reads yes/no cleanly.
    ask = _approve_licence(argparse.Namespace(json=False, command="decide"))
    assert ask is not None
    assert ask(licence) is True
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Test License 1.0" in captured.err
    assert "https://example.com/license" in captured.err
    monkeypatch.setattr(sys, "stdin", _TTY("n\n"))
    assert ask(licence) is False


def test_a_revision_is_ippon_s_to_delete_only_when_its_download_brought_new_files(
    remote: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A load, a pull and a pinned file's download record with one rule for what `ippon remove all --models` may
    delete: a revision whose download brought files the Hub cache didn't hold. A revision whose files the cache held
    already, under another revision, stays, whichever way it came, even when another app downloads a file of another
    revision meanwhile; a pinned file ippon downloaded goes too."""
    import huggingface_hub as hf
    from huggingface_hub import constants

    with licences.accepting(True):
        hub.fetch("org/first", ("*.json", "*.safetensors"))  # a load: every file new
    remote.sha = "b" * 40
    pull.prepare("org/first").run(accept_license=True)  # the same files at another commit: linked, not downloaded
    remote.sha = "c" * 40
    cached = hf.try_to_load_from_cache

    def another_app_downloads_meanwhile(*args: Any, **kwargs: Any) -> Any:
        (Path(constants.HF_HUB_CACHE) / "models--org--first" / "blobs" / ("0" * 64)).write_bytes(b"another revision")
        return cached(*args, **kwargs)

    monkeypatch.setattr(hf, "try_to_load_from_cache", another_app_downloads_meanwhile)
    with licences.accepting(True):
        hub.fetch("org/first", ("*.json", "*.safetensors"))
    monkeypatch.setattr(hf, "try_to_load_from_cache", cached)
    weights = hashlib.sha256(WEIGHTS).hexdigest()
    with licences.accepting(True):
        hub.pinned_file("org/third", remote.sha, "model.safetensors", weights)
    record = json.loads((cache_dir() / hub.DOWNLOADS).read_text())["snapshots"]
    ours = {key for key, snapshot in record.items() if snapshot["downloaded"]}
    assert ours == {"org/first@" + "a" * 40, "org/third@" + "c" * 40}
    assert len(record) == 4


def test_removing_ippon_s_downloads_leaves_a_fork_at_the_same_commit(remote: Any) -> None:
    """A fork shares its parent's commits: `ippon remove all --models` deletes the revision ippon downloaded, by its
    repository, and leaves the fork's revision at the same commit, which ippon never fetched."""
    import shutil

    from huggingface_hub import constants

    with licences.accepting(True):
        hub.fetch("org/first", ("*.json", "*.safetensors"))
    cache = Path(constants.HF_HUB_CACHE)
    shutil.copytree(cache / "models--org--first", cache / "models--fork--first", symlinks=True)
    for entry in (
        cache / "models--fork--first" / "snapshots" / ("a" * 40)
    ).iterdir():  # its own blobs, as the Hub links
        blob = Path(entry.readlink()).name
        entry.unlink()
        entry.symlink_to(Path("..") / ".." / "blobs" / blob)
    done = hub.delete_downloads()
    assert [line.split(":")[0] for line in done] == ["org/first"]
    assert not (cache / "models--org--first" / "snapshots" / ("a" * 40)).exists()
    fork = cache / "models--fork--first" / "snapshots" / ("a" * 40)
    assert fork.is_dir()
    assert all(f.exists() for f in fork.iterdir())


def test_removing_ippon_s_downloads_keeps_a_revision_holding_another_app_s_file(remote: Any) -> None:
    """Another app's quantization of the same commit sits in the snapshot ippon downloaded into: `ippon remove all
    --models` keeps the whole revision rather than delete a file ippon never fetched."""
    from huggingface_hub import constants

    with licences.accepting(True):
        hub.fetch("org/first", ("*.json", "*.safetensors"))
    repo = Path(constants.HF_HUB_CACHE) / "models--org--first"
    (repo / "blobs" / ("0" * 64)).write_bytes(b"another app's q4")
    (repo / "snapshots" / ("a" * 40) / "model-q4.gguf").symlink_to(Path("..") / ".." / "blobs" / ("0" * 64))
    done = hub.delete_downloads()
    assert done == [f"org/first: revision {'a' * 12} kept, it holds files another app downloaded"]
    assert (repo / "snapshots" / ("a" * 40) / "model-q4.gguf").read_bytes() == b"another app's q4"


@pytest.mark.parametrize("failure", ["cancel", "disconnect", "stall", "oversized", "hash", "disk"])
def test_transfer_preserves_partial_and_resumes_the_exact_http_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from huggingface_hub import constants

    data = b"weights" * 600_000  # 4.2 MB: past the 1 MiB reads and the 2 MiB where the disconnect cuts
    revision, etag = "a" * 40, hashlib.sha256(data).hexdigest()
    ranges: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_HEAD(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", etag)
            self.send_header("X-Repo-Commit", revision)
            self.end_headers()

        def do_GET(self) -> None:
            offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            ranges.append(offset)
            self.send_response(206 if offset else 200)
            self.send_header("Content-Length", str(len(data) - offset))
            if offset:
                self.send_header("Content-Range", f"bytes {offset}-{len(data) - 1}/{len(data)}")
            self.end_headers()
            try:
                for start in range(offset, len(data), 65536):
                    if failure in ("disconnect", "stall") and len(ranges) == 1 and start >= 2 << 20:
                        time.sleep(5 if failure == "stall" else 0)  # silent past the Hub's read timeout, then gone
                        return
                    self.wfile.write(data[start : start + 65536])
                    time.sleep(0.001)
            except (BrokenPipeError, ConnectionResetError):
                pass

    cache = tmp_path / "hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    monkeypatch.setattr(constants, "HF_HUB_DOWNLOAD_TIMEOUT", 1)
    partial = cache / "models--org--model" / "blobs" / f"{etag}.incomplete"
    item = downloads.File("org/model", revision, "model.safetensors", len(data), False, partial)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    monkeypatch.setattr("huggingface_hub.hf_hub_url", lambda *a, **kw: f"http://127.0.0.1:{server.server_port}/weights")
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()

    def cancel(event: downloads.Progress) -> None:
        if event.phase == "downloading" and 0 < event.done < event.total:
            raise KeyboardInterrupt

    try:
        if failure in ("oversized", "hash"):
            partial.parent.mkdir(parents=True)
            partial.write_bytes(data + b"extra" if failure == "oversized" else b"x" * len(data))
            with pytest.raises(DownloadFailed, match=r"removed.*repeat"):
                downloads.transfer([item])
            assert not partial.exists()
            expected_ranges = [0]
        elif failure == "disk":

            def full_disk(descriptor: int) -> None:
                raise OSError(errno.ENOSPC, "No space left on device")

            with monkeypatch.context() as fault:
                fault.setattr("os.fsync", full_disk)
                with pytest.raises(InsufficientDisk, match=r"free.*space"):
                    downloads.transfer([item])
            assert partial.read_bytes() == data
            expected_ranges = [0]
        else:
            begun = time.monotonic()
            with pytest.raises(KeyboardInterrupt if failure == "cancel" else DownloadFailed):
                downloads.transfer([item], cancel if failure == "cancel" else None)
            assert time.monotonic() - begun < 4  # a silent server fails the read at the timeout, not when it hangs up
            offset = partial.stat().st_size
            assert 0 < offset < len(data)
            expected_ranges = [0, offset]
        events: list[downloads.Progress] = []
        folder = downloads.transfer([item], events.append)
        assert ranges == expected_ranges
        assert (folder / item.name).read_bytes() == data
        assert not partial.exists()
        assert events[-1].phase == "complete"
        assert events[-1].done == len(data)
        # Pull records the weights' manifest; a later offline load first asks for configs with other patterns.
        monkeypatch.setenv("IPPON_CACHE", str(tmp_path / "cache"))
        hub.record_download("org/model", ("*.safetensors",), folder)
        monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
        assert hub.fetch("org/model", ("*.json",)) == folder
    finally:
        server.shutdown()
        server.server_close()
        serving.join()


def test_a_gguf_repository_named_alone_is_answered_with_its_builds_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GGUF repository holds a build per file and no config.json: naming it alone is a request to finish, with the
    build to name (LiquidAI/d1-Omni-2B-GGUF:Q8_0), not "no supported model configuration"."""
    files = ["README.md", "d1-Omni-2B-Q8_0.gguf", "mmproj-d1-Omni-2B-Q8_0.gguf"]
    listing = SimpleNamespace(
        id="LiquidAI/d1-Omni-2B-GGUF", sha="0" * 40, siblings=[SimpleNamespace(rfilename=f, size=1) for f in files]
    )
    monkeypatch.setattr(downloads, "info", lambda *a, **k: listing)
    with pytest.raises(InvalidRequest, match=r"name the one to load, LiquidAI/d1-Omni-2B-GGUF:Q8_0 "):
        downloads.configs("LiquidAI/d1-Omni-2B-GGUF")


def test_a_renamed_repository_downloads_under_the_name_it_was_asked_by(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Hub answers a renamed repository (org/old-name, now org/new-name) under its new name, and the cache keeps a
    fetch by the name asked: the weights must land beside the configs, not apart."""
    renamed = SimpleNamespace(id="org/new-name", sha="7ac669b8" * 5, siblings=[])
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda self, repo, **kw: renamed)
    assert downloads.info("org/old-name@" + "7ac669b8" * 5).id == "org/old-name"


def test_no_network_is_a_failed_download_never_a_model_ippon_cant_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """An agent reads unsupported_model (exit 4) as "take another build": only the Hub's own answer that the repository
    isn't there may say it, and a connection that failed is download_failed (exit 13), to run again."""

    def failing(error: Exception) -> Any:
        def model_info(self: Any, repo: str, **kw: Any) -> Any:
            raise error

        return model_info

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", failing(httpx2.ConnectError("no route to host")))
    with pytest.raises(DownloadFailed, match="run the command again"):
        downloads.info("org/model")
    missing = httpx2.Response(404, request=httpx2.Request("GET", "https://huggingface.co/api/models/org/model"))
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", failing(RepositoryNotFoundError("gone", response=missing)))
    with pytest.raises(ModelNotFound) as refused:
        downloads.info("org/model")
    # one line that says what to do, not the Hub's paragraphs about tokens
    assert (
        str(refused.value)
        == "no model org/model on Hugging Face, or it is private (if it is yours, log in with `hf auth login`)"
    )


def test_a_custom_licence_in_one_repository_is_not_accepted_for_another() -> None:
    """Accepting a custom licence that a card links as a file of its own repository (LICENSE) accepts it for that
    repository only, while repositories that link one licence's URL share its acceptance."""

    def custom(repo: str, link: str) -> licences.Licence:
        card = SimpleNamespace(to_dict=lambda: {"license": "other", "license_link": link})
        return licences.of_metadata(SimpleNamespace(id=repo, sha="c" * 40, card_data=card, siblings=[]))

    assert custom("a/model", "LICENSE").key != custom("b/model", "LICENSE").key
    url = "https://example.com/model-licence"
    assert custom("a/model", url).key == custom("b/model", url).key


@pytest.mark.parametrize(
    "front",
    [
        "license: apache-2.0  # see LICENSE\n",  # a comment after the value
        "license: other\nlicense_name: >-\n  kettei-community-model-license-1.0\nlicense_link: LICENSE\n",  # folded
        "tags:\n  - text-classification\n\"license\": 'mit'\n",  # a quoted key and value after a list
    ],
)
def test_a_folder_reads_its_licence_as_a_download_of_the_same_card_does(tmp_path: Path, front: str) -> None:
    """A loaded model's licence (from its folder's README.md) is the one its download asked about (from the Hub's
    metadata, which huggingface_hub reads as YAML), whatever YAML the card's front matter is written in."""
    (tmp_path / "README.md").write_text(f"---\n{front}---\n\n# A model\n", encoding="utf-8")
    metadata = metadata_load(tmp_path / "README.md")
    card = SimpleNamespace(to_dict=lambda: metadata)
    downloaded = licences.of_metadata(SimpleNamespace(id="org/model", sha="c" * 40, card_data=card, siblings=[]))
    assert licences.of_folder(tmp_path, "org/model", "c" * 40) == downloaded
