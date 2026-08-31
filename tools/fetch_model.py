"""
Downloads a Gemma 4 checkpoint and converts it into a .g4dense container, in one step.

This is what the web UI runs when a new user has no models. It exists separately from
download_weights.py because the UI needs three things that tool does not provide: one model at a
time, machine-readable progress, and a conversion step at the end.

Progress is written to stdout as single lines the server parses:

    STAGE <name>                     download | convert | verify
    PROGRESS <percent> <message>     percent is 0-100 across the WHOLE job
    DONE <container path>
    ERROR <message>

Anything else on stdout is human-readable detail and is ignored by the parser. stdout is
line-buffered and flushed explicitly, because the server reads it as the job runs -- a buffered
pipe would make a 40-minute job look frozen.

Usage:
    python tools/fetch_model.py --model e2b
    python tools/fetch_model.py --model 31b
    python tools/fetch_model.py --check          # report what is installed, then exit
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# Pinned revisions, matching docs/MODEL_SOURCES.md. A moving target would silently change the
# weights under a container that claims to be reproducible.
MODELS = {
    "e2b": {
        "repo": "mlx-community/gemma-4-e2b-it-4bit",
        "revision": "238767527555cb75a05732a84dff5d6ba0dd6809",
        "checkpoint": "gemma-4-e2b-it-4bit",
        "container": "gemma-4-e2b-dense.g4dense",
        "label": "Gemma 4 E2B",
    },
    "31b": {
        "repo": "mlx-community/gemma-4-31b-it-4bit",
        "revision": "696d436c404745a59f30e4939a658162b0a9e57f",
        "checkpoint": "gemma-4-31b-it-4bit",
        "container": "gemma-4-31b-dense.g4dense",
        "label": "Gemma 4 31B Dense",
    },
}

# The download is the long pole, so it owns most of the progress bar.
DOWNLOAD_SHARE = 80.0


def emit(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def stage(name):
    emit(f"STAGE {name}")


def progress(pct, message):
    emit(f"PROGRESS {max(0.0, min(100.0, pct)):.1f} {message}")


def dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def gb(n):
    return n / (1024.0 ** 3)


def check_environment():
    """Reports the runtime prerequisites as JSON. The UI calls this before offering a download,
    so a missing dependency is a clear message up front rather than a failure 20 minutes in."""
    out = {"python": sys.version.split()[0], "huggingface_hub": None, "models": {}}
    try:
        import huggingface_hub
        out["huggingface_hub"] = huggingface_hub.__version__
    except ImportError:
        pass

    try:
        usage = shutil.disk_usage(str(REPO_ROOT))
        out["disk_free_bytes"] = usage.free
    except OSError:
        out["disk_free_bytes"] = 0

    for key, m in MODELS.items():
        ckpt = MODELS_DIR / m["checkpoint"]
        cont = MODELS_DIR / m["container"]
        out["models"][key] = {
            "label": m["label"],
            "repo": m["repo"],
            "checkpoint_present": ckpt.is_dir(),
            "checkpoint_bytes": dir_size(ckpt) if ckpt.is_dir() else 0,
            "container_present": cont.is_file(),
            "container_bytes": cont.stat().st_size if cont.is_file() else 0,
        }
    print(json.dumps(out))
    return 0


def remote_total_bytes(repo, revision):
    """Total download size, so the progress bar is real rather than a spinner. Returns 0 if the
    metadata call fails -- the download still runs, it just reports bytes instead of percent."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        return sum(getattr(f, "size", 0) or 0 for f in (info.siblings or []))
    except Exception:
        return 0


def do_download(m, dest):
    from huggingface_hub import snapshot_download

    total = remote_total_bytes(m["repo"], m["revision"])
    dest.mkdir(parents=True, exist_ok=True)

    # snapshot_download reports progress through tqdm, which is awkward to parse and writes to
    # stderr. Polling the destination directory is less precise but robust, and it is what the
    # user actually cares about: bytes on disk out of bytes expected.
    done = threading.Event()

    def watch():
        while not done.wait(1.5):
            have = dir_size(dest)
            if total > 0:
                pct = (have / total) * DOWNLOAD_SHARE
                progress(pct, f"downloading {gb(have):.1f} / {gb(total):.1f} GB")
            else:
                progress(0.0, f"downloading {gb(have):.1f} GB")

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        snapshot_download(
            repo_id=m["repo"],
            revision=m["revision"],
            local_dir=str(dest),
            max_workers=4,
        )
    finally:
        done.set()
        watcher.join(timeout=3)

    progress(DOWNLOAD_SHARE, f"downloaded {gb(dir_size(dest)):.1f} GB")


def do_convert(m, src, out_path):
    stage("convert")
    progress(DOWNLOAD_SHARE, "starting conversion")

    cmd = [sys.executable, str(REPO_ROOT / "tools" / "convert_hf_to_g4dense.py"),
           "--input", str(src), "--out", str(out_path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(REPO_ROOT))

    # The converter reports its stages but not a percentage. Map the stages it does print onto
    # the remaining 20% so the bar keeps moving through a step that takes minutes.
    marks = [
        ("Packing embeddings", 84.0, "packing embeddings"),
        ("Packing per-layer", 86.0, "packing per-layer embeddings"),
        ("transformer blocks", 88.0, "packing transformer layers"),
        ("Computing payload", 96.0, "checksumming and writing"),
    ]
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            emit("  " + line)
        for needle, pct, msg in marks:
            if needle in line:
                progress(pct, msg)
                break

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"conversion failed with exit code {rc}")


def main():
    ap = argparse.ArgumentParser(description="Download and convert a Gemma 4 checkpoint.")
    ap.add_argument("--model", choices=sorted(MODELS.keys()), help="which model to fetch")
    ap.add_argument("--check", action="store_true", help="report installed prerequisites as JSON")
    args = ap.parse_args()

    if args.check:
        return check_environment()
    if not args.model:
        ap.error("--model is required unless --check is given")

    m = MODELS[args.model]
    ckpt = MODELS_DIR / m["checkpoint"]
    out_path = MODELS_DIR / m["container"]

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        emit("ERROR huggingface_hub is not installed. Install it with: pip install huggingface_hub")
        return 2

    t0 = time.time()
    try:
        if out_path.is_file():
            emit(f"{m['label']} is already converted.")
            progress(100.0, "already installed")
            emit(f"DONE {out_path}")
            return 0

        stage("download")
        if ckpt.is_dir() and any(ckpt.glob("*.safetensors")):
            emit(f"Checkpoint already present at {ckpt}, skipping download.")
            progress(DOWNLOAD_SHARE, "checkpoint already downloaded")
        else:
            do_download(m, ckpt)

        do_convert(m, ckpt, out_path)

        stage("verify")
        progress(97.0, "verifying container")
        rc = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "verify_weights.py"),
                             str(out_path)], cwd=str(REPO_ROOT)).returncode
        if rc != 0:
            raise RuntimeError("the converted container failed verification")

        progress(100.0, f"ready in {time.time() - t0:.0f}s")
        emit(f"DONE {out_path}")
        return 0

    except KeyboardInterrupt:
        emit("ERROR cancelled")
        return 130
    except Exception as e:
        # A partial container would be loadable-looking garbage, so it must not survive a
        # failure. The checkpoint is kept: it is the expensive half and is resumable.
        if out_path.is_file():
            try:
                out_path.unlink()
                emit(f"Removed incomplete container {out_path}")
            except OSError:
                pass
        emit(f"ERROR {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
