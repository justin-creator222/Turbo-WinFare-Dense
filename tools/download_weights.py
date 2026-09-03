"""
Downloads the pinned Gemma 4 31B and E2B draft model checkpoints from Hugging Face Hub.
"""

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

TARGET_REPO = "mlx-community/gemma-4-31b-it-4bit"
TARGET_REVISION = "696d436c404745a59f30e4939a658162b0a9e57f"

DRAFT_REPO = "mlx-community/gemma-4-e2b-it-4bit"
DRAFT_REVISION = "238767527555cb75a05732a84dff5d6ba0dd6809"


def ensure_huggingface_hub():
    """Checks if huggingface_hub is installed. If not, automatically runs pip install huggingface_hub."""
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        pass

    print("huggingface_hub is not installed. Installing huggingface_hub automatically via pip...")
    cmd = [sys.executable, "-m", "pip", "install", "huggingface_hub"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        output_lines = []
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                output_lines.append(line)
                print("  " + line)
        rc = proc.wait()

        if rc != 0 and any("break-system-packages" in l for l in output_lines):
            print("Retrying installation with --break-system-packages...")
            retry_cmd = cmd + ["--break-system-packages"]
            proc = subprocess.Popen(retry_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    print("  " + line)
            rc = proc.wait()

        if rc != 0:
            print(f"Error: pip install huggingface_hub failed with exit code {rc}", file=sys.stderr)
            return False

        import importlib
        importlib.invalidate_caches()
        import huggingface_hub  # noqa: F401
        print("Successfully installed huggingface_hub.\n")
        return True
    except Exception as e:
        print(f"Error: Failed to install huggingface_hub automatically: {e}", file=sys.stderr)
        return False


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading Gemma 4 Checkpoints from Hugging Face Hub")
    print("=" * 60)

    if not ensure_huggingface_hub():
        sys.exit(1)

    from huggingface_hub import snapshot_download

    # 1. Download Draft E2B
    draft_dir = MODELS_DIR / "gemma-4-e2b-it-4bit"
    print(f"\n[1/2] Downloading {DRAFT_REPO} (@ {DRAFT_REVISION[:8]})...")
    t0 = time.time()
    snapshot_download(
        repo_id=DRAFT_REPO,
        revision=DRAFT_REVISION,
        local_dir=str(draft_dir),
        max_workers=4
    )
    print(f"Draft model download complete in {time.time() - t0:.1f}s -> {draft_dir}")

    # 2. Download Target 31B
    target_dir = MODELS_DIR / "gemma-4-31b-it-4bit"
    print(f"\n[2/2] Downloading {TARGET_REPO} (@ {TARGET_REVISION[:8]})...")
    t0 = time.time()
    snapshot_download(
        repo_id=TARGET_REPO,
        revision=TARGET_REVISION,
        local_dir=str(target_dir),
        max_workers=4
    )
    print(f"Target 31B model download complete in {time.time() - t0:.1f}s -> {target_dir}")
    print("\nAll model weights downloaded successfully.")


if __name__ == "__main__":
    main()
