"""
Downloads the pinned Gemma 4 31B and E2B draft model checkpoints from Hugging Face Hub.
"""

import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

TARGET_REPO = "mlx-community/gemma-4-31b-it-4bit"
TARGET_REVISION = "696d436c404745a59f30e4939a658162b0a9e57f"

DRAFT_REPO = "mlx-community/gemma-4-e2b-it-4bit"
DRAFT_REVISION = "238767527555cb75a05732a84dff5d6ba0dd6809"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading Gemma 4 Checkpoints from Hugging Face Hub")
    print("=" * 60)

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
