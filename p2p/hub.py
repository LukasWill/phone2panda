"""Move data and checkpoints between Colab, Kaggle and this repo via the Hugging Face Hub.

Set HF_TOKEN (Colab: Secrets panel; Kaggle: Add-ons -> Secrets). Repos are created private.
"""
from __future__ import annotations

import os
from pathlib import Path


def push(local_dir: str, repo_id: str, path_in_repo: str = "", repo_type: str = "dataset"):
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type=repo_type, path_in_repo=path_in_repo or None)


def pull(repo_id: str, local_dir: str, allow_patterns: list[str] | None = None, repo_type: str = "dataset") -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow_patterns,
                                  token=os.environ.get("HF_TOKEN")))
