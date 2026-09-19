"""Hashing, deterministic seed derivation, and immutable run support."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def sha256(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_turn_seed(generation_seed: int, stimulus_id: str, turn_index: int, speaker: str) -> int:
    """Condition-free v1 mapping, enabling common random numbers across conditions."""
    message = f"v1-blake2b|{generation_seed}|{stimulus_id}|{turn_index}|{speaker}".encode()
    return int.from_bytes(hashlib.blake2b(message, digest_size=8).digest(), "big") % (2**63 - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def software_manifest() -> dict[str, Any]:
    versions = {"python": sys.version, "platform": platform.platform()}
    for package in ("numpy", "torch", "transformers", "sklearn", "pandas"):
        try:
            module = __import__(package if package != "sklearn" else "sklearn")
            versions[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[package] = "not_installed"
    return versions


def git_state(root: Path) -> dict[str, Any]:
    git_binary = shutil.which("git")
    if git_binary is None:
        desktop_root = Path(os.environ.get("LOCALAPPDATA", "")) / "GitHubDesktop"
        candidates = sorted(desktop_root.glob("app-*/resources/app/git/cmd/git.exe"))
        git_binary = str(candidates[-1]) if candidates else None

    def git(*args: str) -> str | None:
        if git_binary is None:
            return None
        try:
            return subprocess.check_output([git_binary, *args], cwd=root, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "git_available": status is not None,
    }


def make_run_id(config: dict[str, Any]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{sha256(canonical_json(config))[:10]}"


def create_immutable_run_dir(root: Path, run_id: str) -> Path:
    path = root / run_id
    # A directory is immutable once it has a manifest. Empty pre-created directories are not allowed.
    if path.exists():
        raise FileExistsError(f"Run directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    target = run_dir / "manifest.json"
    if target.exists():
        raise FileExistsError(f"Manifest already exists: {target}")
    payload = {
        **manifest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "software": software_manifest(),
    }
    target.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return target
