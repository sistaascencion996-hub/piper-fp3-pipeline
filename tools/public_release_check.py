#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SUFFIXES = {
    ".pth", ".pt", ".ckpt", ".safetensors",
    ".h5", ".hdf5", ".bag", ".svo",
    ".pem", ".key",
}
MAX_PUBLIC_FILE_BYTES = 50 * 1024 * 1024

TEXT_SUFFIXES = {
    ".py", ".ps1", ".sh", ".md", ".txt", ".json",
    ".yaml", ".yml", ".toml", ".cff", ".ini", ".cfg",
}

SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),

    # Private/local IPv4 addresses.
    re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|"
        r"192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),

    # Machine-specific absolute home-directory paths.
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    re.compile(r"C:\\\\Users\\\\[^\\\s]+\\\\", re.IGNORECASE),

    re.compile(r"10\\.110\\.129\\.91"),
    re.compile(r"C:\\\\Users\\\\12899(?:\\\\|\\b)", re.IGNORECASE),
]

ALLOW_LOCAL = {
    ROOT / "config" / "pipeline.local.json",
}

errors: list[str] = []

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    rel = path.relative_to(ROOT)

    if ".git" in rel.parts or "__pycache__" in rel.parts:
        continue

    if path.resolve() == Path(__file__).resolve():
        continue

    if path in ALLOW_LOCAL:
        # This file ships only as a local convenience file in the downloadable
        # package and is ignored by .gitignore.
        continue

    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        errors.append(f"forbidden large/private artifact: {rel}")

    if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
        errors.append(
            f"file larger than 50 MiB: {rel} "
            f"({path.stat().st_size / 1024 / 1024:.1f} MiB)"
        )

    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {
        "LICENSE", ".gitignore", ".gitattributes", ".env.example"
    }:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible secret in {rel}: {pattern.pattern}")

if errors:
    print("PUBLIC RELEASE CHECK: FAIL")
    for error in errors:
        print(" -", error)
    sys.exit(1)

print("PUBLIC RELEASE CHECK: PASS")
