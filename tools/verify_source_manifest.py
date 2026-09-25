"""Verify current sources and report preserved versus updated provenance."""

import hashlib
import json
from pathlib import Path
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    updated = 0
    for entry in manifest["files"]:
        path = root / entry["package_path"]
        if not path.is_file():
            failures.append(entry["package_path"])
            continue
        content = path.read_bytes()
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            failures.append(entry["package_path"])
        original = entry.get("original_source_sha256", entry["sha256"])
        updated += original != entry["sha256"]
    if failures:
        print("Source hash mismatch: " + ", ".join(failures), file=sys.stderr)
        sys.exit(1)
    total = len(manifest["files"])
    print(f"Verified {total} current source files: {total - updated} preserved, {updated} updated.")
