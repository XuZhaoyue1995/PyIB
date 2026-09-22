"""Check the immutable source snapshot without accessing the original tree."""

import hashlib
import json
from pathlib import Path
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        path = root / entry["package_path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(entry["package_path"])
    if failures:
        print("Source hash mismatch: " + ", ".join(failures), file=sys.stderr)
        sys.exit(1)
    print(f"Verified {len(manifest['files'])} preserved source files.")
