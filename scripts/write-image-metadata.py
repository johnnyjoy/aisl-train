# SPDX-License-Identifier: Apache-2.0
"""Write /opt/aisl-train/image-metadata.json during the image build."""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def pkg(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> int:
    meta = {
        "aisl_train_version": pkg("aisl-train"),
        "aisl_train_commit": Path("/opt/aisl-train/AISL_TRAIN_COMMIT").read_text(encoding="utf-8").strip(),
        "aisl_ref_requested": Path("/workspace/AISL_REF").read_text(encoding="utf-8").strip(),
        "aisl_commit": Path("/workspace/AISL_COMMIT").read_text(encoding="utf-8").strip(),
        "python": sys.version.split()[0],
        "packages": {
            name: pkg(name)
            for name in (
                "torch",
                "transformers",
                "trl",
                "peft",
                "bitsandbytes",
                "accelerate",
                "datasets",
            )
        },
    }
    out = Path("/opt/aisl-train/image-metadata.json")
    out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
