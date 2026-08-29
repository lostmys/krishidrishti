from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from krishidrishti_ai.config import load_config
from krishidrishti_ai.services.inference import DiagnosisService


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--image", required=True); args = parser.parse_args()
    print(DiagnosisService(load_config(args.config)).diagnose(Path(args.image).read_bytes()))


if __name__ == "__main__": main()
