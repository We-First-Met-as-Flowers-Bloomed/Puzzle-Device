"""Run the exact board-side vision pipeline against one saved image on a PC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from puzzle_vision import PuzzleVisionPipeline, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="input camera image")
    parser.add_argument("--config", default=None, help="runtime JSON config")
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="disable playing-card seam texture scoring",
    )
    parser.add_argument(
        "--output",
        default="debug_result.jpg",
        help="debug overlay image path",
    )
    parser.add_argument(
        "--json",
        default="result.json",
        help="result JSON path",
    )
    args = parser.parse_args()

    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"cannot read image: {args.image}", file=sys.stderr)
        return 2

    config = load_config(args.config)
    pipeline = PuzzleVisionPipeline(config)
    result = pipeline.process(
        np.ascontiguousarray(frame),
        use_texture=config.texture_enabled and not args.no_texture,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result.debug_bgr)
    Path(args.json).write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
