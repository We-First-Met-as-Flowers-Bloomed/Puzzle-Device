"""Fit undistorted camera-frame pixels to robot millimetres."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def apply_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = (matrix @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "points",
        help=(
            "JSON containing frame_px and robot_mm arrays with at least "
            "four corresponding points"
        ),
    )
    parser.add_argument("--output", default="frame_to_robot.json")
    args = parser.parse_args()

    values = json.loads(Path(args.points).read_text(encoding="utf-8"))
    frame = np.asarray(values["frame_px"], dtype=np.float64)
    robot = np.asarray(values["robot_mm"], dtype=np.float64)
    if frame.shape != robot.shape or frame.ndim != 2 or frame.shape[1] != 2:
        raise SystemExit("frame_px and robot_mm must both be N x 2 arrays")
    if len(frame) < 4:
        raise SystemExit("at least four point pairs are required")

    matrix, _ = cv2.findHomography(
        frame.astype(np.float32),
        robot.astype(np.float32),
        method=0,
    )
    if matrix is None:
        raise SystemExit("homography fit failed")
    predicted = apply_homography(frame, matrix)
    errors = np.linalg.norm(predicted - robot, axis=1)
    payload = {
        "frame_to_robot": matrix.tolist(),
        "rms_error_mm": float(np.sqrt(np.mean(errors * errors))),
        "max_error_mm": float(errors.max()),
        "point_count": int(len(frame)),
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
