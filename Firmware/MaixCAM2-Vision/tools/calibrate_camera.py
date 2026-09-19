"""Estimate camera intrinsics from checkerboard images captured before testing."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="glob, for example calibration/*.jpg")
    parser.add_argument("--columns", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=20.0)
    parser.add_argument("--output", default="camera_calibration.json")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit("no calibration images matched")

    pattern = (args.columns, args.rows)
    object_template = np.zeros(
        (args.columns * args.rows, 3), dtype=np.float32
    )
    object_template[:, :2] = (
        np.mgrid[0 : args.columns, 0 : args.rows]
        .T.reshape(-1, 2)
        .astype(np.float32)
        * args.square_mm
    )

    object_points = []
    image_points = []
    image_size = None
    accepted = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            continue
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                40,
                1e-3,
            ),
        )
        object_points.append(object_template.copy())
        image_points.append(corners)
        accepted.append(path)

    if image_size is None or len(accepted) < 8:
        raise SystemExit(
            f"only {len(accepted)} valid boards; capture at least 8 varied views"
        )

    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    payload = {
        "image_size": list(image_size),
        "rms_reprojection_px": float(rms),
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "accepted_images": accepted,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
