"""Click camera pixels and pair them with robot coordinates for calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
RobotPoint = Tuple[float, float]


def pulse_to_mm(
    pulse: float,
    zero_pulse: float,
    pulses_per_mm: float,
    direction: int,
) -> float:
    if pulses_per_mm <= 0:
        raise ValueError("pulses_per_mm must be positive")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    return direction * (pulse - zero_pulse) / pulses_per_mm


def display_to_frame(
    x: int,
    y: int,
    scale: float,
    frame_width: int,
    frame_height: int,
) -> Point:
    if scale <= 0:
        raise ValueError("scale must be positive")
    frame_x = min(frame_width - 1, max(0, int(round(x / scale))))
    frame_y = min(frame_height - 1, max(0, int(round(y / scale))))
    return frame_x, frame_y


def make_payload(
    frame_points: Sequence[Point],
    robot_points: Optional[Sequence[RobotPoint]],
) -> dict:
    if robot_points is not None and len(frame_points) != len(robot_points):
        raise ValueError("frame and robot point counts must match")
    payload = {
        "frame_px": [[int(x), int(y)] for x, y in frame_points],
    }
    if robot_points is not None:
        payload["robot_mm"] = [
            [float(x), float(y)] for x, y in robot_points
        ]
    return payload


def _read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def _undistort_from_config(image: np.ndarray, path: Path) -> np.ndarray:
    values = json.loads(path.read_text(encoding="utf-8"))
    camera_matrix = values.get("camera_matrix")
    distortion = values.get("distortion_coefficients")
    if camera_matrix is None and distortion is None:
        return image
    if camera_matrix is None or distortion is None:
        raise ValueError(
            "camera_matrix and distortion_coefficients must be set together"
        )
    return cv2.undistort(
        image,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
    )


def _draw_points(
    image: np.ndarray,
    points: Sequence[Point],
    scale: float,
) -> np.ndarray:
    canvas = image.copy()
    for index, (frame_x, frame_y) in enumerate(points, start=1):
        x = int(round(frame_x * scale))
        y = int(round(frame_y * scale))
        cv2.drawMarker(
            canvas,
            (x, y),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            18,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{index}: ({frame_x}, {frame_y})",
            (x + 8, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _collect_frame_points(
    frame: np.ndarray,
    max_width: int,
    max_height: int,
) -> Optional[List[Point]]:
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    display = cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    points: List[Point] = []
    window = "Calibration points"

    def refresh() -> None:
        cv2.imshow(window, _draw_points(display, points, scale))

    def mouse_callback(
        event: int,
        x: int,
        y: int,
        _flags: int,
        _parameter: object,
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            point = display_to_frame(x, y, scale, width, height)
            points.append(point)
            print(f"point {len(points)}: frame_px={point}", flush=True)
            refresh()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            removed = points.pop()
            print(f"removed frame_px={removed}", flush=True)
            refresh()

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse_callback)
    refresh()
    print(
        "Left click: add point | Right click/Backspace: undo | "
        "Enter: finish | Esc: cancel",
        flush=True,
    )
    while True:
        key = cv2.waitKey(20) & 0xFF
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            return None
        if key in (10, 13):
            if len(points) >= 4:
                cv2.destroyWindow(window)
                return points
            print("At least four points are required.", flush=True)
        elif key in (8, 127, ord("z")) and points:
            removed = points.pop()
            print(f"removed frame_px={removed}", flush=True)
            refresh()
        elif key == 27:
            cv2.destroyWindow(window)
            return None


def _read_pair(prompt: str) -> Tuple[float, float]:
    while True:
        raw = input(prompt).strip().replace(",", " ")
        parts = raw.split()
        if len(parts) != 2:
            print("Please enter two numbers separated by a space.")
            continue
        try:
            first, second = float(parts[0]), float(parts[1])
        except ValueError:
            print("Both values must be numbers.")
            continue
        if not (math.isfinite(first) and math.isfinite(second)):
            print("Both values must be finite.")
            continue
        return first, second


def _collect_robot_points(
    frame_points: Sequence[Point],
    args: argparse.Namespace,
) -> List[RobotPoint]:
    pulse_mode = args.x_pulses_per_mm is not None
    robot_points: List[RobotPoint] = []
    for index, (frame_x, frame_y) in enumerate(frame_points, start=1):
        if pulse_mode:
            pulse_x, pulse_y = _read_pair(
                f"[{index}/{len(frame_points)}] pixel=({frame_x}, {frame_y}), "
                "enter Xpulse Ypulse: "
            )
            robot_x = pulse_to_mm(
                pulse_x,
                args.x_zero_pulse,
                args.x_pulses_per_mm,
                args.x_direction,
            )
            robot_y = pulse_to_mm(
                pulse_y,
                args.y_zero_pulse,
                args.y_pulses_per_mm,
                args.y_direction,
            )
            print(f"  -> robot_mm=({robot_x:.6f}, {robot_y:.6f})")
        else:
            robot_x, robot_y = _read_pair(
                f"[{index}/{len(frame_points)}] pixel=({frame_x}, {frame_y}), "
                "enter Xmm Ymm: "
            )
        robot_points.append((robot_x, robot_y))
    return robot_points


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Click calibration points in an original camera frame and pair "
            "them with robot coordinates."
        )
    )
    parser.add_argument("image", help="original camera image")
    parser.add_argument(
        "--output",
        default="calibration_points.json",
        help="output JSON accepted by fit_frame_to_robot.py",
    )
    parser.add_argument(
        "--config",
        help="optional runtime config; applies its lens undistortion first",
    )
    parser.add_argument(
        "--pixels-only",
        action="store_true",
        help="save frame_px only and enter robot_mm manually later",
    )
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--x-pulses-per-mm", type=float)
    parser.add_argument("--y-pulses-per-mm", type=float)
    parser.add_argument("--x-zero-pulse", type=float, default=0.0)
    parser.add_argument("--y-zero-pulse", type=float, default=0.0)
    parser.add_argument("--x-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--y-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing output file",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.max_width <= 0 or args.max_height <= 0:
        parser.error("--max-width and --max-height must be positive")
    pulse_values = (args.x_pulses_per_mm, args.y_pulses_per_mm)
    if (pulse_values[0] is None) != (pulse_values[1] is None):
        parser.error(
            "--x-pulses-per-mm and --y-pulses-per-mm must be used together"
        )
    if any(value is not None and value <= 0 for value in pulse_values):
        parser.error("pulses-per-mm values must be positive")
    if args.pixels_only and pulse_values[0] is not None:
        parser.error("--pixels-only cannot be combined with pulse conversion")

    try:
        frame = _read_image(Path(args.image))
        if args.config:
            frame = _undistort_from_config(frame, Path(args.config))
        points = _collect_frame_points(
            frame,
            args.max_width,
            args.max_height,
        )
    except (OSError, ValueError, json.JSONDecodeError, cv2.error) as error:
        print(f"error: {error}")
        return 2
    if points is None:
        print("Cancelled; no file written.")
        return 1

    robot_points = (
        None if args.pixels_only else _collect_robot_points(points, args)
    )
    output = Path(args.output)
    if output.exists() and not args.force:
        answer = input(f"{output} exists. Overwrite? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Cancelled; existing file was not changed.")
            return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(make_payload(points, robot_points), indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(points)} calibration points to {output}")
    if robot_points is None:
        print("Add a matching robot_mm array before fitting the homography.")
    else:
        print(
            "Next: python tools/fit_frame_to_robot.py "
            f'"{output}"'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
