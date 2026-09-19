from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class VisionError(RuntimeError):
    """Base class for a recoverable vision failure."""


class DetectionError(VisionError):
    """Input paper, divider, or pieces could not be measured reliably."""


class SolveError(VisionError):
    """No geometrically valid rectangular assembly was found."""


class AmbiguousSolutionError(SolveError):
    """More than one non-equivalent assembly remains."""


def signed_area(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    return 0.5 * float(
        np.dot(p[:, 0], np.roll(p[:, 1], -1))
        - np.dot(p[:, 1], np.roll(p[:, 0], -1))
    )


def ensure_positive_winding(points: np.ndarray) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if signed_area(result) < 0:
        result = result[::-1].copy()
    return result


def polygon_area(points: np.ndarray) -> float:
    return abs(signed_area(points))


def order_quad(points: np.ndarray) -> np.ndarray:
    """Return image quad as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    edge_a = ordered[1] - ordered[0]
    edge_b = ordered[2] - ordered[1]
    if edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0] < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


def apply_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    mapped = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    if np.any(np.abs(mapped[:, 2]) < 1e-9):
        raise ValueError("homography maps a point to infinity")
    return mapped[:, :2] / mapped[:, 2:3]


def rotation_matrix(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rigid_matrix(angle_rad: float, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = rotation_matrix(angle_rad)
    matrix[:2, 2] = np.asarray(translation, dtype=np.float64)
    return matrix


def rotation_about(angle_rad: float, center: np.ndarray) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64)
    rotate = rigid_matrix(angle_rad, np.zeros(2))
    before = rigid_matrix(0.0, -center)
    after = rigid_matrix(0.0, center)
    return after @ rotate @ before


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return apply_homography(points, matrix)


def wrap_degrees(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def matrix_angle_degrees(matrix: np.ndarray) -> float:
    return wrap_degrees(
        np.degrees(np.arctan2(float(matrix[1, 0]), float(matrix[0, 0])))
    )


def point_segment_distance(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    delta = b - a
    denom = float(np.dot(delta, delta))
    if denom < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip(((pts - a) @ delta) / denom, 0.0, 1.0)
    closest = a + t[:, None] * delta
    return np.linalg.norm(pts - closest, axis=1)


def point_line_distance(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    a = np.asarray(start, dtype=np.float64)
    direction = np.asarray(end, dtype=np.float64) - a
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return np.linalg.norm(pts - a, axis=1)
    return np.abs(
        direction[0] * (pts[:, 1] - a[1])
        - direction[1] * (pts[:, 0] - a[0])
    ) / length


@dataclass
class PaperObservation:
    image_bgr: np.ndarray
    frame_quad_px: np.ndarray
    frame_to_paper_px: np.ndarray
    paper_px_to_frame: np.ndarray
    divider_y_px: float
    paper_color_lab: np.ndarray
    detection_score: float
    px_per_mm: float

    @property
    def width_mm(self) -> float:
        return self.image_bgr.shape[1] / self.px_per_mm

    @property
    def height_mm(self) -> float:
        return self.image_bgr.shape[0] / self.px_per_mm

    @property
    def divider_y_mm(self) -> float:
        return self.divider_y_px / self.px_per_mm


@dataclass
class PieceObservation:
    piece_id: int
    contour_px: np.ndarray
    polygon_mm: np.ndarray
    mask: np.ndarray
    source_center_mm: np.ndarray
    pickup_mm: np.ndarray
    area_mm2: float
    approximation_error_mm: float
    boundary_reconstructed: bool = False


@dataclass
class AssemblyMetrics:
    fill_ratio: float
    overlap_ratio: float
    hole_ratio: float
    boundary_p95_mm: float
    boundary_p99_mm: float
    side_coverage_min: float
    convexity_ratio: float
    corner_error_max_mm: float
    rectangle_center_mm: np.ndarray
    rectangle_size_mm: Tuple[float, float]
    rectangle_box_mm: np.ndarray
    connected_components: int


@dataclass
class AssemblySolution:
    transforms: Dict[int, np.ndarray]
    metrics: AssemblyMetrics
    geometry_score: float
    texture_score: float
    total_score: float
    explored_nodes: int
    valid_candidate_count: int
    search_complete: bool
    score_margin: Optional[float] = None
    preserve_center_symmetry: bool = False
    center_symmetric_piece_pair: Optional[Tuple[int, int]] = None
    trained_template_name: Optional[str] = None
    placement_spread_scale: float = 1.0


@dataclass
class MoveCommand:
    piece_id: int
    source_center_paper_mm: np.ndarray
    pickup_paper_mm: np.ndarray
    target_center_paper_mm: np.ndarray
    place_paper_mm: np.ndarray
    rotation_cw_deg: float
    pickup_robot_mm: np.ndarray
    place_robot_mm: np.ndarray
    rotation_robot_deg: float
    target_polygon_paper_mm: np.ndarray

    def to_uart_dict(
        self,
        xy_pulses_per_mm: float,
    ) -> Dict[str, object]:
        clockwise_rotation = float(self.rotation_robot_deg) % 360.0
        if abs(clockwise_rotation - 360.0) < 1e-9:
            clockwise_rotation = 0.0
        return {
            "piece_id": int(self.piece_id),
            "pickup_x_pulse": int(
                round(float(self.pickup_robot_mm[0]) * xy_pulses_per_mm)
            ),
            "pickup_y_pulse": int(
                round(float(self.pickup_robot_mm[1]) * xy_pulses_per_mm)
            ),
            "place_x_pulse": int(
                round(float(self.place_robot_mm[0]) * xy_pulses_per_mm)
            ),
            "place_y_pulse": int(
                round(float(self.place_robot_mm[1]) * xy_pulses_per_mm)
            ),
            "rotation_deg": clockwise_rotation,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "piece_id": int(self.piece_id),
            "source_center_paper_mm": self.source_center_paper_mm.tolist(),
            "pickup_paper_mm": self.pickup_paper_mm.tolist(),
            "target_center_paper_mm": self.target_center_paper_mm.tolist(),
            "place_paper_mm": self.place_paper_mm.tolist(),
            "rotation_cw_deg": float(self.rotation_cw_deg),
            "pickup_robot_mm": self.pickup_robot_mm.tolist(),
            "place_robot_mm": self.place_robot_mm.tolist(),
            "rotation_robot_deg": float(self.rotation_robot_deg),
            "target_polygon_paper_mm": self.target_polygon_paper_mm.tolist(),
        }
