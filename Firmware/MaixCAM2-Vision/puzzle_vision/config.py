from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class VisionConfig:
    # MaixCAM2 capture.
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    capture_frames: int = 5

    # A4 metric rectification.
    paper_short_mm: float = 210.0
    paper_long_mm: float = 297.0
    px_per_mm: float = 3.0
    # Optional undistorted frame-pixel corners for a fixed camera/paper rig.
    # Order is top-left, top-right, bottom-right, bottom-left.  When present,
    # paper contour and divider searches are bypassed.
    fixed_paper_quad_px: Optional[List[List[float]]] = None
    # Keep the hard gate permissive: a complete A4 sheet can be only 5%--10%
    # of a 16:9 camera frame when the camera is mounted high.
    paper_min_area_ratio: float = 0.04
    paper_max_area_ratio: float = 0.96
    # Apparent side-length ratio is not projectively invariant.  The divider
    # score remains the stronger A4 discriminator under an oblique view.
    paper_aspect_tolerance: float = 0.55
    divider_search_min: float = 0.35
    divider_search_max: float = 0.65
    divider_min_coverage: float = 0.42
    divider_exclusion_mm: float = 4.0
    orientation_min_area_ratio: float = 1.20

    # Piece segmentation. OpenCV Lab channel units are used for the threshold.
    foreground_min_lab_distance: float = 11.0
    foreground_mad_scale: float = 6.0
    paper_border_exclusion_mm: float = 2.0
    min_piece_area_mm2: float = 100.0
    # The official target can be as large as 90 mm x 120 mm and may consist
    # of a single piece, so the upper bound must exceed 10,800 mm².
    max_piece_area_mm2: float = 12000.0
    expected_piece_count: int = 0  # 0 means accept 1..4.
    polygon_epsilon_mm: float = 0.7
    min_polygon_edge_mm: float = 7.0
    max_polygon_vertices: int = 5

    # Geometry solver.
    # "figure2" uses the fixed four-piece contest template, "generic" uses
    # edge search, and "auto" tries the bounded template path before generic.
    solver_mode: str = "auto"
    solver_min_contact_mm: float = 8.0
    # Camera-extracted corners can move by about 2 mm after perspective
    # correction.  Three millimetres still sits well inside the official
    # 20 mm corresponding-vertex tolerance while retaining true seam pairs.
    solver_collinear_tolerance_mm: float = 3.0
    solver_max_overlap_mm2: float = 28.0
    solver_raster_px_per_mm: float = 2.0
    solver_max_nodes: int = 120000
    # Each search attempt is bounded independently.  The pipeline may run one
    # recovery attempt with a different schedule, while both attempts share
    # solver_total_time_limit_ms.
    # Zero lets one attempt use the remaining total budget.
    solver_time_limit_ms: int = 15000
    solver_total_time_limit_ms: int = 30000
    # C1/C2 mate filtering keeps at most a few candidates per source edge;
    # leave headroom for online third-piece T-junction anchors.
    solver_max_pair_candidates: int = 64
    solver_max_leaf_candidates: int = 256
    target_short_min_mm: float = 45.0
    target_short_max_mm: float = 95.0
    target_long_min_mm: float = 85.0
    target_long_max_mm: float = 125.0
    # Reject loose "almost rectangles" before the non-textured fast return.
    # Real camera runs stay above about 0.95, while wrong strip assemblies can
    # otherwise pass the old 0.90 gate at roughly 0.92.
    min_rectangle_fill_ratio: float = 0.93
    max_overlap_ratio: float = 0.018
    max_hole_ratio: float = 0.01
    max_boundary_p95_mm: float = 6.0
    max_boundary_p99_mm: float = 7.0
    # A high fill ratio alone can still describe a clipped-corner polygon or
    # a compact concave blob.  Require every side and corner of the fitted
    # rectangle to be supported by the measured outer silhouette.
    min_rectangle_side_coverage: float = 0.88
    min_rectangle_convexity_ratio: float = 0.96
    max_rectangle_corner_error_mm: float = 5.0
    texture_enabled: bool = True
    # Printed continuity must outweigh small contour errors from hand-cut
    # card stock; 0.20 separates the real seam in the photographed 5-heart
    # sample while geometry still supplies most of the total score.
    texture_weight: float = 0.20
    ambiguity_margin: float = 0.012

    # Final placement policy in the lower half of the A4 sheet.
    target_paper_edge_margin_mm: float = 10.0
    target_divider_margin_mm: float = 12.0
    placement_spread_mm: float = 5.0

    # Optional camera calibration. Empty means no lens undistortion.
    camera_matrix: Optional[List[List[float]]] = None
    distortion_coefficients: Optional[List[float]] = None

    # Undistorted camera-frame pixels to robot-mm planar transform.  Composing
    # this online with the detected paper pose keeps robot coordinates correct
    # if the A4 sheet is shifted or rotated.
    frame_to_robot: List[List[float]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    robot_calibrated: bool = False
    robot_theta_sign: float = 1.0
    robot_xy_pulses_per_mm: float = 80.0
    robot_x_max_pulse: int = 24000
    robot_y_max_pulse: int = 16700

    # Runtime integration.
    start_mode: str = "touch"  # "touch", "immediate", or legacy "func_key"
    uart_enabled: bool = False
    uart_device: str = "/dev/ttyS2"
    uart_baudrate: int = 115200
    save_debug_images: bool = True

    def validate(self) -> None:
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.camera_fps not in (30, 60):
            raise ValueError("camera_fps must be a MaixCAM2-supported 30 or 60")
        if self.capture_frames <= 0:
            raise ValueError("capture_frames must be positive")
        if self.px_per_mm <= 0:
            raise ValueError("px_per_mm must be positive")
        if self.fixed_paper_quad_px is not None:
            fixed_quad = np.asarray(
                self.fixed_paper_quad_px, dtype=np.float64
            )
            if fixed_quad.shape != (4, 2):
                raise ValueError(
                    "fixed_paper_quad_px must contain four [x, y] points"
                )
            if not np.all(np.isfinite(fixed_quad)):
                raise ValueError(
                    "fixed_paper_quad_px must contain finite coordinates"
                )
            fixed_area = 0.5 * abs(
                float(
                    np.dot(fixed_quad[:, 0], np.roll(fixed_quad[:, 1], -1))
                    - np.dot(
                        fixed_quad[:, 1],
                        np.roll(fixed_quad[:, 0], -1),
                    )
                )
            )
            if fixed_area < 1.0:
                raise ValueError(
                    "fixed_paper_quad_px must enclose a positive area"
                )
        if not 0.0 < self.divider_search_min < self.divider_search_max < 1.0:
            raise ValueError("divider search fractions must satisfy 0 < min < max < 1")
        if self.orientation_min_area_ratio <= 1.0:
            raise ValueError("orientation_min_area_ratio must be greater than 1")
        if self.expected_piece_count not in (0, 1, 2, 3, 4):
            raise ValueError("expected_piece_count must be 0..4")
        if self.solver_mode not in ("auto", "figure2", "generic"):
            raise ValueError(
                "solver_mode must be 'auto', 'figure2', or 'generic'"
            )
        if not 0 < self.min_piece_area_mm2 < self.max_piece_area_mm2:
            raise ValueError("piece area limits must be positive and ordered")
        if self.max_polygon_vertices < 3:
            raise ValueError("max_polygon_vertices must be at least 3")
        if (
            self.solver_max_nodes <= 0
            or self.solver_max_pair_candidates <= 0
            or self.solver_max_leaf_candidates <= 0
        ):
            raise ValueError("solver search limits must be positive")
        if self.solver_time_limit_ms < 0:
            raise ValueError(
                "solver_time_limit_ms must be zero or positive"
            )
        if self.solver_total_time_limit_ms <= 0:
            raise ValueError(
                "solver_total_time_limit_ms must be positive"
            )
        if (
            self.solver_time_limit_ms > 0
            and self.solver_time_limit_ms
            > self.solver_total_time_limit_ms
        ):
            raise ValueError(
                "solver_time_limit_ms must not exceed "
                "solver_total_time_limit_ms"
            )
        if not (
            0 < self.target_short_min_mm < self.target_short_max_mm
            and 0 < self.target_long_min_mm < self.target_long_max_mm
        ):
            raise ValueError("target size limits must be positive and ordered")
        if not (
            0.0 < self.min_rectangle_side_coverage <= 1.0
            and 0.0 < self.min_rectangle_convexity_ratio <= 1.0
            and self.max_rectangle_corner_error_mm > 0.0
        ):
            raise ValueError(
                "rectangle side/convexity limits must be in (0, 1] and "
                "corner error must be positive"
            )
        if (
            not np.isfinite(self.placement_spread_mm)
            or self.placement_spread_mm < 0.0
        ):
            raise ValueError("placement_spread_mm must be zero or positive")
        if self.start_mode not in ("touch", "immediate", "func_key"):
            raise ValueError(
                "start_mode must be touch, immediate, or func_key"
            )
        frame_to_robot = np.asarray(self.frame_to_robot, dtype=np.float64)
        if frame_to_robot.shape != (3, 3):
            raise ValueError("frame_to_robot must be a 3x3 matrix")
        if not np.all(np.isfinite(frame_to_robot)):
            raise ValueError("frame_to_robot must contain finite numbers")
        if abs(float(np.linalg.det(frame_to_robot))) < 1e-9:
            raise ValueError("frame_to_robot must be invertible")
        if self.robot_theta_sign not in (-1.0, 1.0):
            raise ValueError("robot_theta_sign must be -1 or 1")
        if (
            not np.isfinite(self.robot_xy_pulses_per_mm)
            or self.robot_xy_pulses_per_mm <= 0
        ):
            raise ValueError("robot_xy_pulses_per_mm must be positive")
        if self.robot_x_max_pulse <= 0 or self.robot_y_max_pulse <= 0:
            raise ValueError("robot pulse limits must be positive")
        if self.uart_baudrate <= 0:
            raise ValueError("uart_baudrate must be positive")
        if self.uart_enabled and self.uart_device != "/dev/ttyS2":
            raise ValueError(
                "this MaixCAM2 build maps U2T/U2R (B0/B1) only to "
                "/dev/ttyS2"
            )
        if self.uart_enabled and not self.robot_calibrated:
            raise ValueError(
                "set robot_calibrated=true only after frame-to-robot calibration"
            )
        if (self.camera_matrix is None) != (
            self.distortion_coefficients is None
        ):
            raise ValueError(
                "camera_matrix and distortion_coefficients must be set together"
            )
        if self.camera_matrix is not None:
            camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
            distortion = np.asarray(
                self.distortion_coefficients, dtype=np.float64
            )
            if camera_matrix.shape != (3, 3):
                raise ValueError("camera_matrix must be a 3x3 matrix")
            if distortion.size < 4:
                raise ValueError(
                    "distortion_coefficients must contain at least four values"
                )
            if not (
                np.all(np.isfinite(camera_matrix))
                and np.all(np.isfinite(distortion))
            ):
                raise ValueError("camera calibration must contain finite numbers")


def _update_known_fields(config: VisionConfig, values: Dict[str, Any]) -> None:
    known = set(asdict(config))
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError("unknown config fields: " + ", ".join(unknown))
    for name, value in values.items():
        setattr(config, name, value)


def load_config(path: Optional[str] = None) -> VisionConfig:
    config = VisionConfig()
    if path:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("configuration root must be a JSON object")
        _update_known_fields(config, values)
    config.validate()
    return config
