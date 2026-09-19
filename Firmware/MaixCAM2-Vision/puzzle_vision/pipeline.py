from __future__ import annotations

from dataclasses import dataclass, replace
import time as py_time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import VisionConfig
from .detector import (
    _complete_clipped_polygon_corner,
    _refit_polygon_edges,
    detect_paper_and_pieces,
    undistort_frame,
)
from .geometry import (
    AssemblySolution,
    MoveCommand,
    PaperObservation,
    PieceObservation,
    SolveError,
    apply_homography,
    matrix_angle_degrees,
    polygon_area,
    rigid_matrix,
    transform_points,
    wrap_degrees,
)
from .solver import solve_puzzle


def _has_printed_texture(
    paper_bgr: np.ndarray,
    pieces: List[PieceObservation],
    px_per_mm: float,
) -> bool:
    """Distinguish printed card faces from solid-colour geometry pieces."""
    lab = cv2.cvtColor(paper_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_u8 = lab.astype(np.uint8)
    detail = np.maximum.reduce(
        [
            cv2.morphologyEx(
                lab_u8[:, :, channel],
                cv2.MORPH_GRADIENT,
                np.ones((3, 3), dtype=np.uint8),
            )
            for channel in range(3)
        ]
    )
    erosion_size = max(3, int(round(1.2 * px_per_mm)))
    if erosion_size % 2 == 0:
        erosion_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)
    )
    strong_printed_pixels = 0
    inspected_pixels = 0
    textured_pieces = 0
    for piece in pieces:
        safe_mask = cv2.erode(piece.mask, kernel, iterations=1) > 0
        pixels = lab[safe_mask]
        if len(pixels) < 40:
            continue
        reference = np.median(pixels, axis=0)
        luma_delta = np.abs(pixels[:, 0] - reference[0])
        chroma_delta = np.linalg.norm(
            pixels[:, 1:3] - reference[1:3], axis=1
        )
        local_detail = detail[safe_mask]
        strong_printed = (
            ((luma_delta > 55.0) | (chroma_delta > 20.0))
            & (local_detail > 20)
        )
        piece_printed_pixels = int(np.count_nonzero(strong_printed))
        strong_printed_pixels += piece_printed_pixels
        if (
            piece_printed_pixels >= 20
            and piece_printed_pixels / len(pixels) >= 0.008
        ):
            textured_pieces += 1
        inspected_pixels += len(pixels)
    return (
        inspected_pixels >= 100
        and strong_printed_pixels >= 40
        and strong_printed_pixels / inspected_pixels >= 0.008
        and textured_pieces >= 2
    )


def _recovery_piece_hypothesis(
    pieces: List[PieceObservation],
    config: VisionConfig,
) -> List[PieceObservation]:
    """Recover virtual sharp corners hidden by rounded cuts or blur."""
    recovered: List[PieceObservation] = []
    changed_corners = 0
    completed_corners = 0
    for piece in pieces:
        measured_px = piece.polygon_mm * config.px_per_mm
        refitted_px = _refit_polygon_edges(
            piece.contour_px, measured_px, config
        )
        completed_px = _complete_clipped_polygon_corner(
            refitted_px, config
        )
        if completed_px is not None:
            recovered_px = completed_px
            completed_corners += 1
        else:
            recovered_px = refitted_px
            displacement_mm = (
                np.linalg.norm(refitted_px - measured_px, axis=1)
                / config.px_per_mm
            )
            changed_corners += int(
                np.count_nonzero(displacement_mm > 0.1)
            )
        recovered.append(
            replace(
                piece,
                polygon_mm=recovered_px / config.px_per_mm,
                area_mm2=(
                    piece.area_mm2
                    + (
                        polygon_area(recovered_px)
                        - polygon_area(measured_px)
                    )
                    / (config.px_per_mm * config.px_per_mm)
                ),
                boundary_reconstructed=bool(
                    piece.boundary_reconstructed
                    or completed_px is not None
                    or np.any(
                        np.linalg.norm(
                            refitted_px - measured_px, axis=1
                        )
                        > 0.1 * config.px_per_mm
                    )
                ),
            )
        )
    if changed_corners or completed_corners:
        print(
            "[vision] recovery refitted "
            f"{changed_corners} rounded/blurred corners and completed "
            f"{completed_corners} print-clipped corners from straight-edge "
            "support",
            flush=True,
        )
    return recovered


@dataclass
class PipelineResult:
    status: str
    robot_calibrated: bool
    paper: PaperObservation
    pieces: List[PieceObservation]
    solution: AssemblySolution
    commands: List[MoveCommand]
    debug_bgr: np.ndarray
    rectified_debug_bgr: np.ndarray
    foreground_mask: np.ndarray
    residual_image: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "robot_calibrated": bool(self.robot_calibrated),
            "paper": {
                "frame_quad_px": self.paper.frame_quad_px.tolist(),
                "frame_to_paper_px": self.paper.frame_to_paper_px.tolist(),
                "paper_px_to_frame": self.paper.paper_px_to_frame.tolist(),
                "divider_y_mm": float(self.paper.divider_y_mm),
                "size_mm": [
                    float(self.paper.width_mm),
                    float(self.paper.height_mm),
                ],
                "detection_score": float(self.paper.detection_score),
            },
            "pieces": [
                {
                    "piece_id": int(piece.piece_id),
                    "polygon_mm": piece.polygon_mm.tolist(),
                    "source_center_mm": piece.source_center_mm.tolist(),
                    "pickup_mm": piece.pickup_mm.tolist(),
                    "area_mm2": float(piece.area_mm2),
                    "approximation_error_mm": float(
                        piece.approximation_error_mm
                    ),
                }
                for piece in self.pieces
            ],
            "assembly": {
                "geometry_score": float(self.solution.geometry_score),
                "texture_score": float(self.solution.texture_score),
                "total_score": float(self.solution.total_score),
                "score_margin": (
                    None
                    if self.solution.score_margin is None
                    else float(self.solution.score_margin)
                ),
                "valid_candidate_count": int(
                    self.solution.valid_candidate_count
                ),
                "search_complete": bool(self.solution.search_complete),
                "preserve_center_symmetry": bool(
                    self.solution.preserve_center_symmetry
                ),
                "trained_template_name": (
                    self.solution.trained_template_name
                ),
                "placement_spread_scale": float(
                    self.solution.placement_spread_scale
                ),
                "center_symmetric_piece_pair": (
                    None
                    if self.solution.center_symmetric_piece_pair is None
                    else list(
                        self.solution.center_symmetric_piece_pair
                    )
                ),
                "explored_nodes": int(self.solution.explored_nodes),
                "rectangle_size_mm": list(
                    self.solution.metrics.rectangle_size_mm
                ),
                "fill_ratio": float(self.solution.metrics.fill_ratio),
                "overlap_ratio": float(self.solution.metrics.overlap_ratio),
                "hole_ratio": float(self.solution.metrics.hole_ratio),
                "boundary_p95_mm": float(
                    self.solution.metrics.boundary_p95_mm
                ),
                "boundary_p99_mm": float(
                    self.solution.metrics.boundary_p99_mm
                ),
                "side_coverage_min": float(
                    self.solution.metrics.side_coverage_min
                ),
                "convexity_ratio": float(
                    self.solution.metrics.convexity_ratio
                ),
                "corner_error_max_mm": float(
                    self.solution.metrics.corner_error_max_mm
                ),
            },
            "commands": [command.to_dict() for command in self.commands],
        }


def _translation(vector: np.ndarray) -> np.ndarray:
    return rigid_matrix(0.0, np.asarray(vector, dtype=np.float64))


def _long_edge_angle(rectangle_box: np.ndarray) -> float:
    box = np.asarray(rectangle_box, dtype=np.float64)
    best_vector = None
    best_length = -1.0
    for index in range(4):
        vector = box[(index + 1) % 4] - box[index]
        length = float(np.linalg.norm(vector))
        if length > best_length:
            best_length = length
            best_vector = vector
    if best_vector is None:
        raise ValueError("rectangle box has no edges")
    return float(np.arctan2(best_vector[1], best_vector[0]))


def _target_global_transform(
    paper: PaperObservation,
    pieces: List[PieceObservation],
    solution: AssemblySolution,
    config: VisionConfig,
) -> np.ndarray:
    center = solution.metrics.rectangle_center_mm
    current_long_angle = _long_edge_angle(
        solution.metrics.rectangle_box_mm
    )
    safe_x_min = config.target_paper_edge_margin_mm
    safe_x_max = paper.width_mm - config.target_paper_edge_margin_mm
    safe_y_min = paper.divider_y_mm + config.target_divider_margin_mm
    safe_y_max = paper.height_mm - config.target_paper_edge_margin_mm
    target_center = np.array(
        [0.5 * (safe_x_min + safe_x_max), 0.5 * (safe_y_min + safe_y_max)],
        dtype=np.float64,
    )

    candidates: List[Tuple[float, np.ndarray]] = []
    for target_angle in (0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi):
        rotation = target_angle - current_long_angle
        global_transform = (
            _translation(target_center)
            @ rigid_matrix(rotation, np.zeros(2))
            @ _translation(-center)
        )
        all_target_points = []
        total_cost = 0.0
        for piece in pieces:
            transform = global_transform @ solution.transforms[piece.piece_id]
            target_polygon = transform_points(piece.polygon_mm, transform)
            all_target_points.append(target_polygon)
            place = transform_points(
                piece.pickup_mm.reshape(1, 2), transform
            )[0]
            total_cost += float(np.linalg.norm(place - piece.pickup_mm))
            total_cost += 0.12 * abs(matrix_angle_degrees(transform))
        target_points = np.vstack(all_target_points)
        fits = (
            float(target_points[:, 0].min()) >= safe_x_min
            and float(target_points[:, 0].max()) <= safe_x_max
            and float(target_points[:, 1].min()) >= safe_y_min
            and float(target_points[:, 1].max()) <= safe_y_max
        )
        if fits:
            candidates.append((total_cost, global_transform))

    if not candidates:
        raise ValueError(
            "solved rectangle does not fit in the configured lower-half safe area"
        )
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _paper_points_to_robot(
    points_mm: np.ndarray,
    paper: PaperObservation,
    frame_to_robot: np.ndarray,
) -> np.ndarray:
    paper_px = np.asarray(points_mm, dtype=np.float64).reshape(-1, 2)
    paper_px = paper_px * paper.px_per_mm
    frame_px = apply_homography(paper_px, paper.paper_px_to_frame)
    return apply_homography(frame_px, frame_to_robot)


def _robot_rotation(
    source_mm: np.ndarray,
    transform: np.ndarray,
    paper: PaperObservation,
    frame_to_robot: np.ndarray,
) -> float:
    source_tip = source_mm + np.array([1.0, 0.0])
    target = transform_points(source_mm.reshape(1, 2), transform)[0]
    target_tip = transform_points(source_tip.reshape(1, 2), transform)[0]
    source_robot = _paper_points_to_robot(
        np.vstack([source_mm, source_tip]), paper, frame_to_robot
    )
    target_robot = _paper_points_to_robot(
        np.vstack([target, target_tip]), paper, frame_to_robot
    )
    source_angle = np.arctan2(
        source_robot[1, 1] - source_robot[0, 1],
        source_robot[1, 0] - source_robot[0, 0],
    )
    target_angle = np.arctan2(
        target_robot[1, 1] - target_robot[0, 1],
        target_robot[1, 0] - target_robot[0, 0],
    )
    return wrap_degrees(np.degrees(target_angle - source_angle))


def _build_commands(
    paper: PaperObservation,
    pieces: List[PieceObservation],
    solution: AssemblySolution,
    config: VisionConfig,
) -> List[MoveCommand]:
    global_transform = _target_global_transform(
        paper, pieces, solution, config
    )
    assembly_center = transform_points(
        solution.metrics.rectangle_center_mm.reshape(1, 2),
        global_transform,
    )[0]
    frame_to_robot = np.asarray(config.frame_to_robot, dtype=np.float64)
    median_piece_area = float(
        np.median([piece.area_mm2 for piece in pieces])
    )
    nominal_spread_mm = (
        config.placement_spread_mm
        * solution.placement_spread_scale
    )
    final_transforms = {
        piece.piece_id: (
            global_transform @ solution.transforms[piece.piece_id]
        )
        for piece in pieces
    }
    target_centers = {
        piece.piece_id: transform_points(
            piece.source_center_mm.reshape(1, 2),
            final_transforms[piece.piece_id],
        )[0]
        for piece in pieces
    }

    def adaptive_spread_mm(piece: PieceObservation) -> float:
        area_scale = np.sqrt(
            median_piece_area / max(piece.area_mm2, 1e-6)
        )
        return float(
            np.clip(
                nominal_spread_mm * area_scale,
                0.8 * nominal_spread_mm,
                1.6 * nominal_spread_mm,
            )
        )

    symmetric_offsets: Dict[int, np.ndarray] = {}
    symmetric_pair = solution.center_symmetric_piece_pair
    pieces_by_id = {piece.piece_id: piece for piece in pieces}
    if (
        solution.preserve_center_symmetry
        and symmetric_pair is not None
        and config.placement_spread_mm > 0.0
        and all(piece_id in pieces_by_id for piece_id in symmetric_pair)
    ):
        rank_piece_id, opposite_piece_id = symmetric_pair
        pair_direction = (
            target_centers[rank_piece_id]
            - target_centers[opposite_piece_id]
        )
        pair_length = float(np.linalg.norm(pair_direction))
        if pair_length > 1e-9:
            pair_spread_mm = max(
                adaptive_spread_mm(pieces_by_id[rank_piece_id]),
                adaptive_spread_mm(
                    pieces_by_id[opposite_piece_id]
                ),
            )
            pair_offset = (
                pair_spread_mm * pair_direction / pair_length
            )
            symmetric_offsets[rank_piece_id] = pair_offset
            symmetric_offsets[opposite_piece_id] = -pair_offset

    commands: List[MoveCommand] = []
    for piece in pieces:
        final_transform = final_transforms[piece.piece_id]
        target_center = target_centers[piece.piece_id].copy()
        place = transform_points(
            piece.pickup_mm.reshape(1, 2), final_transform
        )[0]
        target_polygon = transform_points(
            piece.polygon_mm, final_transform
        )
        spread_direction = target_center - assembly_center
        spread_length = float(np.linalg.norm(spread_direction))
        spread_offset = symmetric_offsets.get(piece.piece_id)
        can_use_adaptive_spread = (
            not solution.preserve_center_symmetry
            or symmetric_pair is not None
        )
        if (
            spread_offset is None
            and can_use_adaptive_spread
            and config.placement_spread_mm > 0.0
            and spread_length > 1e-9
        ):
            # Small fragments have a larger relative pickup/pose error and
            # receive more clearance; large fragments are more stable and
            # stay closer to the solved assembly.  Equal-area cuts retain the
            # configured nominal spread exactly.
            piece_spread_mm = adaptive_spread_mm(piece)
            spread_offset = (
                piece_spread_mm
                * spread_direction
                / spread_length
            )
        if spread_offset is not None:
            target_center = target_center + spread_offset
            place = place + spread_offset
            target_polygon = target_polygon + spread_offset
        pickup_robot = _paper_points_to_robot(
            piece.pickup_mm.reshape(1, 2), paper, frame_to_robot
        )[0]
        place_robot = _paper_points_to_robot(
            place.reshape(1, 2), paper, frame_to_robot
        )[0]
        paper_rotation = matrix_angle_degrees(final_transform)
        robot_rotation = _robot_rotation(
            piece.pickup_mm, final_transform, paper, frame_to_robot
        )
        robot_rotation = wrap_degrees(
            config.robot_theta_sign * robot_rotation
        )
        commands.append(
            MoveCommand(
                piece_id=piece.piece_id,
                source_center_paper_mm=piece.source_center_mm.copy(),
                pickup_paper_mm=piece.pickup_mm.copy(),
                target_center_paper_mm=target_center,
                place_paper_mm=place,
                rotation_cw_deg=paper_rotation,
                pickup_robot_mm=pickup_robot,
                place_robot_mm=place_robot,
                rotation_robot_deg=robot_rotation,
                target_polygon_paper_mm=target_polygon,
            )
        )
    area_by_id = {
        piece.piece_id: piece.area_mm2 for piece in pieces
    }
    # Place large, mechanically stable fragments first.  Putting a large
    # piece down after a small one is more likely to sweep or nudge the small
    # piece; the final small fragments can safely enter the remaining gaps.
    commands.sort(
        key=lambda command: (
            -float(area_by_id[command.piece_id]),
            int(command.piece_id),
        )
    )
    return commands


def _debug_visualization(
    paper: PaperObservation,
    pieces: List[PieceObservation],
    solution: AssemblySolution,
    commands: List[MoveCommand],
    status: str,
) -> np.ndarray:
    image = paper.image_bgr.copy()
    scale = paper.px_per_mm
    divider_y = int(round(paper.divider_y_px))
    cv2.line(
        image,
        (0, divider_y),
        (image.shape[1] - 1, divider_y),
        (0, 255, 255),
        2,
    )
    colors = [
        (255, 90, 40),
        (50, 220, 80),
        (60, 100, 255),
        (230, 80, 220),
    ]
    commands_by_id = {command.piece_id: command for command in commands}
    for piece in pieces:
        color = colors[piece.piece_id % len(colors)]
        source = np.round(piece.polygon_mm * scale).astype(np.int32)
        cv2.polylines(image, [source], True, color, 3, cv2.LINE_AA)
        pickup = tuple(np.round(piece.pickup_mm * scale).astype(int))
        cv2.circle(image, pickup, 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"S{piece.piece_id}",
            (pickup[0] + 7, pickup[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        command = commands_by_id[piece.piece_id]
        target = np.round(
            command.target_polygon_paper_mm * scale
        ).astype(np.int32)
        cv2.polylines(image, [target], True, color, 3, cv2.LINE_AA)
        place = tuple(np.round(command.place_paper_mm * scale).astype(int))
        cv2.drawMarker(
            image,
            place,
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )
        cv2.putText(
            image,
            f"T{piece.piece_id} {command.rotation_cw_deg:+.1f}deg",
            (place[0] + 7, place[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 34), (20, 20, 20), -1)
    cv2.putText(
        image,
        (
            f"{status} pieces={len(pieces)} "
            f"fill={solution.metrics.fill_ratio:.3f} "
            f"side={solution.metrics.side_coverage_min:.2f} "
            f"score={solution.total_score:.3f}"
        ),
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _frame_debug_visualization(
    frame_bgr: np.ndarray,
    paper: PaperObservation,
    pieces: List[PieceObservation],
    solution: AssemblySolution,
    commands: List[MoveCommand],
    status: str,
    config: VisionConfig,
) -> np.ndarray:
    """Draw source and solved target poses on the original camera view."""
    image = undistort_frame(frame_bgr, config).copy()
    colors = [
        (255, 90, 40),
        (50, 220, 80),
        (60, 100, 255),
        (230, 80, 220),
    ]

    def to_frame(points_mm: np.ndarray) -> np.ndarray:
        paper_px = (
            np.asarray(points_mm, dtype=np.float64).reshape(-1, 2)
            * paper.px_per_mm
        )
        frame_px = apply_homography(
            paper_px, paper.paper_px_to_frame
        )
        return np.round(frame_px).astype(np.int32)

    paper_quad = np.round(paper.frame_quad_px).astype(np.int32)
    cv2.polylines(
        image, [paper_quad], True, (0, 210, 255), 2, cv2.LINE_AA
    )
    divider = to_frame(
        np.array(
            [
                [0.0, paper.divider_y_mm],
                [paper.width_mm, paper.divider_y_mm],
            ],
            dtype=np.float64,
        )
    )
    cv2.line(
        image,
        tuple(divider[0]),
        tuple(divider[1]),
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    target_overlay = image.copy()
    for command in commands:
        color = colors[command.piece_id % len(colors)]
        target = to_frame(command.target_polygon_paper_mm)
        cv2.fillPoly(target_overlay, [target], color, cv2.LINE_AA)
    image = cv2.addWeighted(
        target_overlay, 0.18, image, 0.82, 0.0
    )

    commands_by_id = {
        command.piece_id: command for command in commands
    }
    for piece in pieces:
        command = commands_by_id[piece.piece_id]
        color = colors[piece.piece_id % len(colors)]
        source = to_frame(piece.polygon_mm)
        target = to_frame(command.target_polygon_paper_mm)
        pickup = tuple(to_frame(piece.pickup_mm)[0])
        place = tuple(to_frame(command.place_paper_mm)[0])

        cv2.polylines(
            image, [source], True, color, 3, cv2.LINE_AA
        )
        cv2.polylines(
            image, [target], True, color, 3, cv2.LINE_AA
        )
        cv2.circle(image, pickup, 6, color, -1, cv2.LINE_AA)
        cv2.drawMarker(
            image,
            place,
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=3,
        )
        cv2.arrowedLine(
            image,
            pickup,
            place,
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.04,
        )
        cv2.putText(
            image,
            f"S{piece.piece_id}",
            (pickup[0] + 8, pickup[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"T{piece.piece_id} {command.rotation_cw_deg:+.1f}deg",
            (place[0] + 8, place[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    banner_height = max(38, int(round(image.shape[0] * 0.055)))
    banner = image.copy()
    cv2.rectangle(
        banner,
        (0, 0),
        (image.shape[1] - 1, banner_height),
        (20, 20, 20),
        -1,
    )
    image = cv2.addWeighted(banner, 0.82, image, 0.18, 0.0)
    cv2.putText(
        image,
        (
            f"{status} pieces={len(pieces)} "
            f"fill={solution.metrics.fill_ratio:.3f} "
            f"side={solution.metrics.side_coverage_min:.2f}"
        ),
        (12, int(round(banner_height * 0.72))),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, min(0.9, image.shape[1] / 1500.0)),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _frame_detection_visualization(
    frame_bgr: np.ndarray,
    paper: PaperObservation,
    pieces: List[PieceObservation],
    config: VisionConfig,
) -> np.ndarray:
    """Draw measured source poses on the camera view before solving."""
    image = undistort_frame(frame_bgr, config).copy()
    colors = [
        (255, 90, 40),
        (50, 220, 80),
        (60, 100, 255),
        (230, 80, 220),
    ]

    def to_frame(points_mm: np.ndarray) -> np.ndarray:
        paper_px = (
            np.asarray(points_mm, dtype=np.float64).reshape(-1, 2)
            * paper.px_per_mm
        )
        return np.round(
            apply_homography(paper_px, paper.paper_px_to_frame)
        ).astype(np.int32)

    paper_quad = np.round(paper.frame_quad_px).astype(np.int32)
    cv2.polylines(
        image, [paper_quad], True, (0, 210, 255), 2, cv2.LINE_AA
    )
    divider = to_frame(
        np.array(
            [
                [0.0, paper.divider_y_mm],
                [paper.width_mm, paper.divider_y_mm],
            ],
            dtype=np.float64,
        )
    )
    cv2.line(
        image,
        tuple(divider[0]),
        tuple(divider[1]),
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for piece in pieces:
        color = colors[piece.piece_id % len(colors)]
        source = to_frame(piece.polygon_mm)
        pickup = tuple(to_frame(piece.pickup_mm)[0])
        cv2.polylines(
            image, [source], True, color, 3, cv2.LINE_AA
        )
        cv2.circle(image, pickup, 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"P{piece.piece_id} V={len(piece.polygon_mm)}",
            (pickup[0] + 8, pickup[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return image


def _detection_visualization(
    paper: PaperObservation,
    pieces: List[PieceObservation],
) -> np.ndarray:
    image = paper.image_bgr.copy()
    scale = paper.px_per_mm
    divider_y = int(round(paper.divider_y_px))
    cv2.line(
        image,
        (0, divider_y),
        (image.shape[1] - 1, divider_y),
        (0, 255, 255),
        2,
    )
    colors = [
        (255, 90, 40),
        (50, 220, 80),
        (60, 100, 255),
        (230, 80, 220),
    ]
    for piece in pieces:
        color = colors[piece.piece_id % len(colors)]
        polygon = np.round(
            piece.polygon_mm * scale
        ).astype(np.int32)
        cv2.polylines(
            image, [polygon], True, color, 3, cv2.LINE_AA
        )
        x, y, _, _ = cv2.boundingRect(polygon)
        pickup = tuple(
            np.round(piece.pickup_mm * scale).astype(int)
        )
        cv2.circle(image, pickup, 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            (
                f"P{piece.piece_id} "
                f"A={piece.area_mm2:.0f} "
                f"V={len(piece.polygon_mm)}"
            ),
            (x + 4, max(22, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            color,
            2,
            cv2.LINE_AA,
        )

    banner_top = max(0, image.shape[0] - 34)
    cv2.rectangle(
        image,
        (0, banner_top),
        (image.shape[1] - 1, image.shape[0] - 1),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        image,
        f"DETECTED pieces={len(pieces)}",
        (8, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _solve_with_recovery(
    pieces: List[PieceObservation],
    paper_bgr: np.ndarray,
    config: VisionConfig,
    use_texture: bool,
    template_pieces: Optional[List[PieceObservation]] = None,
) -> AssemblySolution:
    """Run two distinct bounded schedules under one total wall-clock budget."""
    deadline = (
        py_time.monotonic()
        + config.solver_total_time_limit_ms / 1000.0
    )

    def next_attempt_config() -> Optional[VisionConfig]:
        remaining_ms = int(
            max(0.0, deadline - py_time.monotonic()) * 1000.0
        )
        if remaining_ms <= 0:
            return None
        attempt_limit_ms = config.solver_time_limit_ms
        if attempt_limit_ms <= 0:
            attempt_limit_ms = remaining_ms
        return replace(
            config,
            solver_time_limit_ms=max(
                1, min(attempt_limit_ms, remaining_ms)
            ),
        )

    primary_config = next_attempt_config()
    assert primary_config is not None
    primary_solution: Optional[AssemblySolution] = None
    primary_error: Optional[SolveError] = None
    try:
        primary_solution = solve_puzzle(
            pieces,
            paper_bgr,
            primary_config,
            use_texture=use_texture,
            recovery_mode=False,
            template_pieces=template_pieces,
        )
    except SolveError as error:
        primary_error = error

    if (
        primary_solution is not None
        and primary_solution.search_complete
    ):
        return primary_solution

    # Recovery also expands three-piece split-edge/T-junction coverage.  The
    # primary bounded queue can otherwise miss the only near-rectangular card
    # layout even though all three contours were measured correctly.
    retry_supported = (
        len(pieces) in (3, 4) and config.solver_mode != "figure2"
    )
    retry_config = (
        next_attempt_config() if retry_supported else None
    )
    if retry_config is not None and use_texture:
        # Attempt 1 deliberately accepts broader 4 mm seam collinearity for
        # rough hand cuts.  Attempt 2 must expose a genuinely different
        # candidate ordering: rich J/Q/K/Joker artwork is measured cleanly
        # enough that a tighter seam tolerance preserves the true mates.
        retry_config = replace(
            retry_config,
            solver_collinear_tolerance_mm=min(
                retry_config.solver_collinear_tolerance_mm,
                2.5,
            ),
        )
    if retry_config is None:
        if primary_solution is not None:
            return primary_solution
        assert primary_error is not None
        raise primary_error

    reason = (
        "primary search was incomplete"
        if primary_solution is not None
        else f"primary search failed: {primary_error}"
    )
    print(
        "[vision] "
        f"{reason}; starting recovery search with "
        f"{retry_config.solver_time_limit_ms} ms remaining budget",
        flush=True,
    )
    try:
        recovery_pieces = (
            _recovery_piece_hypothesis(pieces, retry_config)
            if use_texture
            else pieces
        )
        recovery_solution = solve_puzzle(
            recovery_pieces,
            paper_bgr,
            retry_config,
            use_texture=use_texture,
            recovery_mode=True,
        )
    except SolveError as recovery_error:
        if primary_solution is not None:
            print(
                "[vision] recovery search failed; keeping the primary "
                f"SEARCH_LIMIT candidate: {recovery_error}",
                flush=True,
            )
            return primary_solution
        assert primary_error is not None
        raise SolveError(
            "both solve attempts failed; "
            f"primary: {primary_error}; recovery: {recovery_error}"
        ) from recovery_error

    if (
        primary_solution is not None
        and not recovery_solution.search_complete
        and primary_solution.total_score
        < recovery_solution.total_score
    ):
        return primary_solution
    return recovery_solution


class PuzzleVisionPipeline:
    def __init__(self, config: VisionConfig):
        self.config = config
        self.last_detection_debug_bgr: Optional[np.ndarray] = None
        self.last_frame_detection_debug_bgr: Optional[np.ndarray] = None

    def process(
        self,
        frame_bgr: np.ndarray,
        use_texture: bool = True,
        detection_callback: Optional[
            Callable[[np.ndarray], None]
        ] = None,
    ) -> PipelineResult:
        detect_started = py_time.monotonic()
        print("[vision] detecting paper and pieces...", flush=True)
        paper, pieces, foreground_mask, residual = (
            detect_paper_and_pieces(frame_bgr, self.config)
        )
        detect_ms = int(
            round((py_time.monotonic() - detect_started) * 1000.0)
        )
        print(
            f"[vision] detected {len(pieces)} pieces in {detect_ms} ms; "
            "solving puzzle...",
            flush=True,
        )
        self.last_detection_debug_bgr = _detection_visualization(
            paper, pieces
        )
        self.last_frame_detection_debug_bgr = (
            _frame_detection_visualization(
                frame_bgr, paper, pieces, self.config
            )
        )
        if detection_callback is not None:
            detection_callback(self.last_detection_debug_bgr)
        solve_started = py_time.monotonic()
        solve_config = self.config
        effective_texture = use_texture and _has_printed_texture(
            paper.image_bgr,
            pieces,
            self.config.px_per_mm,
        )
        if use_texture and not effective_texture:
            print(
                "[vision] solid-colour pieces detected; texture scoring skipped",
                flush=True,
            )
        if effective_texture:
            # Printed-card contours are less stable than plain white pieces:
            # artwork may share the A4 colour and hand-cut edges can move by
            # several millimetres.  Texture continuity supplies an additional
            # disambiguation signal, so use a measurement budget that remains
            # below the problem's 20 mm corresponding-vertex allowance.
            solve_config = replace(
                self.config,
                solver_collinear_tolerance_mm=max(
                    self.config.solver_collinear_tolerance_mm, 4.0
                ),
                solver_max_overlap_mm2=max(
                    self.config.solver_max_overlap_mm2, 60.0
                ),
                min_rectangle_fill_ratio=min(
                    self.config.min_rectangle_fill_ratio, 0.90
                ),
                max_overlap_ratio=max(
                    self.config.max_overlap_ratio, 0.025
                ),
                max_hole_ratio=max(
                    self.config.max_hole_ratio, 0.025
                ),
                max_boundary_p95_mm=max(
                    self.config.max_boundary_p95_mm, 8.0
                ),
                max_boundary_p99_mm=max(
                    self.config.max_boundary_p99_mm, 10.0
                ),
                min_rectangle_side_coverage=min(
                    self.config.min_rectangle_side_coverage, 0.84
                ),
                min_rectangle_convexity_ratio=min(
                    self.config.min_rectangle_convexity_ratio, 0.94
                ),
                max_rectangle_corner_error_mm=max(
                    self.config.max_rectangle_corner_error_mm, 6.0
                ),
            )
            print(
                "[vision] printed-card mode: using <=10 mm measured-boundary "
                "tolerance",
                flush=True,
            )
        solver_pieces = (
            [
                replace(
                    piece,
                    polygon_mm=np.round(piece.polygon_mm),
                )
                for piece in pieces
            ]
            if effective_texture
            else pieces
        )
        solution = _solve_with_recovery(
            solver_pieces,
            paper.image_bgr,
            solve_config,
            effective_texture,
            template_pieces=pieces,
        )
        solve_ms = int(
            round((py_time.monotonic() - solve_started) * 1000.0)
        )
        print(f"[vision] puzzle solved in {solve_ms} ms", flush=True)
        commands = _build_commands(
            paper, pieces, solution, self.config
        )
        status = "OK"
        if not solution.search_complete:
            status = "SEARCH_LIMIT"
        elif (
            solution.score_margin is not None
            and solution.score_margin < self.config.ambiguity_margin
        ):
            status = "LOW_MARGIN"
        rectified_debug = _debug_visualization(
            paper, pieces, solution, commands, status
        )
        debug = _frame_debug_visualization(
            frame_bgr,
            paper,
            pieces,
            solution,
            commands,
            status,
            self.config,
        )
        return PipelineResult(
            status=status,
            robot_calibrated=self.config.robot_calibrated,
            paper=paper,
            pieces=pieces,
            solution=solution,
            commands=commands,
            debug_bgr=debug,
            rectified_debug_bgr=rectified_debug,
            foreground_mask=foreground_mask,
            residual_image=residual,
        )
