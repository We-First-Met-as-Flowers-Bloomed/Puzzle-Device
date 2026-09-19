from __future__ import annotations

import heapq
import itertools
import math
import time as py_time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import VisionConfig
from .geometry import (
    AssemblyMetrics,
    AssemblySolution,
    PieceObservation,
    SolveError,
    ensure_positive_winding,
    point_segment_distance,
    polygon_area,
    rigid_matrix,
    rotation_about,
    wrap_degrees,
)


@dataclass
class _Contact:
    piece_a: int
    edge_a: int
    piece_b: int
    edge_b: int
    start: np.ndarray
    end: np.ndarray
    length: float


@dataclass
class _Leaf:
    transforms: Dict[int, np.ndarray]
    metrics: AssemblyMetrics
    geometry_score: float
    texture_score: float = 0.0
    total_score: float = 0.0


@dataclass
class _CardStripState:
    piece_id: int
    flipped: bool
    transform: np.ndarray
    inverse: np.ndarray
    width_mm: float
    height_mm: float


def _leaves_are_measurement_equivalent(
    leaf_a: _Leaf,
    leaf_b: _Leaf,
    pieces_by_id: Dict[int, PieceObservation],
    tolerance_mm: float = 2.0,
) -> bool:
    """Merge pose variants caused only by sub-pixel endpoint choices."""
    if set(leaf_a.transforms) != set(leaf_b.transforms):
        return False
    for piece_id in leaf_a.transforms:
        polygon = pieces_by_id[piece_id].polygon_mm
        placed_a = _transform_rigid(polygon, leaf_a.transforms[piece_id])
        placed_b = _transform_rigid(polygon, leaf_b.transforms[piece_id])
        if float(np.max(np.linalg.norm(placed_a - placed_b, axis=1))) > tolerance_mm:
            return False
    return True


def _leaves_are_globally_equivalent(
    leaf_a: _Leaf,
    leaf_b: _Leaf,
    pieces_by_id: Dict[int, PieceObservation],
    anchor_id: int,
    tolerance_mm: float = 2.0,
) -> bool:
    """Merge identical assemblies expressed in different global poses."""
    alignment = (
        leaf_a.transforms[anchor_id]
        @ np.linalg.inv(leaf_b.transforms[anchor_id])
    )
    for piece_id, piece in pieces_by_id.items():
        placed_a = _transform_rigid(
            piece.polygon_mm, leaf_a.transforms[piece_id]
        )
        placed_b = _transform_rigid(
            piece.polygon_mm,
            alignment @ leaf_b.transforms[piece_id],
        )
        if (
            float(
                np.max(
                    np.linalg.norm(placed_a - placed_b, axis=1)
                )
            )
            > tolerance_mm
        ):
            return False
    return True


def _edges(polygon: np.ndarray) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    for index in range(len(polygon)):
        yield index, polygon[index], polygon[(index + 1) % len(polygon)]


def _transform_rigid(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply the solver's 2-D rigid transform without homography overhead."""
    pts = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    return pts @ transform[:2, :2].T + transform[:2, 2]


def _segment_length(start: np.ndarray, end: np.ndarray) -> float:
    return math.hypot(
        float(end[0] - start[0]), float(end[1] - start[1])
    )


def _collinear_overlap(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    tolerance: float,
    require_opposite: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    vector_a = a1 - a0
    vector_b = b1 - b0
    length_a = _segment_length(a0, a1)
    length_b = _segment_length(b0, b1)
    if length_a < 1e-6 or length_b < 1e-6:
        return None
    direction_a_x = float(vector_a[0]) / length_a
    direction_a_y = float(vector_a[1]) / length_a
    direction_b_x = float(vector_b[0]) / length_b
    direction_b_y = float(vector_b[1]) / length_b
    dot = direction_a_x * direction_b_x + direction_a_y * direction_b_y
    if require_opposite and dot > -0.985:
        return None
    if not require_opposite and abs(dot) < 0.985:
        return None
    delta_b0 = b0 - a0
    delta_b1 = b1 - a0
    distance_b0 = abs(
        direction_a_x * float(delta_b0[1])
        - direction_a_y * float(delta_b0[0])
    )
    distance_b1 = abs(
        direction_a_x * float(delta_b1[1])
        - direction_a_y * float(delta_b1[0])
    )
    if max(distance_b0, distance_b1) > tolerance:
        return None
    projection_b0 = (
        float(delta_b0[0]) * direction_a_x
        + float(delta_b0[1]) * direction_a_y
    )
    projection_b1 = (
        float(delta_b1[0]) * direction_a_x
        + float(delta_b1[1]) * direction_a_y
    )
    overlap_start = max(0.0, min(projection_b0, projection_b1))
    overlap_end = min(length_a, max(projection_b0, projection_b1))
    overlap = overlap_end - overlap_start
    if overlap <= 0:
        return None
    direction_a = np.array(
        [direction_a_x, direction_a_y], dtype=np.float64
    )
    start = a0 + direction_a * overlap_start
    end = a0 + direction_a * overlap_end
    return start, end, overlap


def _contacts_between(
    piece_a: int,
    polygon_a: np.ndarray,
    piece_b: int,
    polygon_b: np.ndarray,
    config: VisionConfig,
) -> List[_Contact]:
    contacts: List[_Contact] = []
    for edge_a, a0, a1 in _edges(polygon_a):
        for edge_b, b0, b1 in _edges(polygon_b):
            overlap = _collinear_overlap(
                a0,
                a1,
                b0,
                b1,
                config.solver_collinear_tolerance_mm,
                require_opposite=True,
            )
            if overlap is None:
                continue
            start, end, length = overlap
            if length >= config.solver_min_contact_mm:
                contacts.append(
                    _Contact(
                        piece_a=piece_a,
                        edge_a=edge_a,
                        piece_b=piece_b,
                        edge_b=edge_b,
                        start=start,
                        end=end,
                        length=length,
                    )
                )
    return contacts


def _raster_metrics(
    polygons: Sequence[np.ndarray], config: VisionConfig
) -> AssemblyMetrics:
    if not polygons:
        raise ValueError("at least one polygon is required")
    scale = config.solver_raster_px_per_mm
    all_points = np.vstack(polygons)
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    margin_mm = 5.0
    size = np.ceil((maximum - minimum + 2.0 * margin_mm) * scale).astype(int) + 3
    if np.any(size <= 0) or np.any(size > 700):
        raise SolveError("candidate assembly exceeds solver canvas")
    offset = -minimum + margin_mm

    accumulation = np.zeros((int(size[1]), int(size[0])), dtype=np.uint16)
    for polygon in polygons:
        points = np.round((polygon + offset) * scale).astype(np.int32)
        layer = np.zeros(accumulation.shape, dtype=np.uint8)
        cv2.fillPoly(layer, [points], 1)
        accumulation += layer.astype(np.uint16)

    union = (accumulation > 0).astype(np.uint8)
    # A sub-millimetre mate gap can remain one raster pixel wide and stay
    # connected to the outside.  Without closing that sampling crack,
    # findContours follows an internal seam and reports it as outer boundary.
    silhouette = cv2.morphologyEx(
        union,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    sum_area = sum(polygon_area(polygon) for polygon in polygons)
    overlap_area = 0.0
    for index, polygon_a in enumerate(polygons):
        for polygon_b in polygons[index + 1 :]:
            overlap_area += _polygon_overlap_area(polygon_a, polygon_b)
    # Polygon areas are metric and do not include the one-pixel perimeter
    # inflation of fillPoly.  Pairwise subtraction is exact for the normal
    # non-triple-overlap candidates and conservative for invalid triple ones.
    union_area = max(0.0, sum_area - overlap_area)
    overlap_ratio = overlap_area / max(sum_area, 1e-6)

    component_count, _ = cv2.connectedComponents(union)
    connected_components = max(0, component_count - 1)

    contours, hierarchy = cv2.findContours(
        silhouette, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise SolveError("candidate union has no contour")
    outer_indices = []
    hole_area_px = 0.0
    if hierarchy is not None:
        for index, relation in enumerate(hierarchy[0]):
            if relation[3] < 0:
                outer_indices.append(index)
            else:
                hole_area_px += abs(float(cv2.contourArea(contours[index])))
    if not outer_indices:
        outer_indices = [int(np.argmax([cv2.contourArea(c) for c in contours]))]
    outer = max(outer_indices, key=lambda idx: cv2.contourArea(contours[idx]))
    outer_points_px = contours[outer].reshape(-1, 2).astype(np.float32)

    rect = cv2.minAreaRect(outer_points_px)
    rect_center_px = np.array(rect[0], dtype=np.float64)
    rect_size_px = np.array(rect[1], dtype=np.float64)
    if np.any(rect_size_px < 1e-6):
        raise SolveError("candidate rectangle is degenerate")
    box_px = cv2.boxPoints(rect).astype(np.float64)
    rectangle_area = float(rect_size_px[0] * rect_size_px[1]) / (scale * scale)
    fill_ratio = union_area / max(rectangle_area, 1e-6)
    hole_area = hole_area_px / (scale * scale)
    hole_ratio = hole_area / max(rectangle_area, 1e-6)

    box_mm = box_px / scale - offset
    outer_mm = outer_points_px.astype(np.float64) / scale - offset
    distances = []
    for index in range(4):
        distances.append(
            point_segment_distance(
                outer_mm, box_mm[index], box_mm[(index + 1) % 4]
            )
        )
    boundary_distance = np.min(np.stack(distances, axis=1), axis=1)
    boundary_p95 = float(np.percentile(boundary_distance, 95))
    boundary_p99 = float(np.percentile(boundary_distance, 99))

    # A genuine rectangle needs evidence on all four sides, not merely a high
    # union-area/minAreaRect ratio.  Raster contour samples are spaced by at
    # most one solver pixel, so occupancy along each fitted side provides a
    # stable straight-side coverage measurement even for a rotated rectangle.
    side_band_mm = max(
        2.0, float(config.solver_collinear_tolerance_mm)
    )
    side_coverages = []
    for index in range(4):
        start = box_mm[index]
        end = box_mm[(index + 1) % 4]
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 1e-6:
            side_coverages.append(0.0)
            continue
        unit = vector / length
        relative = outer_mm - start
        projection = relative @ unit
        line_distance = np.abs(
            vector[0] * relative[:, 1]
            - vector[1] * relative[:, 0]
        ) / length
        near_side = (
            (projection >= -0.5)
            & (projection <= length + 0.5)
            & (line_distance <= side_band_mm)
        )
        bin_count = max(2, int(np.ceil(length * scale)) + 1)
        occupied = np.zeros(bin_count, dtype=np.uint8)
        if np.any(near_side):
            positions = np.clip(
                np.round(projection[near_side] * scale).astype(np.int32),
                0,
                bin_count - 1,
            )
            occupied[positions] = 1
            # Cover the harmless one-bin gaps produced by a diagonal raster
            # staircase without bridging a real missing side segment.
            occupied = cv2.dilate(
                occupied.reshape(1, -1),
                np.ones((1, 3), dtype=np.uint8),
                iterations=1,
            ).reshape(-1)
        side_coverages.append(float(np.mean(occupied)))
    side_coverage_min = float(min(side_coverages))

    corner_errors = [
        float(np.min(np.linalg.norm(outer_mm - corner, axis=1)))
        for corner in box_mm
    ]
    corner_error_max = float(max(corner_errors))

    outer_area_px = abs(float(cv2.contourArea(contours[outer])))
    hull = cv2.convexHull(outer_points_px)
    hull_area_px = abs(float(cv2.contourArea(hull)))
    convexity_ratio = outer_area_px / max(hull_area_px, 1e-6)

    size_mm = tuple(sorted((rect_size_px / scale).tolist()))
    return AssemblyMetrics(
        fill_ratio=float(fill_ratio),
        overlap_ratio=float(overlap_ratio),
        hole_ratio=float(hole_ratio),
        boundary_p95_mm=boundary_p95,
        boundary_p99_mm=boundary_p99,
        side_coverage_min=side_coverage_min,
        convexity_ratio=float(convexity_ratio),
        corner_error_max_mm=corner_error_max,
        rectangle_center_mm=rect_center_px / scale - offset,
        rectangle_size_mm=(float(size_mm[0]), float(size_mm[1])),
        rectangle_box_mm=box_mm,
        connected_components=connected_components,
    )


def _metrics_are_valid(metrics: AssemblyMetrics, config: VisionConfig) -> bool:
    short, long = metrics.rectangle_size_mm
    return (
        metrics.connected_components == 1
        and config.target_short_min_mm <= short <= config.target_short_max_mm
        and config.target_long_min_mm <= long <= config.target_long_max_mm
        and metrics.fill_ratio >= config.min_rectangle_fill_ratio
        and metrics.overlap_ratio <= config.max_overlap_ratio
        and metrics.hole_ratio <= config.max_hole_ratio
        and metrics.boundary_p95_mm <= config.max_boundary_p95_mm
        and metrics.boundary_p99_mm <= config.max_boundary_p99_mm
        and metrics.side_coverage_min
        >= config.min_rectangle_side_coverage
        and metrics.convexity_ratio
        >= config.min_rectangle_convexity_ratio
        and metrics.corner_error_max_mm
        <= config.max_rectangle_corner_error_mm
    )


def _metrics_are_refinable_printed_candidate(
    metrics: AssemblyMetrics,
    config: VisionConfig,
) -> bool:
    """Keep only near-rectangles that a few millimetres of correction can fix."""
    short, long = metrics.rectangle_size_mm
    return (
        metrics.connected_components == 1
        and config.target_short_min_mm <= short <= config.target_short_max_mm
        and config.target_long_min_mm <= long <= config.target_long_max_mm
        and metrics.fill_ratio >= 0.85
        and metrics.overlap_ratio <= 0.04
        and metrics.hole_ratio <= 0.04
        and metrics.boundary_p95_mm <= 10.0
        and metrics.boundary_p99_mm <= 12.0
        and metrics.side_coverage_min >= 0.40
        and metrics.convexity_ratio >= 0.90
        and metrics.corner_error_max_mm <= 8.0
    )


def _geometry_score(
    metrics: AssemblyMetrics,
    config: VisionConfig,
    polygons: Optional[Sequence[np.ndarray]] = None,
) -> float:
    # Every size inside the official range is equally plausible.  Do not bias
    # the ranking toward the midpoint of that range; rank only assembly quality.
    score = (
        max(0.0, 1.0 - metrics.fill_ratio)
        + 4.0 * metrics.overlap_ratio
        + 2.0 * metrics.hole_ratio
        + 0.025 * metrics.boundary_p95_mm
        + 0.015 * metrics.boundary_p99_mm
        + 0.10 * max(0.0, 1.0 - metrics.side_coverage_min)
        + 0.08 * max(0.0, 1.0 - metrics.convexity_ratio)
        + 0.01 * metrics.corner_error_max_mm
    )
    if polygons:
        boundary_support, corner_support = _rectangle_feature_scores(
            polygons,
            max(1.5, min(2.5, config.solver_collinear_tolerance_mm)),
        )
        # The measured-card regression has two raster-valid arrangements with
        # almost identical fill.  The real one is distinguished by long,
        # straight outer chains and supported rectangle corners.
        score += (
            0.12 * (1.0 - boundary_support)
            + 0.01 * (1.0 - corner_support)
        )
    return score


def _refine_printed_layout(
    transforms: Dict[int, np.ndarray],
    pieces_by_id: Dict[int, PieceObservation],
    config: VisionConfig,
) -> Optional[_Leaf]:
    """Locally correct the small contour-induced pose error of a card layout."""
    refined = {
        piece_id: matrix.copy()
        for piece_id, matrix in transforms.items()
    }
    anchor_id = max(
        pieces_by_id,
        key=lambda piece_id: (
            pieces_by_id[piece_id].area_mm2,
            -piece_id,
        ),
    )
    moving_ids = [
        piece_id
        for piece_id in sorted(pieces_by_id)
        if piece_id != anchor_id
    ]

    def evaluate(
        candidate: Dict[int, np.ndarray],
    ) -> Tuple[float, AssemblyMetrics, List[np.ndarray]]:
        polygons = [
            _transform_rigid(
                pieces_by_id[piece_id].polygon_mm,
                candidate[piece_id],
            )
            for piece_id in sorted(pieces_by_id)
        ]
        metrics = _raster_metrics(polygons, config)
        return _geometry_score(metrics, config, polygons), metrics, polygons

    try:
        best_score, best_metrics, _ = evaluate(refined)
    except SolveError:
        return None
    if not _metrics_are_refinable_printed_candidate(best_metrics, config):
        return None

    # Coarse-to-fine coordinate descent is deliberately bounded: at most
    # 180 raster evaluations for four pieces, with the largest piece fixed.
    for translation_step_mm, rotation_step_deg in (
        (3.0, 3.0),
        (2.0, 2.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (0.25, 0.25),
    ):
        for _ in range(2):
            improved = False
            for piece_id in moving_ids:
                current = refined[piece_id]
                placed = _transform_rigid(
                    pieces_by_id[piece_id].polygon_mm,
                    current,
                )
                center = placed.mean(axis=0)
                adjustments = (
                    rigid_matrix(
                        0.0,
                        np.array([translation_step_mm, 0.0]),
                    ),
                    rigid_matrix(
                        0.0,
                        np.array([-translation_step_mm, 0.0]),
                    ),
                    rigid_matrix(
                        0.0,
                        np.array([0.0, translation_step_mm]),
                    ),
                    rigid_matrix(
                        0.0,
                        np.array([0.0, -translation_step_mm]),
                    ),
                    rotation_about(
                        math.radians(rotation_step_deg), center
                    ),
                    rotation_about(
                        math.radians(-rotation_step_deg), center
                    ),
                )
                local_best = best_score
                local_matrix: Optional[np.ndarray] = None
                local_metrics: Optional[AssemblyMetrics] = None
                for adjustment in adjustments:
                    candidate = dict(refined)
                    candidate[piece_id] = adjustment @ current
                    try:
                        score, metrics, _ = evaluate(candidate)
                    except SolveError:
                        continue
                    if score + 1e-9 < local_best:
                        local_best = score
                        local_matrix = candidate[piece_id]
                        local_metrics = metrics
                if local_matrix is not None and local_metrics is not None:
                    refined[piece_id] = local_matrix
                    best_score = local_best
                    best_metrics = local_metrics
                    improved = True
            if not improved:
                break

    if not _metrics_are_valid(best_metrics, config):
        return None
    final_polygons = [
        _transform_rigid(
            pieces_by_id[piece_id].polygon_mm,
            refined[piece_id],
        )
        for piece_id in sorted(pieces_by_id)
    ]
    return _Leaf(
        transforms=refined,
        metrics=best_metrics,
        geometry_score=_geometry_score(
            best_metrics, config, final_polygons
        ),
    )


def _pose_key(matrix: np.ndarray) -> Tuple[int, int, int]:
    angle = wrap_degrees(
        np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
    )
    return (
        int(round(angle / 0.35)),
        int(round(float(matrix[0, 2]) / 0.35)),
        int(round(float(matrix[1, 2]) / 0.35)),
    )


def _pair_cycle_support(
    fixed_id: int,
    moving_id: int,
    relative: np.ndarray,
    pair_cache: Dict[Tuple[int, int], List[np.ndarray]],
    piece_ids: Sequence[int],
) -> float:
    """Return soft pose-graph support from a closing three-piece loop.

    ``relative`` maps the moving piece into the fixed piece's coordinates.
    A third piece closes the loop when ``T_fixed,moving @ T_moving,third``
    agrees with a directly measured ``T_fixed,third`` relation.  Missing
    loops remain neutral so tree-shaped puzzles are never rejected.
    """
    best_normalized_error = float("inf")
    for third_id in piece_ids:
        if third_id in (fixed_id, moving_id):
            continue
        direct_relations = pair_cache.get((fixed_id, third_id), ())
        moving_relations = pair_cache.get((moving_id, third_id), ())
        if not direct_relations or not moving_relations:
            continue
        direct_stack = np.asarray(direct_relations, dtype=np.float64)
        moving_stack = np.asarray(moving_relations, dtype=np.float64)
        composed = np.matmul(relative[None, :, :], moving_stack)
        translation_error = np.linalg.norm(
            direct_stack[:, None, :2, 2]
            - composed[None, :, :2, 2],
            axis=2,
        )
        direct_angle = np.degrees(
            np.arctan2(
                direct_stack[:, 1, 0],
                direct_stack[:, 0, 0],
            )
        )
        composed_angle = np.degrees(
            np.arctan2(
                composed[:, 1, 0],
                composed[:, 0, 0],
            )
        )
        angle_error = np.abs(
            (
                direct_angle[:, None]
                - composed_angle[None, :]
                + 180.0
            )
            % 360.0
            - 180.0
        )
        normalized_error = (
            np.square(translation_error / 2.5)
            + np.square(angle_error / 3.0)
        )
        best_normalized_error = min(
            best_normalized_error,
            float(np.min(normalized_error)),
        )
    if not np.isfinite(best_normalized_error):
        return 0.0
    return float(math.exp(-0.5 * best_normalized_error))


def _solution_key(transforms: Dict[int, np.ndarray]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(
        (piece_id,) + _pose_key(transforms[piece_id])
        for piece_id in sorted(transforms)
    )


def _ray_opposition_error(
    ray_a: np.ndarray, ray_b: np.ndarray
) -> float:
    """Return zero for the supplementary adjacent rays required by C2."""
    length_a = math.hypot(float(ray_a[0]), float(ray_a[1]))
    length_b = math.hypot(float(ray_b[0]), float(ray_b[1]))
    if min(length_a, length_b) < 1e-6:
        return 1.0
    cosine = (
        float(ray_a[0]) * float(ray_b[0])
        + float(ray_a[1]) * float(ray_b[1])
    ) / (length_a * length_b)
    return 1.0 + float(np.clip(cosine, -1.0, 1.0))


def _endpoint_continuation_error(
    placed_polygon: np.ndarray,
    placed_edge: int,
    placed_at_start: bool,
    source_polygon: np.ndarray,
    source_edge: int,
    source_at_start: bool,
) -> float:
    """Evaluate the paper's C2 supplementary-angle mate constraint."""
    placed_count = len(placed_polygon)
    source_count = len(source_polygon)
    if placed_at_start:
        placed_endpoint = placed_polygon[placed_edge]
        placed_neighbor = placed_polygon[(placed_edge - 1) % placed_count]
    else:
        placed_endpoint = placed_polygon[(placed_edge + 1) % placed_count]
        placed_neighbor = placed_polygon[(placed_edge + 2) % placed_count]
    if source_at_start:
        source_endpoint = source_polygon[source_edge]
        source_neighbor = source_polygon[(source_edge - 1) % source_count]
    else:
        source_endpoint = source_polygon[(source_edge + 1) % source_count]
        source_neighbor = source_polygon[(source_edge + 2) % source_count]
    return _ray_opposition_error(
        placed_neighbor - placed_endpoint,
        source_neighbor - source_endpoint,
    )


def _logical_edge_offsets(
    fixed_length: float,
    moving_length: float,
    measured_edge_lengths: Sequence[float],
    tolerance_mm: float,
    recovery_mode: bool = False,
) -> List[float]:
    """Return bounded interior offsets implied by measured puzzle edges.

    Crossing-cut layouts can place a short seam strictly inside a longer
    collinear seam.  The normal pass deliberately keeps the historical first
    two hypotheses.  Recovery keeps more of the already-derived positions so
    a correct offset is not discarded merely because another piece was
    numbered first.
    """
    low = min(0.0, fixed_length - moving_length)
    high = max(0.0, fixed_length - moving_length)
    if high - low <= 2.0 * tolerance_mm:
        return []

    candidates = []
    for length in measured_edge_lengths:
        value = float(length)
        candidates.extend(
            [
                value,
                value - moving_length,
                fixed_length - value,
                fixed_length - moving_length - value,
            ]
        )
    candidates.append(0.5 * (low + high))

    result: List[float] = []
    endpoint_margin = max(0.75, 0.35 * tolerance_mm)
    result_limit = 8 if recovery_mode else 2
    for value in candidates:
        if not low + endpoint_margin < value < high - endpoint_margin:
            continue
        value = min(high, max(low, value))
        if any(abs(value - previous) < 0.75 for previous in result):
            continue
        result.append(value)
        if len(result) >= result_limit:
            break
    return result


def _candidate_transforms(
    new_piece: PieceObservation,
    pieces_by_id: Dict[int, PieceObservation],
    placed: Dict[int, np.ndarray],
    config: VisionConfig,
    recovery_mode: bool = False,
) -> List[np.ndarray]:
    """Build a bounded C1/C2 mating graph for one new piece.

    The previous search aligned every source edge to every destination edge
    and only rejected poses after full contact measurement.  On MaixCAM2 that
    made the branching factor dominate runtime.  Crossing-cut mates must
    instead pass the paper's edge-length (C1) and supplementary-angle (C2)
    tests.  Full mates use a midpoint fit; unequal T-junction mates retain the
    two possible endpoint fits.
    """
    source_polygon = new_piece.polygon_mm
    ranked_by_source_edge: Dict[
        int,
        Dict[
            Tuple[int, int, int],
            Tuple[float, np.ndarray, Tuple[int, int], bool],
        ],
    ] = {}
    angle_limit = 0.70
    tolerance = float(config.solver_collinear_tolerance_mm)
    measured_edge_lengths = [
        _segment_length(start, end)
        for piece in pieces_by_id.values()
        for _, start, end in _edges(piece.polygon_mm)
    ]

    for placed_id, placed_matrix in placed.items():
        placed_polygon = _transform_rigid(
            pieces_by_id[placed_id].polygon_mm, placed_matrix
        )
        for placed_edge, p0, p1 in _edges(placed_polygon):
            placed_vector = p1 - p0
            placed_length = _segment_length(p0, p1)
            if placed_length < config.solver_min_contact_mm:
                continue
            for source_edge, q0, q1 in _edges(source_polygon):
                source_vector = q1 - q0
                source_length = _segment_length(q0, q1)
                if source_length < config.solver_min_contact_mm:
                    continue
                maximum_length = max(placed_length, source_length)
                minimum_length = min(placed_length, source_length)
                difference = maximum_length - minimum_length
                full_tolerance = max(
                    2.5 * tolerance, 0.12 * maximum_length
                )
                full_match = difference <= full_tolerance
                length_ratio = minimum_length / maximum_length
                if not full_match and length_ratio > 0.88:
                    continue

                angle = float(
                    np.arctan2(-placed_vector[1], -placed_vector[0])
                    - np.arctan2(source_vector[1], source_vector[0])
                )
                cosine = math.cos(angle)
                sine = math.sin(angle)
                rotation = np.array(
                    [[cosine, -sine], [sine, cosine]], dtype=np.float64
                )
                rotated_polygon = source_polygon @ rotation.T
                rotated_q0 = q0 @ rotation.T
                rotated_q1 = q1 @ rotation.T
                edge_ranked = ranked_by_source_edge.setdefault(
                    source_edge, {}
                )

                candidate_specs = []
                if full_match:
                    translation = (
                        0.5 * (p0 + p1)
                        - 0.5 * (rotated_q0 + rotated_q1)
                    )
                    candidate_specs.append(
                        (translation, True, False, False, False)
                    )
                else:
                    candidate_specs.extend(
                        [
                            (
                                p1 - rotated_q0,
                                False,
                                False,
                                True,
                                False,
                            ),
                            (
                                p0 - rotated_q1,
                                False,
                                True,
                                False,
                                False,
                            ),
                        ]
                    )
                    if length_ratio <= 0.65:
                        fixed_axis = placed_vector / placed_length
                        for offset in _logical_edge_offsets(
                            placed_length,
                            source_length,
                            measured_edge_lengths,
                            tolerance,
                            recovery_mode=recovery_mode,
                        ):
                            translation = (
                                p0
                                + fixed_axis * offset
                                - rotated_q1
                            )
                            candidate_specs.append(
                                (
                                    translation,
                                    False,
                                    False,
                                    False,
                                    True,
                                )
                            )

                for (
                    translation,
                    check_both,
                    placed_at_start,
                    source_at_start,
                    logical_offset,
                ) in candidate_specs:
                    matrix = rigid_matrix(angle, translation)
                    candidate_polygon = rotated_polygon + translation
                    if logical_offset:
                        angle_error = 0.0
                        length_cost = 0.35 + 0.08 * (1.0 - length_ratio)
                    elif check_both:
                        endpoint_errors = (
                            _endpoint_continuation_error(
                                placed_polygon,
                                placed_edge,
                                True,
                                candidate_polygon,
                                source_edge,
                                False,
                            ),
                            _endpoint_continuation_error(
                                placed_polygon,
                                placed_edge,
                                False,
                                candidate_polygon,
                                source_edge,
                                True,
                            )
                        )
                        # A legal concave/polyline cut can continue into
                        # another seam at one endpoint.  Keep the strong
                        # endpoint as C2 evidence and treat the other as a
                        # soft ranking term rather than a hard rejection.
                        angle_error = min(endpoint_errors) + 0.15 * max(
                            endpoint_errors
                        )
                        length_cost = difference / maximum_length
                    else:
                        angle_error = _endpoint_continuation_error(
                            placed_polygon,
                            placed_edge,
                            placed_at_start,
                            candidate_polygon,
                            source_edge,
                            source_at_start,
                        )
                        length_cost = 0.16 + 0.08 * (1.0 - length_ratio)
                    if (
                        not full_match
                        and not logical_offset
                        and angle_error > angle_limit
                    ):
                        continue
                    angle_weight = (
                        0.0
                        if logical_offset
                        else (0.20 if full_match else 0.55)
                    )
                    rank = (
                        length_cost
                        + angle_weight * angle_error
                        - 0.002 * minimum_length
                    )
                    key = _pose_key(matrix)
                    previous = edge_ranked.get(key)
                    if previous is None or rank < previous[0]:
                        edge_ranked[key] = (
                            rank,
                            matrix,
                            (placed_id, placed_edge),
                            logical_offset,
                        )

    # Paper-style top-T filtering: retain both endpoint alignments for each of
    # the best few edge-pair hypotheses.  Ranking six individual poses used to
    # discard the correct far-end placement of a long seam whenever its
    # near-end C2 measurement happened to look cleaner.
    ranked: List[Tuple[float, np.ndarray]] = []
    for edge_candidates in ranked_by_source_edge.values():
        by_edge_pair: Dict[
            Tuple[int, int], List[Tuple[float, np.ndarray, bool]]
        ] = {}
        for rank, matrix, pair_key, logical in edge_candidates.values():
            by_edge_pair.setdefault(pair_key, []).append(
                (rank, matrix, logical)
            )
        best_pairs = sorted(
            by_edge_pair.values(),
            key=lambda items: min(item[0] for item in items),
        )[:3]
        for variants in best_pairs:
            ordinary = sorted(
                (
                    (rank, matrix)
                    for rank, matrix, logical in variants
                    if not logical
                ),
                key=lambda item: item[0],
            )[:2]
            logical = sorted(
                (
                    (rank, matrix)
                    for rank, matrix, is_logical in variants
                    if is_logical
                ),
                key=lambda item: item[0],
            )[: (4 if recovery_mode else 1)]
            ranked.extend(ordinary + logical)
    ordered = sorted(ranked, key=lambda item: item[0])
    return [
        matrix
        for _, matrix in ordered[: config.solver_max_pair_candidates]
    ]


def _t_junction_transforms(
    new_piece: PieceObservation,
    polygons: Dict[int, np.ndarray],
    config: VisionConfig,
) -> List[np.ndarray]:
    """Generate bounded poses enabled by a third-piece interior edge anchor."""
    tolerance = config.solver_collinear_tolerance_mm
    source_polygon = new_piece.polygon_mm
    ranked_by_source_edge: Dict[
        int, Dict[Tuple[int, int, int], Tuple[float, np.ndarray]]
    ] = {}
    for owner_id, polygon in polygons.items():
        for _, start, end in _edges(polygon):
            direction = end - start
            length = _segment_length(start, end)
            if length < config.solver_min_contact_mm:
                continue
            unit = direction / length
            anchors: List[np.ndarray] = []
            for other_id, other in polygons.items():
                if other_id == owner_id:
                    continue
                line_distance = np.abs(
                    direction[0] * (other[:, 1] - start[1])
                    - direction[1] * (other[:, 0] - start[0])
                ) / length
                projection = (other - start) @ unit
                interior = (
                    (line_distance <= tolerance)
                    & (projection > tolerance)
                    & (projection < length - tolerance)
                )
                anchors.extend(other[interior])
            if not anchors:
                continue
            anchor_array = np.asarray(anchors, dtype=np.float64)
            anchor_keys = np.round(anchor_array / 0.05).astype(np.int64)
            _, unique_indices = np.unique(
                anchor_keys, axis=0, return_index=True
            )
            anchor_array = anchor_array[np.sort(unique_indices)]
            anchor_projections = (anchor_array - start) @ unit
            known_positions = np.concatenate(
                [np.array([0.0, length]), anchor_projections]
            )

            for source_edge, q0, q1 in _edges(source_polygon):
                source_vector = q1 - q0
                source_length = _segment_length(q0, q1)
                if (
                    source_length < config.solver_min_contact_mm
                    or source_length > length + 2.0 * tolerance
                ):
                    continue
                angle = float(
                    np.arctan2(-direction[1], -direction[0])
                    - np.arctan2(source_vector[1], source_vector[0])
                )
                rotation = np.array(
                    [
                        [math.cos(angle), -math.sin(angle)],
                        [math.sin(angle), math.cos(angle)],
                    ],
                    dtype=np.float64,
                )
                rotated_endpoints = (q0 @ rotation.T, q1 @ rotation.T)
                edge_ranked = ranked_by_source_edge.setdefault(
                    source_edge, {}
                )
                for endpoint_index, endpoint in enumerate(rotated_endpoints):
                    other_endpoint = rotated_endpoints[1 - endpoint_index]
                    for anchor_index, anchor in enumerate(anchor_array):
                        translation = anchor - endpoint
                        matrix = rigid_matrix(angle, translation)
                        other_projection = float(
                            np.dot(
                                other_endpoint + translation - start,
                                unit,
                            )
                        )
                        low = min(
                            float(anchor_projections[anchor_index]),
                            other_projection,
                        )
                        high = max(
                            float(anchor_projections[anchor_index]),
                            other_projection,
                        )
                        if low < -tolerance or high > length + tolerance:
                            continue
                        key = _pose_key(matrix)
                        junction_error = float(
                            np.min(
                                np.abs(known_positions - other_projection)
                            )
                        )
                        rank = (
                            junction_error / max(length, 1.0)
                            - 0.002 * source_length
                        )
                        previous = edge_ranked.get(key)
                        if previous is None or rank < previous[0]:
                            edge_ranked[key] = (rank, matrix)
    ranked: List[Tuple[float, np.ndarray]] = []
    for edge_candidates in ranked_by_source_edge.values():
        ranked.extend(
            sorted(edge_candidates.values(), key=lambda item: item[0])[:6]
        )
    return [
        matrix
        for _, matrix in sorted(ranked, key=lambda item: item[0])[
            : config.solver_max_pair_candidates
        ]
    ]


def _incremental_valid(
    polygons: Dict[int, np.ndarray],
    new_piece_id: int,
    total_piece_area: float,
    config: VisionConfig,
) -> bool:
    all_points = np.vstack(list(polygons.values()))
    rect = cv2.minAreaRect(all_points.astype(np.float32))
    short, long = sorted((float(rect[1][0]), float(rect[1][1])))
    rect_area = short * long
    tolerance = config.solver_collinear_tolerance_mm
    if (
        short > config.target_short_max_mm + tolerance
        or long > config.target_long_max_mm + tolerance
        or rect_area
        > total_piece_area
        / max(config.min_rectangle_fill_ratio - 0.02, 0.75)
    ):
        return False
    overlap_area = 0.0
    new_polygon = polygons[new_piece_id]
    for piece_id, polygon in polygons.items():
        if piece_id == new_piece_id:
            continue
        overlap_area += _polygon_overlap_area(new_polygon, polygon)
        if overlap_area > config.solver_max_overlap_mm2:
            return False
    return True


def _cached_incremental_valid(
    polygons: Dict[int, np.ndarray],
    transforms: Dict[int, np.ndarray],
    new_piece_id: int,
    pieces_by_id: Dict[int, PieceObservation],
    total_piece_area: float,
    config: VisionConfig,
    overlap_cache: Dict[
        Tuple[int, int, Tuple[int, int, int]],
        float,
    ],
) -> bool:
    """Incremental validation with rigid-relative pair overlap reuse."""
    all_points = np.vstack(list(polygons.values()))
    rect = cv2.minAreaRect(all_points.astype(np.float32))
    short, long = sorted((float(rect[1][0]), float(rect[1][1])))
    rect_area = short * long
    tolerance = config.solver_collinear_tolerance_mm
    if (
        short > config.target_short_max_mm + tolerance
        or long > config.target_long_max_mm + tolerance
        or rect_area
        > total_piece_area
        / max(config.min_rectangle_fill_ratio - 0.02, 0.75)
    ):
        return False

    overlap_area = 0.0
    new_matrix = transforms[new_piece_id]
    for other_id in polygons:
        if other_id == new_piece_id:
            continue
        if new_piece_id < other_id:
            first_id, second_id = new_piece_id, other_id
            relative = np.linalg.inv(new_matrix) @ transforms[other_id]
        else:
            first_id, second_id = other_id, new_piece_id
            relative = np.linalg.inv(transforms[other_id]) @ new_matrix
        cache_key = (first_id, second_id, _pose_key(relative))
        pair_overlap = overlap_cache.get(cache_key)
        if pair_overlap is None:
            second_polygon = _transform_rigid(
                pieces_by_id[second_id].polygon_mm,
                relative,
            )
            pair_overlap = _polygon_overlap_area(
                pieces_by_id[first_id].polygon_mm,
                second_polygon,
            )
            overlap_cache[cache_key] = pair_overlap
        overlap_area += pair_overlap
        if overlap_area > config.solver_max_overlap_mm2:
            return False
    return True


def _pair_groups_incrementally_valid(
    polygons: Dict[int, np.ndarray],
    moving_piece_ids: Sequence[int],
    total_piece_area: float,
    config: VisionConfig,
) -> bool:
    """Validate a rigid 2+2 merge without rechecking either internal seam."""
    all_points = np.vstack(list(polygons.values()))
    rect = cv2.minAreaRect(all_points.astype(np.float32))
    short, long = sorted((float(rect[1][0]), float(rect[1][1])))
    rect_area = short * long
    tolerance = config.solver_collinear_tolerance_mm
    if (
        short > config.target_short_max_mm + tolerance
        or long > config.target_long_max_mm + tolerance
        or rect_area
        > total_piece_area
        / max(config.min_rectangle_fill_ratio - 0.02, 0.75)
    ):
        return False

    moving_ids = set(moving_piece_ids)
    fixed_ids = set(polygons) - moving_ids
    for moving_id in moving_ids:
        overlap_area = 0.0
        for fixed_id in fixed_ids:
            overlap_area += _polygon_overlap_area(
                polygons[moving_id],
                polygons[fixed_id],
            )
            if overlap_area > config.solver_max_overlap_mm2:
                return False
    return True


def _rectangle_feature_scores(
    polygons: Sequence[np.ndarray], tolerance_mm: float
) -> Tuple[float, float]:
    """Measure outer-side coverage and supported 90-degree rectangle corners."""
    if not polygons:
        return 0.0, 0.0
    points = np.vstack(polygons).astype(np.float32)
    rect = cv2.minAreaRect(points)
    width, height = map(float, rect[1])
    perimeter = 2.0 * (width + height)
    if min(width, height, perimeter) < 1e-6:
        return 0.0, 0.0
    box = cv2.boxPoints(rect).astype(np.float64)

    boundary_length = 0.0
    vertices: List[Tuple[np.ndarray, float]] = []
    for polygon in polygons:
        contour = np.asarray(polygon, dtype=np.float64)
        count = len(contour)
        for index, start in enumerate(contour):
            end = contour[(index + 1) % count]
            edge_length = _segment_length(start, end)
            side_error = min(
                max(
                    float(
                        point_segment_distance(
                            start[None, :],
                            box[side],
                            box[(side + 1) % 4],
                        )[0]
                    ),
                    float(
                        point_segment_distance(
                            end[None, :],
                            box[side],
                            box[(side + 1) % 4],
                        )[0]
                    ),
                )
                for side in range(4)
            )
            if side_error <= tolerance_mm:
                boundary_length += edge_length

            previous = contour[(index - 1) % count] - start
            following = end - start
            denominator = _segment_length(
                np.zeros(2), previous
            ) * _segment_length(np.zeros(2), following)
            if denominator < 1e-9:
                continue
            cosine = float(
                np.clip(np.dot(previous, following) / denominator, -1.0, 1.0)
            )
            vertices.append(
                (start, math.degrees(math.acos(cosine)))
            )

    boundary_support = min(1.0, boundary_length / perimeter)
    corner_radius = max(5.0, 2.5 * tolerance_mm)
    corner_values = []
    for corner in box:
        best = 0.0
        for vertex, angle_deg in vertices:
            distance_score = max(
                0.0,
                1.0 - float(np.linalg.norm(vertex - corner)) / corner_radius,
            )
            angle_score = max(0.0, 1.0 - abs(angle_deg - 90.0) / 25.0)
            best = max(best, distance_score * angle_score)
        corner_values.append(best)
    return boundary_support, float(np.mean(corner_values))


def _partial_priority(
    polygons: Dict[int, np.ndarray],
    total_piece_area: float,
    config: VisionConfig,
) -> float:
    points = np.vstack(list(polygons.values())).astype(np.float32)
    rect = cv2.minAreaRect(points)
    short, long = sorted((float(rect[1][0]), float(rect[1][1])))
    rect_area = max(short * long, 1e-6)
    placed_area = sum(polygon_area(polygon) for polygon in polygons.values())
    compactness_error = max(
        0.0, 1.0 - min(placed_area / rect_area, 1.0)
    )
    final_area_error = abs(rect_area - total_piece_area) / total_piece_area
    undersize = (
        max(0.0, config.target_short_min_mm - short)
        / max(config.target_short_min_mm, 1.0)
        + max(0.0, config.target_long_min_mm - long)
        / max(config.target_long_min_mm, 1.0)
    )
    contact = 0.0
    ids = sorted(polygons)
    for index, piece_a in enumerate(ids):
        for piece_b in ids[index + 1 :]:
            contact += sum(
                item.length
                for item in _contacts_between(
                    piece_a,
                    polygons[piece_a],
                    piece_b,
                    polygons[piece_b],
                    config,
                )
            )
    boundary_support, corner_support = _rectangle_feature_scores(
        list(polygons.values()),
        max(1.5, min(2.5, config.solver_collinear_tolerance_mm)),
    )
    return (
        0.65 * compactness_error
        + 0.12 * final_area_error
        + 0.06 * undersize
        - 0.18 * boundary_support
        - 0.02 * corner_support
        - 0.0004 * contact
    )


def _unfinished_priority(
    polygons: Dict[int, np.ndarray],
    total_piece_area: float,
    config: VisionConfig,
    source_rank: int,
) -> float:
    """Rank an unfinished assembly without final-rectangle analysis."""
    points = np.vstack(list(polygons.values())).astype(np.float32)
    rect = cv2.minAreaRect(points)
    short, long = sorted((float(rect[1][0]), float(rect[1][1])))
    rect_area = max(short * long, 1e-6)
    placed_area = sum(polygon_area(polygon) for polygon in polygons.values())
    compactness_error = max(
        0.0, 1.0 - min(placed_area / rect_area, 1.0)
    )
    final_area_error = abs(rect_area - total_piece_area) / max(
        total_piece_area, 1e-6
    )
    undersize = (
        max(0.0, config.target_short_min_mm - short)
        / max(config.target_short_min_mm, 1.0)
        + max(0.0, config.target_long_min_mm - long)
        / max(config.target_long_min_mm, 1.0)
    )
    # Candidate transforms are already ordered by C1/C2 seam quality.  Keep
    # that evidence as a small tie-breaker without repeating contact tracing.
    return (
        0.65 * compactness_error
        + 0.12 * final_area_error
        + 0.06 * undersize
        + 0.002 * source_rank
    )


def _convex_sat_overlap_depth(
    polygon_a: np.ndarray, polygon_b: np.ndarray
) -> float:
    """Return the minimum separating-axis penetration for two convex polygons."""
    contour_a = np.asarray(polygon_a, dtype=np.float64)
    contour_b = np.asarray(polygon_b, dtype=np.float64)
    minimum_depth = float("inf")
    for contour in (contour_a, contour_b):
        edge_vectors = np.roll(contour, -1, axis=0) - contour
        for edge in edge_vectors:
            length = math.hypot(float(edge[0]), float(edge[1]))
            if length < 1e-9:
                continue
            axis = np.array([-edge[1], edge[0]], dtype=np.float64) / length
            projection_a = contour_a @ axis
            projection_b = contour_b @ axis
            depth = min(
                float(projection_a.max()), float(projection_b.max())
            ) - max(float(projection_a.min()), float(projection_b.min()))
            if depth <= 0.0:
                return 0.0
            minimum_depth = min(minimum_depth, depth)
    return 0.0 if not np.isfinite(minimum_depth) else minimum_depth


def _polygon_overlap_area(
    polygon_a: np.ndarray, polygon_b: np.ndarray
) -> float:
    contour_a = np.asarray(polygon_a, dtype=np.float32)
    contour_b = np.asarray(polygon_b, dtype=np.float32)
    if cv2.isContourConvex(contour_a) and cv2.isContourConvex(contour_b):
        # intersectConvexConvex occasionally returns an entire polygon when
        # two measured convex contours have an exactly collinear shared edge.
        # The separating-axis depth is both cheaper and numerically reliable
        # for distinguishing that zero-area contact from real penetration.
        if _convex_sat_overlap_depth(contour_a, contour_b) <= 1e-4:
            return 0.0
        area, _ = cv2.intersectConvexConvex(contour_a, contour_b)
        if area > 1e-6:
            return float(area)
        # OpenCV can return zero when two convex polygons share a nearly
        # collinear edge even though one still contains vertices of the
        # other.  Falling through to the raster path prevents a nested piece
        # from being accepted as a zero-overlap seam.
        contains_vertex = any(
            cv2.pointPolygonTest(
                contour_a, tuple(map(float, point)), False
            )
            > 0
            for point in contour_b
        ) or any(
            cv2.pointPolygonTest(
                contour_b, tuple(map(float, point)), False
            )
            > 0
            for point in contour_a
        )
        if not contains_vertex:
            return 0.0

    all_points = np.vstack([polygon_a, polygon_b])
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    scale = 2.0
    margin = 2.0
    size = np.ceil((maximum - minimum + 2.0 * margin) * scale).astype(int) + 3
    if np.any(size <= 0) or np.any(size > 500):
        return float("inf")
    offset = -minimum + margin
    mask_a = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    mask_b = np.zeros_like(mask_a)
    cv2.fillPoly(
        mask_a,
        [np.round((polygon_a + offset) * scale).astype(np.int32)],
        1,
    )
    cv2.fillPoly(
        mask_b,
        [np.round((polygon_b + offset) * scale).astype(np.int32)],
        1,
    )
    # fillPoly includes the shared rasterized boundary in both masks.  That is
    # zero-area contact geometrically, but for a 120 mm seam at 2 px/mm it can
    # look like about 60 mm² of overlap and wrongly prune a valid assembly.
    # Compare one-pixel interiors; sub-millimetre slivers are intentionally
    # tolerated by the solver's measurement-error budget.
    kernel = np.ones((3, 3), dtype=np.uint8)
    interior_a = cv2.erode(mask_a, kernel, iterations=1)
    interior_b = cv2.erode(mask_b, kernel, iterations=1)
    return float(np.count_nonzero(interior_a & interior_b)) / (scale * scale)


def _sample_lab(
    lab: np.ndarray, points_mm: np.ndarray, px_per_mm: float
) -> np.ndarray:
    points_px = np.asarray(points_mm, dtype=np.float32) * float(px_per_mm)
    map_x = points_px[:, 0].reshape(-1, 1)
    map_y = points_px[:, 1].reshape(-1, 1)
    sampled = cv2.remap(
        lab,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(-1, 3).astype(np.float32)


def _sample_mask(
    mask: np.ndarray, points_mm: np.ndarray, px_per_mm: float
) -> np.ndarray:
    points_px = np.asarray(points_mm, dtype=np.float32) * float(px_per_mm)
    map_x = points_px[:, 0].reshape(-1, 1)
    map_y = points_px[:, 1].reshape(-1, 1)
    sampled = cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return sampled.reshape(-1) > 0


def _is_equal_card_quarters(
    pieces: Sequence[PieceObservation],
) -> bool:
    """Recognize four near-identical rectangular card quadrants."""
    if len(pieces) != 4 or any(len(piece.polygon_mm) != 4 for piece in pieces):
        return False
    areas = np.asarray(
        [piece.area_mm2 for piece in pieces], dtype=np.float64
    )
    if float(areas.min()) < 1e-6 or float(areas.max() / areas.min()) > 1.18:
        return False
    shapes = []
    for piece in pieces:
        rectangle = cv2.minAreaRect(
            np.asarray(piece.polygon_mm, dtype=np.float32)
        )
        short, long = sorted(map(float, rectangle[1]))
        if short < 1e-6 or long < 1e-6:
            return False
        rectangularity = polygon_area(piece.polygon_mm) / (short * long)
        if rectangularity < 0.90:
            return False
        shapes.append((short, long))
    shape_array = np.asarray(shapes, dtype=np.float64)
    return bool(
        np.max(shape_array[:, 0]) / np.min(shape_array[:, 0]) <= 1.15
        and np.max(shape_array[:, 1]) / np.min(shape_array[:, 1]) <= 1.15
    )


def _is_card_strip_partition(
    pieces: Sequence[PieceObservation],
) -> bool:
    """Recognize a playing card cut across its full width into four strips."""
    if len(pieces) != 4 or any(len(piece.polygon_mm) != 4 for piece in pieces):
        return False
    dimensions = []
    for piece in pieces:
        rectangle = cv2.minAreaRect(
            np.asarray(piece.polygon_mm, dtype=np.float32)
        )
        short, long = sorted(map(float, rectangle[1]))
        if short < 1e-6 or long < 1e-6:
            return False
        rectangularity = polygon_area(piece.polygon_mm) / (short * long)
        if rectangularity < 0.88:
            return False
        dimensions.append((short, long))
    shape_array = np.asarray(dimensions, dtype=np.float64)
    common_width = float(np.median(shape_array[:, 1]))
    assembled_aspect = float(np.sum(shape_array[:, 0]) / common_width)
    return bool(
        np.max(shape_array[:, 1]) / np.min(shape_array[:, 1]) <= 1.15
        and 1.25 <= assembled_aspect <= 2.20
    )


def _assembled_card_features(
    points_mm: np.ndarray,
    piece_ids: Sequence[int],
    inverse: Dict[int, np.ndarray],
    safe_masks: Dict[int, np.ndarray],
    references: Dict[int, np.ndarray],
    lab: np.ndarray,
    px_per_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    features = np.zeros((len(points_mm), 3), dtype=np.float32)
    valid = np.zeros(len(points_mm), dtype=bool)
    scale = np.array([35.0, 22.0, 22.0], dtype=np.float32)
    for piece_id in piece_ids:
        source_points = _transform_rigid(
            points_mm, inverse[piece_id]
        )
        inside = (
            _sample_mask(safe_masks[piece_id], source_points, px_per_mm)
            & ~valid
        )
        if not np.any(inside):
            continue
        samples = _sample_lab(lab, source_points, px_per_mm)
        features[inside] = np.clip(
            (samples[inside] - references[piece_id]) / scale,
            -4.0,
            4.0,
        )
        valid[inside] = True
    return features, valid


def _quartered_card_feature_score(
    transformed: Dict[int, np.ndarray],
    inverse: Dict[int, np.ndarray],
    safe_masks: Dict[int, np.ndarray],
    references: Dict[int, np.ndarray],
    lab: np.ndarray,
    px_per_mm: float,
) -> Tuple[float, float]:
    """Score whole-card half-turn symmetry and diagonal corner indices."""
    points = np.vstack(list(transformed.values())).astype(np.float32)
    center, size, angle_deg = cv2.minAreaRect(points)
    width, height = map(float, size)
    if min(width, height) < 8.0:
        return 0.0, 0.0
    angle = math.radians(angle_deg)
    axis_x = np.array([math.cos(angle), math.sin(angle)])
    axis_y = np.array([-math.sin(angle), math.cos(angle)])
    margin = 2.5
    sample_x = np.linspace(
        -0.5 * width + margin,
        0.5 * width - margin,
        max(16, int(round((width - 2.0 * margin) / 2.0))),
    )
    sample_y = np.linspace(
        -0.5 * height + margin,
        0.5 * height - margin,
        max(24, int(round((height - 2.0 * margin) / 2.0))),
    )
    grid_x, grid_y = np.meshgrid(sample_x, sample_y)
    local_x = grid_x.ravel()
    local_y = grid_y.ravel()
    center_array = np.asarray(center, dtype=np.float64)
    sample_points = (
        center_array
        + local_x[:, None] * axis_x
        + local_y[:, None] * axis_y
    )
    mirrored_points = 2.0 * center_array - sample_points
    piece_ids = sorted(transformed)
    feature_a, valid_a = _assembled_card_features(
        sample_points,
        piece_ids,
        inverse,
        safe_masks,
        references,
        lab,
        px_per_mm,
    )
    feature_b, valid_b = _assembled_card_features(
        mirrored_points,
        piece_ids,
        inverse,
        safe_masks,
        references,
        lab,
        px_per_mm,
    )
    valid = valid_a & valid_b
    if np.count_nonzero(valid) < 100:
        return 0.0, 0.0
    strength_a = np.linalg.norm(feature_a, axis=1)
    strength_b = np.linalg.norm(feature_b, axis=1)
    evidence_weight = np.maximum(
        0.0, np.maximum(strength_a, strength_b) - 0.20
    )
    evidence_weight[~valid] = 0.0
    evidence = float(np.sum(evidence_weight))
    if evidence < 8.0:
        return 0.0, evidence
    mismatch = np.linalg.norm(feature_a - feature_b, axis=1)
    symmetry_cost = float(
        np.sum(
            evidence_weight
            * (np.minimum(mismatch / 3.0, 1.0) - 0.25)
        )
        / evidence
    )

    normalized_x = local_x / max(0.5 * width, 1e-6)
    normalized_y = local_y / max(0.5 * height, 1e-6)
    corner_strengths = []
    for sign_x, sign_y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        corner = (
            (sign_x * normalized_x > 0.62)
            & (sign_y * normalized_y > 0.68)
            & valid_a
        )
        values = np.maximum(0.0, strength_a[corner] - 0.20)
        if len(values) == 0:
            corner_strengths.append(0.0)
            continue
        strongest_count = max(3, int(math.ceil(0.12 * len(values))))
        strongest = np.partition(
            values, len(values) - strongest_count
        )[-strongest_count:]
        corner_strengths.append(float(np.mean(strongest)))
    first_diagonal = min(corner_strengths[0], corner_strengths[2])
    second_diagonal = min(corner_strengths[1], corner_strengths[3])
    corner_support = min(1.0, max(first_diagonal, second_diagonal) / 0.45)
    orientation_cost = 0.0
    center_region = (
        (np.abs(normalized_x) < 0.45)
        & (np.abs(normalized_y) < 0.45)
        & valid_a
    )
    ink_map = (
        (strength_a > 0.75) & center_region
    ).reshape(grid_x.shape).astype(np.uint8)
    ink_map = cv2.morphologyEx(
        ink_map,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    component_count, labels, statistics, centroids = (
        cv2.connectedComponentsWithStats(ink_map, connectivity=8)
    )
    if component_count > 1:
        grid_center = np.array(
            [0.5 * (ink_map.shape[1] - 1), 0.5 * (ink_map.shape[0] - 1)]
        )
        candidates = [
            label
            for label in range(1, component_count)
            if statistics[label, cv2.CC_STAT_AREA] >= 8
        ]
        if candidates:
            label = min(
                candidates,
                key=lambda item: float(
                    np.linalg.norm(centroids[item] - grid_center)
                ),
            )
            selected = labels.ravel() == label
            coordinates = np.column_stack(
                [local_x[selected], local_y[selected]]
            )
            centered = coordinates - coordinates.mean(axis=0)
            deviation = centered.std(axis=0)
            if float(deviation.min()) > 0.5:
                skew = np.mean(
                    np.power(centered / deviation, 3),
                    axis=0,
                )
                if height >= width:
                    short_skew, long_skew = map(
                        abs, (float(skew[0]), float(skew[1]))
                    )
                else:
                    short_skew, long_skew = map(
                        abs, (float(skew[1]), float(skew[0]))
                    )
                orientation_cost = (
                    1.00 * max(0.0, short_skew - long_skew)
                    - 0.12 * max(0.0, long_skew - short_skew)
                )
    return (
        symmetry_cost
        - 0.12 * corner_support
        + orientation_cost,
        evidence,
    )


def _prepare_texture_context(
    pieces: Sequence[PieceObservation],
    paper_bgr: np.ndarray,
) -> Tuple[
    np.ndarray,
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
]:
    lab = cv2.cvtColor(paper_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    safe_masks = {
        piece.piece_id: cv2.erode(
            piece.mask,
            np.ones((3, 3), dtype=np.uint8),
            iterations=2,
        )
        for piece in pieces
    }
    references: Dict[int, np.ndarray] = {}
    for piece in pieces:
        reference_mask = safe_masks[piece.piece_id]
        pixels = lab[reference_mask > 0]
        if len(pixels) == 0:
            pixels = lab[piece.mask > 0]
        if len(pixels) == 0:
            references[piece.piece_id] = np.zeros(
                3, dtype=np.float32
            )
            continue
        lightness_cutoff = float(np.percentile(pixels[:, 0], 60.0))
        reference_pixels = pixels[pixels[:, 0] >= lightness_cutoff]
        references[piece.piece_id] = np.median(
            reference_pixels, axis=0
        ).astype(np.float32)
    return lab, safe_masks, references


def _texture_score(
    pieces: Sequence[PieceObservation],
    transforms: Dict[int, np.ndarray],
    paper_bgr: np.ndarray,
    px_per_mm: float,
    config: VisionConfig,
    texture_context: Optional[
        Tuple[
            np.ndarray,
            Dict[int, np.ndarray],
            Dict[int, np.ndarray],
        ]
    ] = None,
) -> float:
    pieces_by_id = {piece.piece_id: piece for piece in pieces}
    if texture_context is None:
        texture_context = _prepare_texture_context(
            pieces, paper_bgr
        )
    lab, safe_masks, references = texture_context
    transformed = {
        piece_id: _transform_rigid(
            pieces_by_id[piece_id].polygon_mm, matrix
        )
        for piece_id, matrix in transforms.items()
    }
    inverse = {
        piece_id: np.linalg.inv(matrix) for piece_id, matrix in transforms.items()
    }
    if (
        _is_equal_card_quarters(pieces)
        or _is_card_strip_partition(pieces)
    ):
        card_score, card_evidence = _quartered_card_feature_score(
            transformed,
            inverse,
            safe_masks,
            references,
            lab,
            px_per_mm,
        )
        return card_score if card_evidence >= 8.0 else 0.0

    texture_cost = 0.0
    total_evidence = 0.0
    ids = sorted(transforms)
    for index, piece_a in enumerate(ids):
        for piece_b in ids[index + 1 :]:
            contacts = _contacts_between(
                piece_a,
                transformed[piece_a],
                piece_b,
                transformed[piece_b],
                config,
            )
            for contact in contacts:
                if contact.length < config.solver_min_contact_mm:
                    continue
                margin = min(1.5, 0.15 * contact.length)
                usable = contact.length - 2.0 * margin
                if usable < 3.0:
                    continue
                sample_count = max(8, int(round(usable / 1.2)))
                direction = (contact.end - contact.start) / contact.length
                positions = np.linspace(
                    margin, contact.length - margin, sample_count
                )
                seam = contact.start + positions[:, None] * direction

                polygon_a = transformed[piece_a]
                edge_a_start = polygon_a[contact.edge_a]
                edge_a_end = polygon_a[(contact.edge_a + 1) % len(polygon_a)]
                edge_direction_a = edge_a_end - edge_a_start
                edge_direction_a /= max(np.linalg.norm(edge_direction_a), 1e-9)
                normal_a = np.array(
                    [-edge_direction_a[1], edge_direction_a[0]]
                )

                polygon_b = transformed[piece_b]
                edge_b_start = polygon_b[contact.edge_b]
                edge_b_end = polygon_b[(contact.edge_b + 1) % len(polygon_b)]
                edge_direction_b = edge_b_end - edge_b_start
                edge_direction_b /= max(np.linalg.norm(edge_direction_b), 1e-9)
                normal_b = np.array(
                    [-edge_direction_b[1], edge_direction_b[0]]
                )

                offset_mm = 0.9
                a_near_global = seam + normal_a * offset_mm
                a_far_global = seam + normal_a * (2.0 * offset_mm)
                b_near_global = seam + normal_b * offset_mm
                b_far_global = seam + normal_b * (2.0 * offset_mm)

                a_near = _transform_rigid(a_near_global, inverse[piece_a])
                a_far = _transform_rigid(a_far_global, inverse[piece_a])
                b_near = _transform_rigid(b_near_global, inverse[piece_b])
                b_far = _transform_rigid(b_far_global, inverse[piece_b])

                valid = (
                    _sample_mask(
                        safe_masks[piece_a], a_near, px_per_mm
                    )
                    & _sample_mask(
                        safe_masks[piece_b], b_near, px_per_mm
                    )
                    & _sample_mask(
                        safe_masks[piece_a], a_far, px_per_mm
                    )
                    & _sample_mask(
                        safe_masks[piece_b], b_far, px_per_mm
                    )
                )
                if np.count_nonzero(valid) < 6:
                    continue

                profile_a = (
                    2.0 * _sample_lab(lab, a_near, px_per_mm)
                    - _sample_lab(lab, a_far, px_per_mm)
                    - references[piece_a]
                )[valid]
                profile_b = (
                    2.0 * _sample_lab(lab, b_near, px_per_mm)
                    - _sample_lab(lab, b_far, px_per_mm)
                    - references[piece_b]
                )[valid]
                color_error = np.linalg.norm(profile_a - profile_b, axis=1)
                color_error = np.minimum(color_error, 45.0)

                if len(profile_a) > 2:
                    gradient_a = np.diff(profile_a, axis=0)
                    gradient_b = np.diff(profile_b, axis=0)
                    gradient_error = np.linalg.norm(
                        gradient_a - gradient_b, axis=1
                    )
                    deviation = 0.5 * (
                        np.linalg.norm(profile_a[:-1], axis=1)
                        + np.linalg.norm(profile_b[:-1], axis=1)
                    )
                    gradient_strength = 0.5 * (
                        np.linalg.norm(gradient_a, axis=1)
                        + np.linalg.norm(gradient_b, axis=1)
                    )
                    evidence = np.minimum(
                        3.0,
                        np.maximum(0.0, deviation - 5.0) / 15.0
                        + gradient_strength / 25.0,
                    )
                    combined = (
                        color_error[:-1] / 30.0
                        + np.minimum(gradient_error, 50.0) / 50.0
                    )
                    # Blank white-to-white seams are neutral.  Matching,
                    # information-rich card artwork earns a small reward;
                    # mismatched artwork incurs a cost.
                    texture_cost += float(
                        np.sum(evidence * (combined - 0.30))
                    )
                    total_evidence += float(np.sum(evidence))
                else:
                    deviation = 0.5 * (
                        np.linalg.norm(profile_a, axis=1)
                        + np.linalg.norm(profile_b, axis=1)
                    )
                    evidence = np.minimum(
                        3.0,
                        np.maximum(0.0, deviation - 5.0) / 15.0,
                    )
                    texture_cost += float(
                        np.sum(evidence * (color_error / 30.0 - 0.30))
                    )
                    total_evidence += float(np.sum(evidence))

    seam_score = (
        float(texture_cost / total_evidence)
        if total_evidence >= 1e-6
        else 0.0
    )
    return seam_score


def _card_strip_states(
    pieces: Sequence[PieceObservation],
) -> Dict[Tuple[int, int], _CardStripState]:
    """Rectify every full-width card strip in its two rigid orientations."""
    states: Dict[Tuple[int, int], _CardStripState] = {}
    for piece in pieces:
        rectangle = cv2.minAreaRect(
            np.asarray(piece.polygon_mm, dtype=np.float32)
        )
        center = np.asarray(rectangle[0], dtype=np.float64)
        box = cv2.boxPoints(rectangle).astype(np.float64)
        edge_vectors = np.roll(box, -1, axis=0) - box
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        long_index = int(np.argmax(edge_lengths))
        long_direction = edge_vectors[long_index]
        width_mm = float(np.max(edge_lengths))
        height_mm = float(np.min(edge_lengths))
        angle = -math.atan2(
            float(long_direction[1]), float(long_direction[0])
        )
        rotation = rigid_matrix(
            angle, np.zeros(2, dtype=np.float64)
        )
        base = rigid_matrix(
            angle, -rotation[:2, :2] @ center
        )
        for flipped in (0, 1):
            transform = base
            if flipped:
                transform = (
                    rigid_matrix(
                        math.pi, np.zeros(2, dtype=np.float64)
                    )
                    @ base
                )
            states[(piece.piece_id, flipped)] = _CardStripState(
                piece_id=piece.piece_id,
                flipped=bool(flipped),
                transform=transform,
                inverse=np.linalg.inv(transform),
                width_mm=width_mm,
                height_mm=height_mm,
            )
    return states


def _card_strip_edge_profiles(
    states: Dict[Tuple[int, int], _CardStripState],
    lab: np.ndarray,
    safe_masks: Dict[int, np.ndarray],
    references: Dict[int, np.ndarray],
    px_per_mm: float,
    boundary_reconstructed: bool = False,
) -> Tuple[
    Dict[Tuple[Tuple[int, int], str, int], Tuple[np.ndarray, np.ndarray]],
    np.ndarray,
]:
    """Sample multi-depth LAB profiles around every possible card seam."""
    minimum_width = min(state.width_mm for state in states.values())
    side_margin = max(3.0, 0.08 * minimum_width)
    half_span = 0.5 * minimum_width - side_margin
    if half_span < 12.0:
        raise SolveError("card strips are too narrow for seam comparison")
    sample_count = max(48, int(round(2.0 * half_span / 0.75)))
    sample_x = np.linspace(-half_span, half_span, sample_count)
    shifts = np.linspace(-2.0, 2.0, 9)
    # Stay beyond the 1--2 mm contour uncertainty only when dark printing
    # forced the detector to reconstruct all four rectangular boundaries.
    # Normal measured contours retain their more local 1--3 mm evidence.
    depths = (
        (3.0, 4.0, 5.0)
        if boundary_reconstructed
        else (1.0, 2.0, 3.0)
    )
    feature_scale = np.array([22.0, 16.0, 16.0], dtype=np.float32)
    profiles: Dict[
        Tuple[Tuple[int, int], str, int],
        Tuple[np.ndarray, np.ndarray],
    ] = {}
    for state_key, state in states.items():
        for edge_name, sign in (("top", -1.0), ("bottom", 1.0)):
            for shift_index, shift in enumerate(shifts):
                samples = []
                valid_depths = []
                for depth in depths:
                    local_points = np.column_stack(
                        [
                            sample_x + shift,
                            np.full(
                                sample_x.shape,
                                sign
                                * (0.5 * state.height_mm - depth),
                            ),
                        ]
                    )
                    source_points = _transform_rigid(
                        local_points, state.inverse
                    )
                    samples.append(
                        (
                            _sample_lab(lab, source_points, px_per_mm)
                            - references[state.piece_id]
                        )
                        / feature_scale
                    )
                    valid_depths.append(
                        _sample_mask(
                            safe_masks[state.piece_id],
                            source_points,
                            px_per_mm,
                        )
                    )
                profiles[(state_key, edge_name, shift_index)] = (
                    np.asarray(samples, dtype=np.float32),
                    np.logical_and.reduce(valid_depths),
                )
    return profiles, shifts


def _card_strip_pair_costs(
    states: Dict[Tuple[int, int], _CardStripState],
    profiles: Dict[
        Tuple[Tuple[int, int], str, int],
        Tuple[np.ndarray, np.ndarray],
    ],
    shifts: np.ndarray,
) -> Tuple[
    Dict[Tuple[Tuple[int, int], Tuple[int, int]], float],
    Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[float, float, float]],
]:
    """Build a directed, reliability-normalized seam compatibility graph."""
    raw: Dict[
        Tuple[Tuple[int, int], Tuple[int, int]],
        Tuple[float, float, float],
    ] = {}
    zero_shift_index = int(np.argmin(np.abs(shifts)))
    for state_a in states:
        features_a, valid_a = profiles[
            (state_a, "bottom", zero_shift_index)
        ]
        for state_b in states:
            if state_a[0] == state_b[0]:
                continue
            best = (float("inf"), 0.0, 0.0)
            for shift_index, shift in enumerate(shifts):
                features_b, valid_b = profiles[
                    (state_b, "top", shift_index)
                ]
                valid = valid_a & valid_b
                minimum_valid = max(
                    22, int(round(0.38 * len(valid)))
                )
                if np.count_nonzero(valid) < minimum_valid:
                    continue

                # Symmetric first-order boundary prediction follows
                # Pomeranz et al.; tangential derivatives add the DC cue from
                # Son et al.  Per-piece white references suppress illumination
                # changes across the fixed camera's field of view.
                predicted_a = 2.0 * features_a[0] - features_a[1]
                predicted_b = 2.0 * features_b[0] - features_b[1]
                error_a = np.linalg.norm(
                    predicted_a - features_b[0], axis=1
                )
                error_b = np.linalg.norm(
                    predicted_b - features_a[0], axis=1
                )
                ink = np.maximum(
                    np.linalg.norm(features_a[0], axis=1),
                    np.linalg.norm(features_b[0], axis=1),
                )
                weights = 0.15 + np.minimum(2.0, ink)
                prediction_cost = float(
                    np.sum(
                        weights[valid]
                        * 0.5
                        * (
                            np.minimum(error_a[valid], 5.0)
                            + np.minimum(error_b[valid], 5.0)
                        )
                    )
                    / np.sum(weights[valid])
                )

                adjacent_valid = valid[:-1] & valid[1:]
                if np.count_nonzero(adjacent_valid) >= 8:
                    tangent_a = np.linalg.norm(
                        np.diff(predicted_a, axis=0)
                        - np.diff(features_b[0], axis=0),
                        axis=1,
                    )
                    tangent_b = np.linalg.norm(
                        np.diff(predicted_b, axis=0)
                        - np.diff(features_a[0], axis=0),
                        axis=1,
                    )
                    tangent_cost = 0.5 * (
                        float(
                            np.median(
                                np.minimum(
                                    tangent_a[adjacent_valid], 4.0
                                )
                            )
                        )
                        + float(
                            np.median(
                                np.minimum(
                                    tangent_b[adjacent_valid], 4.0
                                )
                            )
                        )
                    )
                else:
                    tangent_cost = 0.0

                evidence = float(
                    np.mean(
                        np.clip(ink[valid] - 0.25, 0.0, 2.0)
                    )
                )
                # Blank white seams remain possible, but must not beat a
                # distinctive printed continuation merely by having zero ink.
                cost = (
                    prediction_cost
                    + 0.25 * tangent_cost
                    + 0.18 / (0.25 + evidence)
                    + 0.015 * (float(shift) / 2.0) ** 2
                )
                if cost < best[0]:
                    best = (cost, float(shift), evidence)
            if not np.isfinite(best[0]):
                # A noisy min-area rectangle can place one of the two
                # hypothetical edges just outside the eroded mask.  Keep that
                # orientation in the graph with a clearly noncompetitive
                # cost; other orientations still retain measured evidence.
                best = (8.0, 0.0, 0.0)
            raw[(state_a, state_b)] = best

    normalized: Dict[
        Tuple[Tuple[int, int], Tuple[int, int]], float
    ] = {}
    for (state_a, state_b), (cost, _, _) in raw.items():
        row = [
            value[0]
            for (source, _), value in raw.items()
            if source == state_a
        ]
        column = [
            value[0]
            for (_, target), value in raw.items()
            if target == state_b
        ]
        # Quartile normalization measures how exceptional a match is for
        # both directed edges instead of trusting its absolute pixel error.
        denominator = max(
            1e-3,
            0.5
            * (
                float(np.percentile(row, 25.0))
                + float(np.percentile(column, 25.0))
            ),
        )
        normalized[(state_a, state_b)] = cost / denominator
    return normalized, raw


def _card_strip_outer_corner_strengths(
    states: Dict[Tuple[int, int], _CardStripState],
    lab: np.ndarray,
    safe_masks: Dict[int, np.ndarray],
    references: Dict[int, np.ndarray],
    px_per_mm: float,
) -> Dict[Tuple[Tuple[int, int], str], Tuple[float, float]]:
    """Measure rank/suit evidence at each possible outer card end."""
    total_height = sum(
        state.height_mm
        for key, state in states.items()
        if key[1] == 0
    )
    scale = np.array([35.0, 22.0, 22.0], dtype=np.float32)
    strengths: Dict[
        Tuple[Tuple[int, int], str], Tuple[float, float]
    ] = {}
    for state_key, state in states.items():
        depth = min(
            state.height_mm - 4.0, 0.22 * total_height
        )
        horizontal_span = min(
            0.5 * state.width_mm - 2.0,
            max(8.0, 0.18 * state.width_mm),
        )
        if depth <= 2.0 or horizontal_span <= 2.0:
            strengths[(state_key, "top")] = (0.0, 0.0)
            strengths[(state_key, "bottom")] = (0.0, 0.0)
            continue
        for edge_name in ("top", "bottom"):
            if edge_name == "top":
                y_values = np.linspace(
                    -0.5 * state.height_mm + 2.0,
                    -0.5 * state.height_mm + depth,
                    18,
                )
            else:
                y_values = np.linspace(
                    0.5 * state.height_mm - depth,
                    0.5 * state.height_mm - 2.0,
                    18,
                )
            side_values = []
            for side in ("left", "right"):
                if side == "left":
                    x_values = np.linspace(
                        -0.5 * state.width_mm + 2.0,
                        -0.5 * state.width_mm + horizontal_span,
                        18,
                    )
                else:
                    x_values = np.linspace(
                        0.5 * state.width_mm - horizontal_span,
                        0.5 * state.width_mm - 2.0,
                        18,
                    )
                grid_x, grid_y = np.meshgrid(x_values, y_values)
                source_points = _transform_rigid(
                    np.column_stack(
                        [grid_x.ravel(), grid_y.ravel()]
                    ),
                    state.inverse,
                )
                features = (
                    _sample_lab(lab, source_points, px_per_mm)
                    - references[state.piece_id]
                ) / scale
                valid = _sample_mask(
                    safe_masks[state.piece_id],
                    source_points,
                    px_per_mm,
                )
                if np.count_nonzero(valid) < 36:
                    side_values.append(0.0)
                    continue
                ink = np.clip(
                    (
                        np.linalg.norm(features, axis=1)
                        - 0.35
                    )
                    / 1.20,
                    0.0,
                    1.0,
                )
                side_values.append(float(np.mean(ink[valid])))
            strengths[(state_key, edge_name)] = (
                side_values[0],
                side_values[1],
            )
    return strengths


def _card_strip_layout(
    sequence: Sequence[Tuple[int, int]],
    states: Dict[Tuple[int, int], _CardStripState],
    pieces_by_id: Dict[int, PieceObservation],
) -> Tuple[Dict[int, np.ndarray], List[np.ndarray]]:
    total_height = sum(states[key].height_mm for key in sequence)
    cursor_y = -0.5 * total_height
    transforms: Dict[int, np.ndarray] = {}
    polygons = []
    for state_key in sequence:
        state = states[state_key]
        center_y = cursor_y + 0.5 * state.height_mm
        placement = rigid_matrix(
            0.0, np.array([0.0, center_y], dtype=np.float64)
        )
        matrix = placement @ state.transform
        transforms[state.piece_id] = matrix
        polygons.append(
            _transform_rigid(
                pieces_by_id[state.piece_id].polygon_mm, matrix
            )
        )
        cursor_y += state.height_mm
    return transforms, polygons


def _solve_card_strips(
    pieces: Sequence[PieceObservation],
    paper_bgr: np.ndarray,
    config: VisionConfig,
) -> AssemblySolution:
    """Reassemble four full-width card strips from directed seam affinities."""
    pieces_by_id = {piece.piece_id: piece for piece in pieces}
    states = _card_strip_states(pieces)
    lab, safe_masks, references = _prepare_texture_context(
        pieces, paper_bgr
    )
    profiles, shifts = _card_strip_edge_profiles(
        states,
        lab,
        safe_masks,
        references,
        config.px_per_mm,
        boundary_reconstructed=any(
            piece.boundary_reconstructed for piece in pieces
        ),
    )
    normalized, _ = _card_strip_pair_costs(
        states, profiles, shifts
    )
    outer_strengths = _card_strip_outer_corner_strengths(
        states,
        lab,
        safe_masks,
        references,
        config.px_per_mm,
    )

    best_outgoing = {
        state_a: min(
            (
                normalized[(state_a, state_b)],
                state_b,
            )
            for state_b in states
            if state_a[0] != state_b[0]
        )[1]
        for state_a in states
    }
    best_incoming = {
        state_b: min(
            (
                normalized[(state_a, state_b)],
                state_a,
            )
            for state_a in states
            if state_a[0] != state_b[0]
        )[1]
        for state_b in states
    }

    candidates = []
    explored = 0
    piece_ids = sorted(pieces_by_id)
    for order in itertools.permutations(piece_ids):
        for flips in itertools.product((0, 1), repeat=4):
            explored += 1
            sequence = tuple(zip(order, flips))
            adjacent_pairs = list(zip(sequence, sequence[1:]))
            best_buddy_count = sum(
                best_outgoing[state_a] == state_b
                and best_incoming[state_b] == state_a
                for state_a, state_b in adjacent_pairs
            )
            seam_cost = sum(
                normalized[pair] for pair in adjacent_pairs
            ) + 0.20 * (3 - best_buddy_count)

            top_left, top_right = outer_strengths[
                (sequence[0], "top")
            ]
            bottom_left, bottom_right = outer_strengths[
                (sequence[-1], "bottom")
            ]
            corner_strengths = (
                top_left,
                top_right,
                bottom_right,
                bottom_left,
            )
            diagonal_support = max(
                min(corner_strengths[0], corner_strengths[2]),
                min(corner_strengths[1], corner_strengths[3]),
            )
            adjacent_support = max(
                min(corner_strengths[0], corner_strengths[1]),
                min(corner_strengths[1], corner_strengths[2]),
                min(corner_strengths[2], corner_strengths[3]),
                min(corner_strengths[3], corner_strengths[0]),
            )
            # Standard playing cards repeat the corner index under a
            # half-turn.  This is a semantic validation cue, not a substitute
            # for the three measured seam compatibilities.
            semantic_cost = (
                -7.00 * diagonal_support
                + 4.00 * adjacent_support
            )
            transforms, polygons = _card_strip_layout(
                sequence, states, pieces_by_id
            )
            try:
                metrics = _raster_metrics(polygons, config)
            except SolveError:
                continue
            if not _metrics_are_valid(metrics, config):
                continue
            geometry_score = _geometry_score(
                metrics, config, polygons
            )
            ranking_cost = (
                seam_cost
                + semantic_cost
                + 0.10 * geometry_score
            )
            candidates.append(
                (
                    ranking_cost,
                    sequence,
                    transforms,
                    metrics,
                    geometry_score,
                    best_buddy_count,
                    diagonal_support,
                    adjacent_support,
                )
            )
    if not candidates:
        raise SolveError(
            "directed card-strip reconstruction found no valid rectangle"
        )

    candidates.sort(key=lambda item: item[0])
    distinct = []
    seen = set()
    for candidate in candidates:
        sequence = candidate[1]
        half_turn_equivalent = tuple(
            (piece_id, 1 - flipped)
            for piece_id, flipped in reversed(sequence)
        )
        key = min(sequence, half_turn_equivalent)
        if key in seen:
            continue
        seen.add(key)
        distinct.append(candidate)
    best = distinct[0]
    gap = (
        (distinct[1][0] - best[0]) / 3.0
        if len(distinct) > 1
        else 0.0
    )
    _, _, transforms, metrics, geometry_score, buddy_count, diagonal, adjacent = (
        best
    )
    margin = max(0.0, float(gap))
    if (
        (
            buddy_count < 2
            and diagonal < 0.12
        )
        or (
            diagonal < 0.035
            and buddy_count < 3
        )
        or (
            diagonal <= adjacent + 0.02
            and buddy_count < 3
        )
    ):
        margin = 0.0

    texture_score = 0.10 * (best[0] / 3.0 - 1.0)
    total_score = (
        geometry_score
        + max(config.texture_weight, 0.60) * texture_score
    )
    print(
        "[vision] directed card-strip compatibility checked "
        f"{explored} face orders; best-buddies={buddy_count}/3, "
        f"diagonal-corner={diagonal:.3f}, margin={margin:.4f}",
        flush=True,
    )
    return AssemblySolution(
        transforms=transforms,
        metrics=metrics,
        geometry_score=geometry_score,
        texture_score=texture_score,
        total_score=total_score,
        explored_nodes=explored,
        valid_candidate_count=len(distinct),
        search_complete=True,
        score_margin=margin,
    )


def _quarter_cell_transform(
    piece: PieceObservation,
    cell_center: np.ndarray,
    flipped: bool,
) -> np.ndarray:
    rectangle = cv2.minAreaRect(
        np.asarray(piece.polygon_mm, dtype=np.float32)
    )
    source_center = np.asarray(rectangle[0], dtype=np.float64)
    box = cv2.boxPoints(rectangle).astype(np.float64)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    aligned_direction = edges[int(np.argmin(lengths))]
    angle = -math.atan2(
        float(aligned_direction[1]), float(aligned_direction[0])
    )
    rotation = rigid_matrix(angle, np.zeros(2, dtype=np.float64))
    matrix = rigid_matrix(
        angle,
        np.asarray(cell_center, dtype=np.float64)
        - rotation[:2, :2] @ source_center,
    )
    if not flipped:
        return matrix
    half_turn = rigid_matrix(
        math.pi, 2.0 * np.asarray(cell_center, dtype=np.float64)
    )
    return half_turn @ matrix


def _solve_equal_card_quarters(
    pieces: Sequence[PieceObservation],
    paper_bgr: np.ndarray,
    config: VisionConfig,
) -> AssemblySolution:
    """Enumerate the 48 non-global-equivalent quarter-card layouts."""
    piece_ids = sorted(piece.piece_id for piece in pieces)
    pieces_by_id = {piece.piece_id: piece for piece in pieces}
    dimensions = []
    for piece in pieces:
        rectangle = cv2.minAreaRect(
            np.asarray(piece.polygon_mm, dtype=np.float32)
        )
        dimensions.append(sorted(map(float, rectangle[1])))
    cell_width, cell_height = np.median(
        np.asarray(dimensions, dtype=np.float64), axis=0
    )
    cell_centers = [
        np.array([0.5 * cell_width, 0.5 * cell_height]),
        np.array([1.5 * cell_width, 0.5 * cell_height]),
        np.array([1.5 * cell_width, 1.5 * cell_height]),
        np.array([0.5 * cell_width, 1.5 * cell_height]),
    ]
    texture_context = _prepare_texture_context(
        pieces, paper_bgr
    )

    candidates = []
    explored = 0
    anchor_id = piece_ids[0]
    other_ids = piece_ids[1:]
    for anchor_cell in range(4):
        remaining_cells = [
            cell for cell in range(4) if cell != anchor_cell
        ]
        for assignment in itertools.permutations(other_ids):
            cell_by_piece = {
                anchor_id: anchor_cell,
                **dict(zip(assignment, remaining_cells)),
            }
            for flips in itertools.product((False, True), repeat=3):
                explored += 1
                flip_by_piece = {
                    anchor_id: False,
                    **dict(zip(other_ids, flips)),
                }
                transforms = {
                    piece_id: _quarter_cell_transform(
                        pieces_by_id[piece_id],
                        cell_centers[cell_by_piece[piece_id]],
                        flip_by_piece[piece_id],
                    )
                    for piece_id in piece_ids
                }
                polygons = [
                    _transform_rigid(
                        pieces_by_id[piece_id].polygon_mm,
                        transforms[piece_id],
                    )
                    for piece_id in piece_ids
                ]
                texture_score = _texture_score(
                    pieces,
                    transforms,
                    paper_bgr,
                    config.px_per_mm,
                    config,
                    texture_context=texture_context,
                )
                candidates.append(
                    (texture_score, transforms, polygons)
                )

    leaves: List[_Leaf] = []
    quarter_texture_weight = max(config.texture_weight, 0.60)
    for texture_score, transforms, polygons in sorted(
        candidates, key=lambda item: item[0]
    ):
        try:
            metrics = _raster_metrics(polygons, config)
        except SolveError:
            continue
        if not _metrics_are_valid(metrics, config):
            continue
        geometry_score = _geometry_score(
            metrics, config, polygons
        )
        leaves.append(
            _Leaf(
                transforms=transforms,
                metrics=metrics,
                geometry_score=geometry_score,
                texture_score=texture_score,
                total_score=(
                    geometry_score
                    + quarter_texture_weight * texture_score
                ),
            )
        )
        if len(leaves) >= 12:
            break
    if not leaves:
        raise SolveError(
            "equal-quarter card enumeration found no valid rectangle"
        )
    leaves.sort(key=lambda leaf: leaf.total_score)
    distinct: List[_Leaf] = []
    for leaf in leaves:
        if any(
            _leaves_are_globally_equivalent(
                leaf,
                previous,
                pieces_by_id,
                anchor_id,
            )
            for previous in distinct
        ):
            continue
        distinct.append(leaf)
    best = distinct[0]
    margin = (
        distinct[1].total_score - best.total_score
        if len(distinct) > 1
        else 0.0
    )
    # A rectangle alone is not a valid answer in this mode.  Blank or
    # internally inconsistent faces deliberately become LOW_MARGIN later.
    if best.texture_score >= 0.20:
        margin = 0.0
    print(
        "[vision] equal-quarter card checked "
        f"{explored} face layouts; best texture={best.texture_score:.4f}, "
        f"margin={margin:.4f}",
        flush=True,
    )
    return AssemblySolution(
        transforms=best.transforms,
        metrics=best.metrics,
        geometry_score=best.geometry_score,
        texture_score=best.texture_score,
        total_score=best.total_score,
        explored_nodes=explored,
        valid_candidate_count=len(distinct),
        search_complete=True,
        score_margin=margin,
    )


def _prufer_tree_edges(
    nodes: Sequence[int], sequence: Sequence[int]
) -> List[Tuple[int, int]]:
    """Decode one labelled tree; four pieces have only 4^(4-2)=16 trees."""
    degree = {node: 1 for node in nodes}
    for node in sequence:
        degree[node] += 1
    edges: List[Tuple[int, int]] = []
    for node in sequence:
        leaf = min(item for item in nodes if degree[item] == 1)
        edges.append((leaf, node))
        degree[leaf] -= 1
        degree[node] -= 1
    remaining = sorted(node for node in nodes if degree[node] == 1)
    if len(remaining) == 2:
        edges.append((remaining[0], remaining[1]))
    return edges


def _orient_tree_edges(
    nodes: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    root: int,
) -> List[Tuple[int, int]]:
    adjacency = {node: [] for node in nodes}
    for first, second in edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    ordered: List[Tuple[int, int]] = []
    visited = {root}
    queue = [root]
    while queue:
        parent = queue.pop(0)
        for child in sorted(adjacency[parent]):
            if child in visited:
                continue
            visited.add(child)
            queue.append(child)
            ordered.append((parent, child))
    return ordered


def _spanning_tree_layouts(
    pieces_by_id: Dict[int, PieceObservation],
    pair_cache: Dict[Tuple[int, int], List[np.ndarray]],
    total_piece_area: float,
    config: VisionConfig,
    search_started: float,
    time_limit_seconds: Optional[float],
) -> Tuple[
    List[Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]],
    int,
    bool,
]:
    """Compose redundant pair mates through every four-piece contact tree.

    This is the small-N version of graph-based global reassembly: local C1/C2
    matches remain redundant, while the complete assembly—not a single early
    seam—decides which hypotheses survive.  The tree stage is deterministic
    and bounded before the more expensive online T-junction DFS.
    """
    nodes = sorted(pieces_by_id)
    if not 2 <= len(nodes) <= 4:
        return [], 0, False
    root = max(
        nodes,
        key=lambda piece_id: pieces_by_id[piece_id].area_mm2,
    )
    # Four relations per tree edge cover the best C1/C2 alternatives while
    # reducing the four-piece graph from 3456 to at most 1024 states.  On
    # MaixCAM2 this preserves most of the six-second budget for online
    # T-junction refinement.
    relation_depth = min(4, config.solver_max_pair_candidates)
    retained_limit = min(96, config.solver_max_leaf_candidates)
    retained: Dict[
        Tuple[Tuple[int, ...], ...],
        Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]],
    ] = {}
    # Different labelled trees repeatedly compose the same one-, two-, and
    # three-edge rigid states.  Polygon overlap is relatively expensive on
    # MaixCAM2, so validate each quantized partial pose once while preserving
    # the same four relations per edge and the same final candidate order.
    incremental_valid_cache: Dict[
        Tuple[int, Tuple[Tuple[int, ...], ...]],
        bool,
    ] = {}
    explored = 0
    timed_out = False
    sequences = itertools.product(nodes, repeat=max(0, len(nodes) - 2))
    for sequence in sequences:
        undirected = _prufer_tree_edges(nodes, sequence)
        ordered = _orient_tree_edges(nodes, undirected, root)
        relation_lists = [
            pair_cache[(parent, child)][:relation_depth]
            for parent, child in ordered
        ]
        if any(not relations for relations in relation_lists):
            continue
        for selected in itertools.product(*relation_lists):
            if explored >= config.solver_max_nodes:
                timed_out = True
                break
            explored += 1
            if (
                time_limit_seconds is not None
                and explored % 128 == 0
                and py_time.monotonic() - search_started
                >= time_limit_seconds
            ):
                timed_out = True
                break
            transforms = {
                root: np.eye(3, dtype=np.float64)
            }
            polygons = {
                root: np.asarray(
                    pieces_by_id[root].polygon_mm,
                    dtype=np.float64,
                )
            }
            valid = True
            for (parent, child), relative in zip(ordered, selected):
                matrix = transforms[parent] @ relative
                transforms[child] = matrix
                polygons[child] = _transform_rigid(
                    pieces_by_id[child].polygon_mm, matrix
                )
                partial_key = (child, _solution_key(transforms))
                partial_valid = incremental_valid_cache.get(partial_key)
                if partial_valid is None:
                    partial_valid = _incremental_valid(
                        polygons,
                        child,
                        total_piece_area,
                        config,
                    )
                    incremental_valid_cache[partial_key] = partial_valid
                if not partial_valid:
                    valid = False
                    break
            if not valid:
                continue
            key = _solution_key(transforms)
            priority = _partial_priority(
                polygons, total_piece_area, config
            )
            previous = retained.get(key)
            if previous is None or priority < previous[0]:
                retained[key] = (priority, transforms, polygons)
        if timed_out:
            break
    layouts = sorted(retained.values(), key=lambda item: item[0])[
        :retained_limit
    ]
    return layouts, explored, timed_out


def _solution_from_graph_layouts(
    layouts: Sequence[
        Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
    ],
    pieces: Sequence[PieceObservation],
    paper_bgr: np.ndarray,
    config: VisionConfig,
    use_texture: bool,
    explored_nodes: int,
) -> Optional[AssemblySolution]:
    leaves: List[_Leaf] = []
    seen = set()
    for _, transforms, polygons_by_id in layouts:
        key = _solution_key(transforms)
        if key in seen:
            continue
        seen.add(key)
        polygons = list(polygons_by_id.values())
        try:
            metrics = _raster_metrics(polygons, config)
        except SolveError:
            continue
        if not _metrics_are_valid(metrics, config):
            continue
        geometry_score = _geometry_score(metrics, config, polygons)
        texture_score = (
            _texture_score(
                pieces,
                transforms,
                paper_bgr,
                config.px_per_mm,
                config,
            )
            if use_texture
            else 0.0
        )
        leaves.append(
            _Leaf(
                transforms={
                    piece_id: matrix.copy()
                    for piece_id, matrix in transforms.items()
                },
                metrics=metrics,
                geometry_score=geometry_score,
                texture_score=texture_score,
                total_score=(
                    geometry_score
                    + config.texture_weight * texture_score
                ),
            )
        )
    if not leaves:
        return None
    leaves.sort(key=lambda leaf: leaf.total_score)
    distinct: List[_Leaf] = []
    pieces_by_id = {piece.piece_id: piece for piece in pieces}
    for leaf in leaves:
        if any(
            _leaves_are_measurement_equivalent(
                leaf, previous, pieces_by_id
            )
            for previous in distinct
        ):
            continue
        distinct.append(leaf)
    best = distinct[0]
    margin = (
        distinct[1].total_score - best.total_score
        if len(distinct) > 1
        else None
    )
    return AssemblySolution(
        transforms=best.transforms,
        metrics=best.metrics,
        geometry_score=best.geometry_score,
        texture_score=best.texture_score,
        total_score=best.total_score,
        explored_nodes=explored_nodes,
        valid_candidate_count=len(distinct),
        search_complete=True,
        score_margin=margin,
    )


def solve_puzzle(
    pieces: Sequence[PieceObservation],
    paper_bgr: np.ndarray,
    config: VisionConfig,
    use_texture: bool = True,
    recovery_mode: bool = False,
    template_pieces: Optional[Sequence[PieceObservation]] = None,
) -> AssemblySolution:
    if not 1 <= len(pieces) <= 4:
        raise SolveError("solver supports 1..4 pieces")
    if config.solver_mode == "auto" and not recovery_mode:
        from .template_solver import solve_trained_template_bank

        try:
            print(
                "[vision] trying trained custom template bank",
                flush=True,
            )
            measured_pieces = (
                template_pieces
                if template_pieces is not None
                else pieces
            )
            try:
                return solve_trained_template_bank(
                    measured_pieces,
                    paper_bgr,
                    config,
                    use_texture=use_texture,
                )
            except SolveError:
                if measured_pieces is pieces:
                    raise
                print(
                    "[vision] measured-contour template mismatch; "
                    "retrying quantized contour",
                    flush=True,
                )
                return solve_trained_template_bank(
                    pieces,
                    paper_bgr,
                    config,
                    use_texture=use_texture,
                )
        except SolveError as error:
            print(
                "[vision] trained custom template mismatch; "
                f"continuing normal solver: {error}",
                flush=True,
            )
    if (
        use_texture
        and config.solver_mode != "figure2"
        and _is_equal_card_quarters(pieces)
    ):
        print(
            "[vision] using equal-quarter playing-card face solver",
            flush=True,
        )
        return _solve_equal_card_quarters(
            pieces, paper_bgr, config
        )
    if (
        use_texture
        and config.solver_mode != "figure2"
        and _is_card_strip_partition(pieces)
    ):
        print(
            "[vision] using directed card-strip compatibility solver",
            flush=True,
        )
        try:
            return _solve_card_strips(
                pieces, paper_bgr, config
            )
        except SolveError as error:
            print(
                "[vision] directed card-strip solver could not validate "
                f"the face; using generic solver: {error}",
                flush=True,
            )
    if config.solver_mode == "figure2":
        from .template_solver import solve_figure2_template

        print("[vision] using bounded Figure-2 template solver", flush=True)
        return solve_figure2_template(pieces, config)
    if (
        config.solver_mode == "auto"
        and len(pieces) == 4
        and not recovery_mode
    ):
        from .template_solver import solve_figure2_template

        try:
            print(
                "[vision] trying bounded Figure-2 template solver",
                flush=True,
            )
            return solve_figure2_template(pieces, config)
        except SolveError as error:
            print(
                f"[vision] Figure-2 template mismatch; using generic solver: "
                f"{error}",
                flush=True,
            )

    print(
        "[vision] using bounded generic C1/C2 mating solver"
        + (" (recovery coverage)" if recovery_mode else ""),
        flush=True,
    )
    search_started = py_time.monotonic()
    time_limit_seconds = (
        None
        if config.solver_time_limit_ms == 0
        else config.solver_time_limit_ms / 1000.0
    )
    if len(pieces) == 1:
        metrics = _raster_metrics([pieces[0].polygon_mm], config)
        if (
            len(pieces[0].polygon_mm) != 4
            or not _metrics_are_valid(metrics, config)
        ):
            raise SolveError("the single piece is not a valid target rectangle")
        transform = {pieces[0].piece_id: np.eye(3, dtype=np.float64)}
        score = _geometry_score(metrics, config, [pieces[0].polygon_mm])
        return AssemblySolution(
            transforms=transform,
            metrics=metrics,
            geometry_score=score,
            texture_score=0.0,
            total_score=score,
            explored_nodes=1,
            valid_candidate_count=1,
            search_complete=True,
            score_margin=None,
        )

    pieces_by_id = {piece.piece_id: piece for piece in pieces}
    pair_cache: Dict[Tuple[int, int], List[np.ndarray]] = {}
    identity = np.eye(3, dtype=np.float64)
    for placed_id in pieces_by_id:
        for new_id in pieces_by_id:
            if placed_id == new_id:
                continue
            pair_cache[(placed_id, new_id)] = _candidate_transforms(
                pieces_by_id[new_id],
                pieces_by_id,
                {placed_id: identity},
                config,
                recovery_mode=recovery_mode,
            )
            if (
                time_limit_seconds is not None
                and py_time.monotonic() - search_started
                >= time_limit_seconds
            ):
                raise SolveError(
                    "puzzle search time limit reached while preparing seam "
                    "candidates; check the measured contours and divider"
                )
    # C2 ranking is direction-dependent under contour noise.  A physically
    # valid rigid mate found from A toward B must remain available when the
    # search has already placed B, so merge every reverse pose through its
    # inverse before starting the graph search.
    piece_ids = sorted(pieces_by_id)
    for index, piece_a in enumerate(piece_ids):
        for piece_b in piece_ids[index + 1 :]:
            forward = pair_cache[(piece_a, piece_b)]
            reverse = pair_cache[(piece_b, piece_a)]
            merged_forward: Dict[
                Tuple[int, int, int], np.ndarray
            ] = {
                _pose_key(matrix): matrix for matrix in forward
            }
            merged_reverse: Dict[
                Tuple[int, int, int], np.ndarray
            ] = {
                _pose_key(matrix): matrix for matrix in reverse
            }
            for matrix in reverse:
                inverse = np.linalg.inv(matrix)
                merged_forward.setdefault(_pose_key(inverse), inverse)
            for matrix in forward:
                inverse = np.linalg.inv(matrix)
                merged_reverse.setdefault(_pose_key(inverse), inverse)
            pair_cache[(piece_a, piece_b)] = list(
                merged_forward.values()
            )[: config.solver_max_pair_candidates]
            pair_cache[(piece_b, piece_a)] = list(
                merged_reverse.values()
            )[: config.solver_max_pair_candidates]
    pair_pose_count = sum(len(items) for items in pair_cache.values())
    pair_prepare_ms = int(
        round((py_time.monotonic() - search_started) * 1000.0)
    )
    print(
        f"[vision] C1/C2 graph has {pair_pose_count} pair poses "
        f"({pair_prepare_ms} ms)",
        flush=True,
    )

    total_piece_area = sum(piece.area_mm2 for piece in pieces)
    printed_card_partition = use_texture and len(pieces) in (3, 4)
    printed_four_piece = use_texture and len(pieces) == 4
    printed_four_quadrilaterals = (
        printed_four_piece
        and all(len(piece.polygon_mm) == 4 for piece in pieces)
    )
    printed_card_strips = (
        printed_four_piece and _is_card_strip_partition(pieces)
    )
    if printed_four_piece and not recovery_mode:
        # Repeated real-card runs retained no final rectangle from the broad
        # tree but consumed most of the embedded deadline.  Card cuts need the
        # later split-edge/T-junction path and texture scoring more urgently.
        print(
            "[vision] printed-card mode skips the broad global tree; "
            "reserving budget for pair/T-junction search",
            flush=True,
        )
        graph_layouts = []
        graph_nodes = 0
        graph_timed_out = False
    elif recovery_mode:
        # The primary printed-card search skipped this broad graph, while the
        # primary solid-colour search already exhausted it.  Recovery covers
        # the missing schedule in either case instead of repeating attempt 1.
        if printed_four_piece:
            graph_layouts, graph_nodes, graph_timed_out = (
                _spanning_tree_layouts(
                    pieces_by_id,
                    pair_cache,
                    total_piece_area,
                    config,
                    search_started,
                    time_limit_seconds,
                )
            )
        else:
            graph_layouts = []
            graph_nodes = 0
            graph_timed_out = False
            print(
                "[vision] recovery search skips the already exhausted "
                "global tree",
                flush=True,
            )
    else:
        graph_layouts, graph_nodes, graph_timed_out = (
            _spanning_tree_layouts(
                pieces_by_id,
                pair_cache,
                total_piece_area,
                config,
                search_started,
                time_limit_seconds,
            )
        )
    print(
        f"[vision] global mate graph checked {graph_nodes} tree states; "
        f"kept {len(graph_layouts)} complete layouts",
        flush=True,
    )
    if graph_timed_out:
        deadline_reached = (
            time_limit_seconds is not None
            and py_time.monotonic() - search_started
            >= time_limit_seconds
        )
        if deadline_reached:
            raise SolveError(
                "puzzle search time limit reached in the global mate graph; "
                "check the measured contours and divider"
            )
        raise SolveError(
            f"puzzle search stopped after {graph_nodes} search "
            "nodes (search limit reached)"
        )
    graph_solution = _solution_from_graph_layouts(
        graph_layouts,
        pieces,
        paper_bgr,
        config,
        use_texture,
        graph_nodes,
    )
    if graph_solution is not None:
        if (
            not use_texture
            and graph_solution.geometry_score <= 0.05
        ):
            print(
                "[vision] global mate graph found a high-confidence "
                "rectangle",
                flush=True,
            )
            return graph_solution
        print(
            "[vision] global mate graph retained a valid rectangle; "
            "checking T-junction refinements",
            flush=True,
        )

    def cached_candidates(
        new_id: int,
        placed: Dict[int, np.ndarray],
        placed_polygons: Dict[int, np.ndarray],
    ) -> List[np.ndarray]:
        unique: Dict[Tuple[int, int, int], np.ndarray] = {}
        for placed_id, placed_matrix in placed.items():
            for relative in pair_cache[(placed_id, new_id)]:
                candidate = placed_matrix @ relative
                unique.setdefault(_pose_key(candidate), candidate)
        # Pair caches cannot see a vertex from a third piece that terminates in
        # the middle of a long edge.  Once such an anchor exists in the current
        # partial assembly, generate those T-junction poses online.
        if len(placed) > 1:
            for candidate in _t_junction_transforms(
                pieces_by_id[new_id],
                placed_polygons,
                config,
            ):
                unique.setdefault(_pose_key(candidate), candidate)
        candidates = list(unique.values())
        candidate_cap = 32 if recovery_mode else 24
        if (
            printed_card_partition
            and (
                recovery_mode
                or config.fixed_paper_quad_px is not None
            )
            and len(placed) >= 2
            and len(candidates) > candidate_cap
        ):
            # Once two card fragments are fixed, rank the remaining poses by
            # the compactness of the resulting partial silhouette.  Pair-cache
            # insertion order can put a true reverse-direction mate near rank
            # 50; a fixed first-48 slice would therefore be unsafe.
            ranked = []
            for candidate in candidates:
                candidate_polygons = dict(placed_polygons)
                candidate_polygons[new_id] = _transform_rigid(
                    pieces_by_id[new_id].polygon_mm,
                    candidate,
                )
                ranked.append(
                    (
                        _unfinished_priority(
                            candidate_polygons,
                            total_piece_area,
                            config,
                            0,
                        ),
                        candidate,
                    )
                )
            return [
                candidate
                for _, candidate in sorted(
                    ranked, key=lambda item: item[0]
                )[:candidate_cap]
            ]
        return candidates

    leaves: List[_Leaf] = []
    leaf_keys = set()
    refinable_candidates: List[_Leaf] = []
    refinable_keys = set()
    attempted_refinement_keys = set()

    def retain_refinable_candidate(
        transforms: Dict[int, np.ndarray],
        metrics: AssemblyMetrics,
        polygons: Sequence[np.ndarray],
    ) -> bool:
        if not (
            printed_four_piece
            and (
                recovery_mode
                or config.fixed_paper_quad_px is not None
            )
            and not printed_card_strips
            and _metrics_are_refinable_printed_candidate(metrics, config)
        ):
            return False
        key = _solution_key(transforms)
        if key in refinable_keys:
            return False
        refinable_keys.add(key)
        refinable_candidates.append(
            _Leaf(
                transforms={
                    piece_id: matrix.copy()
                    for piece_id, matrix in transforms.items()
                },
                metrics=metrics,
                geometry_score=_geometry_score(
                    metrics, config, polygons
                ),
            )
        )
        refinable_candidates.sort(
            key=lambda candidate: candidate.geometry_score
        )
        del refinable_candidates[8:]
        return True

    def refine_retained_candidates(
        maximum_attempts: int,
        maximum_valid: int,
    ) -> int:
        attempts = 0
        added = 0
        for candidate in list(refinable_candidates):
            if attempts >= maximum_attempts or added >= maximum_valid:
                break
            source_key = _solution_key(candidate.transforms)
            if source_key in attempted_refinement_keys:
                continue
            attempted_refinement_keys.add(source_key)
            attempts += 1
            refined = _refine_printed_layout(
                candidate.transforms, pieces_by_id, config
            )
            if refined is None:
                continue
            key = _solution_key(refined.transforms)
            if key in leaf_keys:
                continue
            leaf_keys.add(key)
            leaves.append(refined)
            added += 1
            print(
                "[vision] locally refined a printed-card layout: "
                f"fill={refined.metrics.fill_ratio:.3f}, "
                f"side={refined.metrics.side_coverage_min:.3f}, "
                f"hole={refined.metrics.hole_ratio:.3f}",
                flush=True,
            )
        return added

    if graph_solution is not None:
        leaves.append(
            _Leaf(
                transforms={
                    piece_id: matrix.copy()
                    for piece_id, matrix in graph_solution.transforms.items()
                },
                metrics=graph_solution.metrics,
                geometry_score=graph_solution.geometry_score,
                texture_score=graph_solution.texture_score,
                total_score=graph_solution.total_score,
            )
        )
        leaf_keys.add(_solution_key(graph_solution.transforms))
    best_rejected_metrics: Optional[AssemblyMetrics] = None
    best_rejected_score = float("inf")
    explored_nodes = graph_nodes
    queue = []
    counter = 0
    seen_partial = set()

    # Start from the best measured two-piece seams instead of committing to a
    # single largest-piece anchor.  This is the bounded pair-seed stage used by
    # the senior solver: it keeps the correct card topology reachable when the
    # useful long-edge transform exists only in one direction.
    pair_seeds: Dict[
        Tuple[Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        Tuple[
            float,
            Dict[int, np.ndarray],
            Dict[int, np.ndarray],
            frozenset,
        ],
    ] = {}
    pair_seed_inputs = 0
    pair_seed_started = py_time.monotonic()
    for (fixed_id, new_id), relatives in pair_cache.items():
        for relative_rank, relative in enumerate(relatives):
            pair_seed_inputs += 1
            if fixed_id < new_id:
                transforms = {
                    fixed_id: identity,
                    new_id: relative,
                }
            else:
                inverse = np.linalg.inv(relative)
                transforms = {
                    fixed_id: inverse,
                    new_id: identity,
                }
            state_key = (
                tuple(sorted(transforms)),
                _solution_key(transforms),
            )
            # Reverse-direction caches describe the same normalized pair.
            # Skip before polygon overlap and contact scoring on the board.
            if state_key in pair_seeds:
                continue
            polygons = {
                piece_id: _transform_rigid(
                    pieces_by_id[piece_id].polygon_mm, matrix
                )
                for piece_id, matrix in transforms.items()
            }
            if not _incremental_valid(
                polygons,
                max(fixed_id, new_id),
                total_piece_area,
                config,
            ):
                continue
            priority = (
                _unfinished_priority(
                    polygons,
                    total_piece_area,
                    config,
                    relative_rank,
                )
                if use_texture
                else _partial_priority(
                    polygons,
                    total_piece_area,
                    config,
                )
            )
            pair_seeds[state_key] = (
                priority,
                transforms,
                polygons,
                frozenset(set(pieces_by_id) - set(transforms)),
            )
    print(
        f"[vision] pair seeds kept {len(pair_seeds)} unique from "
        f"{pair_seed_inputs} directed poses "
        f"({round((py_time.monotonic() - pair_seed_started) * 1000)} ms)",
        flush=True,
    )

    # With four measured quadrilaterals, one hypothesis from each of the six
    # unordered piece pairs preserves full pair coverage.  Extra variants
    # only repeat expensive board-side ranking before refinement starts.
    if recovery_mode:
        seed_capacity = 48
    elif printed_card_strips:
        seed_capacity = 24
    elif printed_four_quadrilaterals:
        seed_capacity = 6
    elif printed_four_piece:
        seed_capacity = 24
    else:
        seed_capacity = 12
    seed_limit = min(
        seed_capacity, config.solver_max_pair_candidates
    )
    seed_buckets: Dict[
        Tuple[int, int],
        List[
            Tuple[
                Tuple[Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                Tuple[
                    float,
                    Dict[int, np.ndarray],
                    Dict[int, np.ndarray],
                    frozenset,
                ],
            ]
        ],
    ] = {}
    for state_key, seed in sorted(
        pair_seeds.items(), key=lambda item: item[1][0]
    ):
        pair_key = tuple(sorted(seed[1]))
        seed_buckets.setdefault(pair_key, []).append((state_key, seed))

    # Preserve at least one strong hypothesis for every unordered piece pair.
    # A global top-12 list can otherwise be monopolised by several poses of
    # the same attractive-but-wrong seam and make a valid topology unreachable.
    selected_seeds = []
    bucket_depth = 0
    while len(selected_seeds) < seed_limit:
        added = False
        for pair_key in sorted(seed_buckets):
            bucket = seed_buckets[pair_key]
            if bucket_depth < len(bucket):
                selected_seeds.append(bucket[bucket_depth])
                added = True
                if len(selected_seeds) >= seed_limit:
                    break
        if not added:
            break
        bucket_depth += 1

    selected_rerank_started = py_time.monotonic()
    reranked_seed_buckets: Dict[
        Tuple[int, int],
        List[
            Tuple[
                Tuple[Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                Tuple[
                    float,
                    Dict[int, np.ndarray],
                    Dict[int, np.ndarray],
                    frozenset,
                ],
            ]
        ],
    ] = {}
    cycle_supported_seeds = 0
    for state_key, seed in selected_seeds:
        priority, transforms, polygons, unplaced = seed
        pair_key = tuple(sorted(transforms))
        if recovery_mode and len(pieces_by_id) == 4:
            fixed_id, moving_id = pair_key
            relative = (
                np.linalg.inv(transforms[fixed_id])
                @ transforms[moving_id]
            )
            cycle_support = _pair_cycle_support(
                fixed_id,
                moving_id,
                relative,
                pair_cache,
                piece_ids,
            )
            # This is deliberately a small bonus, not a filter.  A valid
            # contact graph may be a tree and therefore have no closing loop.
            priority -= 0.025 * cycle_support
            if cycle_support >= 0.5:
                cycle_supported_seeds += 1
        if printed_four_piece:
            # Re-rank only the bounded queue by straight outer chains.  Do
            # not call _partial_priority here: its pair-contact tracing made
            # 24 seeds consume several seconds on MaixCAM2.
            boundary_support, corner_support = (
                _rectangle_feature_scores(
                    list(polygons.values()),
                    max(
                        1.5,
                        min(
                            2.5,
                            config.solver_collinear_tolerance_mm,
                        ),
                    ),
                )
            )
            priority -= (
                0.18 * boundary_support + 0.02 * corner_support
            )
        reranked_seed_buckets.setdefault(pair_key, []).append(
            (
                state_key,
                (
                    priority,
                    transforms,
                    polygons,
                    unplaced,
                ),
            )
        )
        counter += 1
        heapq.heappush(
            queue,
            (
                priority,
                counter,
                transforms,
                polygons,
                unplaced,
            ),
        )
        seen_partial.add(state_key)
    for bucket in reranked_seed_buckets.values():
        bucket.sort(key=lambda item: item[1][0])
    if printed_four_piece:
        loop_message = (
            f"; {cycle_supported_seeds} have pose-loop support"
            if recovery_mode
            else ""
        )
        print(
            "[vision] selected card seeds reranked in "
            f"{round((py_time.monotonic() - selected_rerank_started) * 1000)} "
            f"ms{loop_message}",
            flush=True,
        )

    # Senior-style 2+2 path: independently form two disjoint strong pairs,
    # then mate the pair groups rigidly through any cross-pair relation.  This
    # reaches four-piece layouts that a one-piece-at-a-time beam can prune,
    # while keeping the candidate set bounded (three pair partitions, two
    # variants per pair, and at most 24 completed states retained).
    completed_pair_states: Dict[
        Tuple[Tuple[int, ...], ...],
        Tuple[
            float,
            Dict[int, np.ndarray],
            Dict[int, np.ndarray],
        ],
    ] = {}
    # Reuse the straight-side/corner evidence already paid for by the bounded
    # card queue.  It promotes the second true seam variant without rescoring
    # every raw pair seed on the embedded CPU.
    two_pair_seed_buckets = (
        reranked_seed_buckets if printed_four_piece else seed_buckets
    )
    pair_keys = sorted(two_pair_seed_buckets)
    two_pair_timed_out = False
    two_pair_checks = 0
    two_pair_seen = set()
    two_pair_started = py_time.monotonic()
    pair_variant_depth = 4 if recovery_mode else 2
    cross_relation_depth = (
        8 if recovery_mode else (5 if printed_four_piece else 8)
    )
    for pair_index, pair_a in enumerate(pair_keys):
        if two_pair_timed_out:
            break
        set_a = set(pair_a)
        for pair_b in pair_keys[pair_index + 1 :]:
            if two_pair_timed_out:
                break
            if set_a & set(pair_b) or set_a | set(pair_b) != set(pieces_by_id):
                continue
            for _, seed_a in two_pair_seed_buckets[pair_a][
                :pair_variant_depth
            ]:
                if two_pair_timed_out:
                    break
                for _, seed_b in two_pair_seed_buckets[pair_b][
                    :pair_variant_depth
                ]:
                    if two_pair_timed_out:
                        break
                    transforms_a = seed_a[1]
                    transforms_b = seed_b[1]
                    for fixed_id in pair_a:
                        if two_pair_timed_out:
                            break
                        for moving_id in pair_b:
                            if two_pair_timed_out:
                                break
                            inverse_moving = np.linalg.inv(
                                transforms_b[moving_id]
                            )
                            # The global tree already evaluated the broad
                            # pair graph.  The 2+2 fallback only needs the
                            # strongest few cross-pair relations.
                            for relative in pair_cache[
                                (fixed_id, moving_id)
                            ][:cross_relation_depth]:
                                two_pair_checks += 1
                                if (
                                    time_limit_seconds is not None
                                    and two_pair_checks % 16 == 0
                                    and py_time.monotonic() - search_started
                                    >= time_limit_seconds
                                ):
                                    two_pair_timed_out = True
                                    break
                                desired_moving = (
                                    transforms_a[fixed_id] @ relative
                                )
                                group_transform = (
                                    desired_moving
                                    @ inverse_moving
                                )
                                transforms = {
                                    piece_id: matrix.copy()
                                    for piece_id, matrix in transforms_a.items()
                                }
                                for piece_id, matrix in transforms_b.items():
                                    transforms[piece_id] = (
                                        group_transform @ matrix
                                    )
                                key = _solution_key(transforms)
                                if key in two_pair_seen:
                                    continue
                                two_pair_seen.add(key)
                                polygons = {
                                    piece_id: _transform_rigid(
                                        pieces_by_id[piece_id].polygon_mm,
                                        matrix,
                                    )
                                    for piece_id, matrix in transforms.items()
                                }
                                if not _pair_groups_incrementally_valid(
                                    polygons,
                                    pair_b,
                                    total_piece_area,
                                    config,
                                ):
                                    continue
                                priority = _partial_priority(
                                    polygons, total_piece_area, config
                                )
                                previous = completed_pair_states.get(key)
                                if previous is None or priority < previous[0]:
                                    completed_pair_states[key] = (
                                        priority,
                                        transforms,
                                        polygons,
                                    )
    print(
        f"[vision] 2+2 checked {two_pair_checks} cross poses; "
        f"{len(two_pair_seen)} unique, "
        f"{len(completed_pair_states)} retained "
        f"({round((py_time.monotonic() - two_pair_started) * 1000)} ms)",
        flush=True,
    )
    if two_pair_timed_out:
        if graph_solution is not None:
            return graph_solution
        raise SolveError(
            "puzzle search time limit reached while preparing bounded "
            "two-pair seeds; check the measured contours and divider"
        )

    sorted_pair_states = sorted(
        completed_pair_states.values(), key=lambda item: item[0]
    )
    if printed_four_piece:
        # Geometry ranking and queue compactness ranking are intentionally
        # different.  Scan the whole bounded 2+2 pool (normally only tens of
        # states), otherwise a correct rich-face card can sit just outside the
        # 24 compactness-ranked queue entries and force thousands of T-junction
        # expansions on the embedded CPU.
        for _, transforms, polygons in sorted_pair_states:
            try:
                pair_metrics = _raster_metrics(
                    list(polygons.values()), config
                )
            except SolveError:
                pair_metrics = None
            if (
                pair_metrics is not None
                and not _metrics_are_valid(pair_metrics, config)
            ):
                retain_refinable_candidate(
                    transforms,
                    pair_metrics,
                    list(polygons.values()),
                )
    for priority, transforms, polygons in sorted_pair_states[:24]:
        state_key = (
            tuple(sorted(transforms)),
            _solution_key(transforms),
        )
        if state_key in seen_partial:
            continue
        counter += 1
        heapq.heappush(
            queue,
            (
                priority,
                counter,
                transforms,
                polygons,
                frozenset(),
            ),
        )
        seen_partial.add(state_key)
    if recovery_mode and refinable_candidates and not leaves:
        refined_count = refine_retained_candidates(
            maximum_attempts=8,
            maximum_valid=3,
        )
        if refined_count and (
            recovery_mode
            or max(leaf.metrics.fill_ratio for leaf in leaves) >= 0.95
        ):
            # The retained 2+2 states already provide independent topology
            # alternatives.  Avoid expanding a large T-junction parent after
            # local refinement has produced strict valid rectangles.
            queue.clear()

    if not queue and not leaves:
        largest_area = max(piece.area_mm2 for piece in pieces)
        anchor = min(
            piece.piece_id
            for piece in pieces
            if piece.area_mm2 >= 0.99 * largest_area
        )
        initial = {anchor: identity}
        initial_polygons = {
            anchor: np.asarray(
                pieces_by_id[anchor].polygon_mm, dtype=np.float64
            )
        }
        remaining = frozenset(set(pieces_by_id) - {anchor})
        state_key = (
            tuple(sorted(initial)),
            _solution_key(initial),
        )
        heapq.heappush(
            queue,
            (
                _partial_priority(
                    initial_polygons, total_piece_area, config
                ),
                counter,
                initial,
                initial_polygons,
                remaining,
            ),
        )
        seen_partial.add(state_key)
    explored_nodes = len(queue)
    print(
        f"[vision] seed queue kept {len(selected_seeds)} pairs + "
        f"{min(24, len(completed_pair_states))} two-pair states",
        flush=True,
    )
    search_truncated = False
    time_limit_reached = False
    candidate_budget_reached = False
    search_overlap_cache: Dict[
        Tuple[int, int, Tuple[int, int, int]],
        float,
    ] = {}
    next_progress_time = search_started + 5.0
    first_valid_node: Optional[int] = (
        explored_nodes if leaves else None
    )
    first_refinable_node: Optional[int] = (
        explored_nodes if refinable_candidates else None
    )
    # Printed candidates receive a separate texture ranking, so a bounded
    # follow-up after the first valid leaf is sufficient on the embedded CPU.
    # Keep the deeper geometry-only check for solid-colour fragments.
    # All-quadrilateral card layouts have a much smaller bounded seed set.
    # Half the normal texture follow-up still checks competing rectangles
    # while fitting the measured board throughput for this path.
    post_leaf_node_budget = (
        384
        if recovery_mode and use_texture
        else (
            96
            if printed_four_quadrilaterals
            else (192 if use_texture else 1000)
        )
    )
    max_search_depth = max(
        (len(item[2]) for item in queue),
        default=0,
    )
    complete_states_checked = 0

    while (
        queue
        and explored_nodes < config.solver_max_nodes
        and len(leaves) < config.solver_max_leaf_candidates
    ):
        now = py_time.monotonic()
        if (
            first_valid_node is not None
            and explored_nodes - first_valid_node >= post_leaf_node_budget
        ):
            candidate_budget_reached = True
            break
        if (
            not leaves
            and first_refinable_node is not None
            and (
                len(refinable_candidates) >= 6
                or explored_nodes - first_refinable_node >= 768
            )
        ):
            candidate_budget_reached = True
            break
        if now >= next_progress_time:
            print(
                f"[vision] solver searched {explored_nodes} nodes "
                f"({now - search_started:.1f} s)...",
                flush=True,
            )
            next_progress_time = now + 5.0
        if (
            time_limit_seconds is not None
            and now - search_started >= time_limit_seconds
        ):
            search_truncated = True
            time_limit_reached = True
            break

        (
            parent_priority,
            _,
            placed,
            placed_polygons,
            unplaced_frozen,
        ) = heapq.heappop(queue)
        max_search_depth = max(max_search_depth, len(placed))
        unplaced = set(unplaced_frozen)
        if not unplaced:
            complete_states_checked += 1
            polygons = list(placed_polygons.values())
            try:
                metrics = _raster_metrics(polygons, config)
            except SolveError:
                continue
            if not _metrics_are_valid(metrics, config):
                rejected_score = _geometry_score(
                    metrics, config, polygons
                )
                if rejected_score < best_rejected_score:
                    best_rejected_score = rejected_score
                    best_rejected_metrics = metrics
                if retain_refinable_candidate(
                    placed, metrics, polygons
                ) and first_refinable_node is None:
                    first_refinable_node = explored_nodes
                continue
            if first_valid_node is None:
                first_valid_node = explored_nodes
            transforms = {
                piece_id: matrix.copy()
                for piece_id, matrix in placed.items()
            }
            geometry_score = _geometry_score(metrics, config, polygons)
            # A near-perfect measured rectangle is safe to return immediately.
            # Hand-cut/card contours can, however, produce several merely valid
            # rectangles; returning the first one made search order decide the
            # answer.  Keep those candidates for the same final geometry
            # ranking used by texture mode.
            if (
                not use_texture
                and geometry_score <= 0.05
            ):
                return AssemblySolution(
                    transforms=transforms,
                    metrics=metrics,
                    geometry_score=geometry_score,
                    texture_score=0.0,
                    total_score=geometry_score,
                    explored_nodes=explored_nodes,
                    valid_candidate_count=1,
                    search_complete=True,
                    score_margin=None,
                )
            key = _solution_key(placed)
            if key in leaf_keys:
                continue
            leaf_keys.add(key)
            leaves.append(
                _Leaf(
                    transforms=transforms,
                    metrics=metrics,
                    geometry_score=geometry_score,
                )
            )
            continue

        # Resolve the more selective small fragments first.  They usually
        # expose fewer plausible long-edge/T-junction poses, which prevents a
        # large, weakly constrained card fragment from multiplying the queue
        # before the distinctive small seam has been fixed.
        for piece_id in sorted(
            unplaced,
            key=lambda item: (
                pieces_by_id[item].area_mm2,
                item,
            ),
        ):
            candidates = cached_candidates(
                piece_id, placed, placed_polygons
            )
            for matrix in candidates:
                if explored_nodes >= config.solver_max_nodes:
                    search_truncated = True
                    break
                if (
                    time_limit_seconds is not None
                    and explored_nodes % 256 == 0
                    and py_time.monotonic() - search_started
                    >= time_limit_seconds
                ):
                    search_truncated = True
                    time_limit_reached = True
                    break
                explored_nodes += 1
                next_placed = dict(placed)
                next_placed[piece_id] = matrix
                next_unplaced = frozenset(unplaced - {piece_id})
                state_key = (
                    tuple(sorted(next_placed)),
                    _solution_key(next_placed),
                )
                if state_key in seen_partial:
                    continue
                next_polygons = dict(placed_polygons)
                next_polygons[piece_id] = _transform_rigid(
                    pieces_by_id[piece_id].polygon_mm, matrix
                )
                if not _cached_incremental_valid(
                    next_polygons,
                    next_placed,
                    piece_id,
                    pieces_by_id,
                    total_piece_area,
                    config,
                    search_overlap_cache,
                ):
                    continue
                seen_partial.add(state_key)
                counter += 1
                priority = (
                    _unfinished_priority(
                        next_polygons,
                        total_piece_area,
                        config,
                        0,
                    )
                    if use_texture
                    else _partial_priority(
                        next_polygons,
                        total_piece_area,
                        config,
                    )
                )
                if (
                    printed_four_quadrilaterals
                    and len(next_placed) > 2
                ):
                    # The partial compactness score otherwise discards the
                    # strong seam evidence that selected the parent pair.
                    priority = (
                        0.25 * priority
                        + 0.75 * parent_priority
                    )
                # A tiny depth preference resolves ties toward complete states.
                priority += 0.002 * len(next_unplaced)
                if not next_unplaced:
                    # A full four-piece topology is cheap to raster-check and
                    # may already be a refinable card rectangle.  Inspect it
                    # before expanding unrelated shallower branches; leaving
                    # it at the ordinary compactness priority caused valid
                    # J/Q/K layouts to wait behind thousands of T-junction
                    # children on the embedded CPU.
                    priority -= 100.0
                heapq.heappush(
                    queue,
                    (
                        priority,
                        counter,
                        next_placed,
                        next_polygons,
                        next_unplaced,
                    ),
                )
            if search_truncated:
                break
        if search_truncated:
            break
    if recovery_mode and not leaves and refinable_candidates:
        refine_retained_candidates(maximum_attempts=6, maximum_valid=3)
    print(
        f"[vision] refinement reached depth {max_search_depth}/"
        f"{len(pieces)}, checked {complete_states_checked} complete states; "
        f"overlap cache={len(search_overlap_cache)}",
        flush=True,
    )
    search_complete = (
        not search_truncated
        and (not queue or candidate_budget_reached)
    )
    if not leaves:
        if time_limit_reached:
            if best_rejected_metrics is None:
                suffix = (
                    f" (time limit {config.solver_time_limit_ms} ms reached; "
                    f"depth {max_search_depth}/{len(pieces)}, "
                    f"checked {complete_states_checked} complete states)"
                )
            else:
                rejected = best_rejected_metrics
                suffix = (
                    f" (time limit {config.solver_time_limit_ms} ms reached; "
                    "closest complete state: "
                    f"fill={rejected.fill_ratio:.3f}, "
                    f"side={rejected.side_coverage_min:.3f}, "
                    f"convexity={rejected.convexity_ratio:.3f}, "
                    f"corner={rejected.corner_error_max_mm:.1f} mm, "
                    f"boundary={rejected.boundary_p95_mm:.1f}/"
                    f"{rejected.boundary_p99_mm:.1f} mm, "
                    f"overlap={rejected.overlap_ratio:.3f}, "
                    f"hole={rejected.hole_ratio:.3f})"
                )
        elif search_complete:
            if best_rejected_metrics is None:
                suffix = (
                    "; verify the detected vertices and that all fragments "
                    "belong to the same original rectangle"
                )
            else:
                rejected = best_rejected_metrics
                suffix = (
                    "; closest candidate failed rectangle checks: "
                    f"fill={rejected.fill_ratio:.3f}, "
                    f"side={rejected.side_coverage_min:.3f}, "
                    f"convexity={rejected.convexity_ratio:.3f}, "
                    f"corner={rejected.corner_error_max_mm:.1f} mm, "
                    f"boundary={rejected.boundary_p95_mm:.1f}/"
                    f"{rejected.boundary_p99_mm:.1f} mm, "
                    f"overlap={rejected.overlap_ratio:.3f}, "
                    f"hole={rejected.hole_ratio:.3f}"
                )
        else:
            suffix = " (search limit reached)"
        raise SolveError(
            f"no rectangular assembly found after {explored_nodes} search nodes"
            f"{suffix}"
        )

    card_strip_mode = (
        use_texture and _is_card_strip_partition(pieces)
    )
    texture_weight = (
        max(config.texture_weight, 0.60)
        if card_strip_mode
        else config.texture_weight
    )
    for leaf in leaves:
        leaf.texture_score = (
            _texture_score(
                pieces,
                leaf.transforms,
                paper_bgr,
                config.px_per_mm,
                config,
            )
            if use_texture
            else 0.0
        )
        leaf.total_score = (
            leaf.geometry_score + texture_weight * leaf.texture_score
        )
    leaves.sort(key=lambda leaf: leaf.total_score)
    distinct_leaves: List[_Leaf] = []
    anchor_id = min(pieces_by_id)
    for leaf in leaves:
        if any(
            _leaves_are_globally_equivalent(
                leaf,
                representative,
                pieces_by_id,
                anchor_id,
            )
            for representative in distinct_leaves
        ):
            continue
        distinct_leaves.append(leaf)
    best = distinct_leaves[0]
    margin = (
        distinct_leaves[1].total_score - best.total_score
        if len(distinct_leaves) > 1
        else None
    )
    # Four same-width strips have many geometrically valid permutations.
    # A weak whole-face match is not a valid playing-card answer even when
    # its outline is rectangular; surface that case as LOW_MARGIN.
    if card_strip_mode and best.texture_score >= 0.20:
        margin = 0.0
    if use_texture and len(distinct_leaves) > 1:
        runner_up = distinct_leaves[1]
        print(
            "[vision] texture candidates "
            f"best=(g={best.geometry_score:.4f}, "
            f"t={best.texture_score:.4f}, total={best.total_score:.4f}) "
            f"next=(g={runner_up.geometry_score:.4f}, "
            f"t={runner_up.texture_score:.4f}, "
            f"total={runner_up.total_score:.4f})",
            flush=True,
        )
    return AssemblySolution(
        transforms=best.transforms,
        metrics=best.metrics,
        geometry_score=best.geometry_score,
        texture_score=best.texture_score,
        total_score=best.total_score,
        explored_nodes=explored_nodes,
        valid_candidate_count=len(distinct_leaves),
        search_complete=search_complete,
        score_margin=margin,
    )
