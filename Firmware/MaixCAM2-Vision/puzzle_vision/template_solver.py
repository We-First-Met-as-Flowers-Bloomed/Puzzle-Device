from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import VisionConfig
from .geometry import (
    AssemblySolution,
    PieceObservation,
    SolveError,
    ensure_positive_winding,
    transform_points,
)


_TEMPLATE_LABELS = ("A", "B", "C", "D")
_TEMPLATE_NODE_LABELS = (
    ("a", "b", "c", "d"),
    ("b", "top_right", "bottom_right"),
    ("d", "c", "f", "e"),
    ("e", "f", "bottom_right", "bottom_left"),
)
_RECTANGLE_WIDTH_MM = 100.0
_RECTANGLE_HEIGHT_MM = 60.0
# The contest clarification judges adjacent corresponding vertices at 2 cm.
# Keep the target below that published limit; never use a reflected transform.
_CORRESPONDING_VERTEX_LIMIT_MM = 20.0
# A relaxed fixed-topology answer is allowed only when three pieces identify
# the known cut very tightly.  The fourth piece may then use the published
# corresponding-point tolerance; this handles hand-cut error without turning
# the fixed template into a permissive generic rectangle detector.
_TOPOLOGY_STRONG_PIECE_RMS_MM = 2.5
_TOPOLOGY_MIN_STRONG_PIECES = 3
# A hand-cut outer side can be split by small endpoint offsets and rasterize at
# about 0.50 coverage.  This exception is limited to a strict Figure-2 shape
# match; fill/corner/overlap get only the measured frame-jitter allowance
# below, while holes, boundary, convexity, connectivity, and target size pass.
_TEMPLATE_SIDE_COVERAGE_MIN = 0.40
_TEMPLATE_FILL_RATIO_MIN = 0.92
_TEMPLATE_OVERLAP_RATIO_MAX = 0.03
_TEMPLATE_CORNER_ERROR_MAX_MM = 6.0


@dataclass
class _Fit:
    matrix: np.ndarray
    scale_to_template: float
    rms_mm: float
    max_error_mm: float
    squared_error: float
    vertex_count: int
    source_shift: int


@dataclass
class _Assignment:
    template_indices: Tuple[int, ...]
    hands: Tuple[int, ...]
    fits: Tuple[_Fit, ...]
    rms_mm: float


@dataclass(frozen=True)
class _TrainedTemplate:
    name: str
    polygons: Tuple[np.ndarray, ...]
    minimum_side_coverage: float
    maximum_boundary_p95_mm: float = 9.0
    maximum_boundary_p99_mm: float = 11.0
    center_symmetric_rank: bool = False
    rank_corner: Tuple[int, int] = (0, 0)
    rank_triangle_index: int = 1
    plain_triangle_index: int = 0
    opposite_rank_piece_index: int = 2
    rank_is_darker: bool = True
    rank_min_dark_ratio: float = 0.10
    rank_corner_limit_mm: float = 14.0
    opposite_rank_corner_limit_mm: float = 28.0
    match_rms_limit_mm: float = 4.0
    match_piece_rms_limit_mm: float = 4.8
    match_vertex_limit_mm: float = 9.0
    match_scale_min: float = 0.80
    match_scale_max: float = 1.22
    match_scale_spread: float = 1.25


def _trained_polygon(points: Sequence[Sequence[float]]) -> np.ndarray:
    return ensure_positive_winding(np.asarray(points, dtype=np.float64))


def _scaled_card_template(
    template: _TrainedTemplate,
    name: str,
    width_mm: float,
    height_mm: float,
) -> _TrainedTemplate:
    all_points = np.vstack(template.polygons)
    lower = np.min(all_points, axis=0)
    size = np.max(all_points, axis=0) - lower
    scale = np.array(
        [width_mm / size[0], height_mm / size[1]],
        dtype=np.float64,
    )
    rank_scale = float(np.max(scale))
    return replace(
        template,
        name=name,
        polygons=tuple(
            _trained_polygon((polygon - lower) * scale)
            for polygon in template.polygons
        ),
        rank_corner_limit_mm=(
            template.rank_corner_limit_mm * rank_scale
        ),
        opposite_rank_corner_limit_mm=(
            template.opposite_rank_corner_limit_mm * rank_scale
        ),
    )


def trained_template_bank() -> Tuple[_TrainedTemplate, ...]:
    """Return the camera-trained cuts and code-generated card-size variants.

    The coordinates are physical millimetres in each recovered target
    rectangle.  They were fitted from four independent captures per cut;
    runtime matching therefore needs only proper rotations and translations.
    """
    trained = (
        _TrainedTemplate(
            name="custom-1-card-k",
            center_symmetric_rank=True,
            rank_corner=(1, -1),
            rank_triangle_index=0,
            plain_triangle_index=1,
            opposite_rank_piece_index=3,
            rank_is_darker=True,
            rank_min_dark_ratio=0.10,
            rank_corner_limit_mm=14.0,
            minimum_side_coverage=0.80,
            polygons=(
                _trained_polygon(
                    (
                        (58.554, 0.431),
                        (86.284, 0.316),
                        (88.635, 28.001),
                    )
                ),
                _trained_polygon(
                    (
                        (0.250, 1.602),
                        (28.106, 1.762),
                        (1.020, 29.822),
                    )
                ),
                _trained_polygon(
                    (
                        (87.229, 58.491),
                        (43.642, 57.437),
                        (59.544, 1.338),
                        (87.645, 27.094),
                    )
                ),
                _trained_polygon(
                    (
                        (0.879, 29.969),
                        (28.248, 1.616),
                        (59.643, 0.990),
                        (43.543, 57.786),
                        (0.968, 58.412),
                    )
                ),
            ),
        ),
        _TrainedTemplate(
            name="custom-2-card-k",
            center_symmetric_rank=True,
            rank_corner=(1, -1),
            rank_triangle_index=1,
            plain_triangle_index=0,
            opposite_rank_piece_index=2,
            minimum_side_coverage=0.76,
            match_rms_limit_mm=4.0,
            match_piece_rms_limit_mm=4.8,
            match_vertex_limit_mm=9.0,
            match_scale_min=0.80,
            match_scale_max=1.22,
            match_scale_spread=1.25,
            polygons=(
                _trained_polygon(
                    (
                        (87.326, 38.649),
                        (87.976, 55.666),
                        (69.959, 56.295),
                    )
                ),
                _trained_polygon(
                    (
                        (88.677, 3.587),
                        (88.301, 21.303),
                        (69.473, 1.542),
                    )
                ),
                _trained_polygon(
                    (
                        (68.797, 57.475),
                        (0.122, 56.661),
                        (0.412, 29.755),
                        (88.488, 37.468),
                    )
                ),
                _trained_polygon(
                    (
                        (87.428, 37.376),
                        (1.472, 29.848),
                        (0.225, 5.040),
                        (68.056, 0.056),
                        (89.717, 22.789),
                    )
                ),
            ),
        ),
        _TrainedTemplate(
            name="custom-3-white",
            # This cut stays uniquely identifiable when a shadow clips the
            # outer endpoints of one fragment.  The relaxed boundary remains
            # within the problem's 20 mm corresponding-point allowance.
            minimum_side_coverage=0.60,
            maximum_boundary_p95_mm=12.0,
            maximum_boundary_p99_mm=20.0,
            match_scale_max=1.25,
            polygons=(
                _trained_polygon(
                    (
                        (100.327, 63.148),
                        (20.838, 62.643),
                        (38.725, 40.829),
                        (100.899, 40.466),
                    )
                ),
                _trained_polygon(
                    (
                        (99.837, 40.473),
                        (39.787, 40.823),
                        (71.256, 0.862),
                        (100.100, 0.468),
                    )
                ),
                _trained_polygon(
                    (
                        (20.838, 62.643),
                        (2.504, 63.179),
                        (-0.016, 4.396),
                        (72.331, -0.156),
                    )
                ),
            ),
        ),
        _TrainedTemplate(
            name="custom-4-white",
            minimum_side_coverage=0.88,
            polygons=(
                _trained_polygon(
                    (
                        (52.986, 21.323),
                        (98.210, 61.158),
                        (19.981, 60.753),
                    )
                ),
                _trained_polygon(
                    (
                        (52.955, 21.296),
                        (70.019, 0.305),
                        (96.454, 0.632),
                        (98.241, 61.185),
                    )
                ),
                _trained_polygon(
                    (
                        (1.246, 0.238),
                        (70.019, 0.305),
                        (21.450, 60.051),
                        (-0.049, 61.436),
                    )
                ),
            ),
        ),
    )
    return trained + (
        _scaled_card_template(
            trained[0],
            "custom-1-card-k-110x80",
            110.0,
            80.0,
        ),
        replace(
            _scaled_card_template(
                trained[1],
                "custom-2-card-k-110x80",
                110.0,
                80.0,
            ),
            minimum_side_coverage=0.40,
        ),
    )


def figure2_template_polygons() -> Tuple[np.ndarray, ...]:
    """Return independent copies of the confirmed Figure-2 polygons."""
    a = np.array([0.0, 0.0])
    b = np.array([20.0, 0.0])
    top_right = np.array([100.0, 0.0])
    bottom_right = np.array([100.0, 60.0])
    bottom_left = np.array([0.0, 60.0])
    d = np.array([0.0, 20.0])
    e = np.array([0.0, 30.0])
    c = np.array([36.0, 12.0])
    f = np.array([76.0, 42.0])
    return (
        ensure_positive_winding(np.array([a, b, c, d])),
        ensure_positive_winding(
            np.array([b, top_right, bottom_right])
        ),
        ensure_positive_winding(np.array([d, c, f, e])),
        ensure_positive_winding(
            np.array([e, f, bottom_right, bottom_left])
        ),
    )


def _opposite_cut_hand_templates(
    templates: Sequence[np.ndarray],
) -> Tuple[np.ndarray, ...]:
    opposite_hand = []
    for polygon in templates:
        result = np.asarray(polygon, dtype=np.float64).copy()
        result[:, 0] = _RECTANGLE_WIDTH_MM - result[:, 0]
        opposite_hand.append(ensure_positive_winding(result))
    return tuple(opposite_hand)


def _collinear_measure(
    polygon: np.ndarray, index: int
) -> Tuple[float, float, bool]:
    count = len(polygon)
    previous = polygon[(index - 1) % count]
    current = polygon[index]
    following = polygon[(index + 1) % count]
    chord = following - previous
    chord_length = float(np.linalg.norm(chord))
    incoming = current - previous
    outgoing = following - current
    incoming_length = float(np.linalg.norm(incoming))
    outgoing_length = float(np.linalg.norm(outgoing))
    if min(chord_length, incoming_length, outgoing_length) < 1e-6:
        return float("inf"), 180.0, False
    distance = abs(
        float(
            chord[0] * (current[1] - previous[1])
            - chord[1] * (current[0] - previous[0])
        )
    ) / chord_length
    cosine = float(
        np.dot(incoming, outgoing) / (incoming_length * outgoing_length)
    )
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    projection = float(
        np.dot(current - previous, chord) / (chord_length * chord_length)
    )
    return distance, angle, -0.05 <= projection <= 1.05


def _clean_polygon(
    polygon: np.ndarray,
    target_vertices: int,
    config: VisionConfig,
) -> Optional[np.ndarray]:
    result = ensure_positive_winding(polygon)
    if len(result) < target_vertices:
        return None
    maximum_distance = max(
        3.5, 1.5 * float(config.solver_collinear_tolerance_mm)
    )
    maximum_angle_deg = 35.0
    while len(result) > target_vertices:
        candidates = []
        for index in range(len(result)):
            distance, angle, between = _collinear_measure(result, index)
            if between:
                candidates.append(
                    (distance + 0.03 * angle, distance, angle, index)
                )
        if not candidates:
            return None
        _, distance, angle, index = min(candidates)
        if distance > maximum_distance or angle > maximum_angle_deg:
            return None
        result = np.delete(result, index, axis=0)
    return ensure_positive_winding(result)


def _best_rotation_fit(
    source: np.ndarray, target: np.ndarray
) -> _Fit:
    """Classify shape by similarity, but return a rigid-only pose."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    best: Optional[_Fit] = None
    for shift in range(len(source)):
        ordered = np.roll(source, shift, axis=0)
        source_center = ordered.mean(axis=0)
        target_center = target.mean(axis=0)
        source_zero = ordered - source_center
        target_zero = target - target_center
        dot = float(np.sum(source_zero * target_zero))
        cross = float(
            np.sum(
                source_zero[:, 0] * target_zero[:, 1]
                - source_zero[:, 1] * target_zero[:, 0]
            )
        )
        if abs(dot) + abs(cross) < 1e-9:
            continue
        angle = float(np.arctan2(cross, dot))
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        row_rotation = np.array(
            [[cosine, sine], [-sine, cosine]], dtype=np.float64
        )
        rotated_zero = source_zero @ row_rotation
        denominator = float(np.sum(rotated_zero * rotated_zero))
        if denominator < 1e-9:
            continue
        scale_to_template = float(
            np.sum(rotated_zero * target_zero) / denominator
        )
        if scale_to_template <= 0.0:
            continue
        translation = target_center - source_center @ row_rotation
        predicted = (
            rotated_zero * scale_to_template + target_center
        )
        errors = np.linalg.norm(predicted - target, axis=1)
        squared_error = float(np.sum(errors * errors))
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :2] = row_rotation.T
        matrix[:2, 2] = translation
        fit = _Fit(
            matrix=matrix,
            scale_to_template=scale_to_template,
            rms_mm=float(np.sqrt(np.mean(errors * errors))),
            max_error_mm=float(np.max(errors)),
            squared_error=squared_error,
            vertex_count=len(source),
            source_shift=shift,
        )
        if best is None or fit.squared_error < best.squared_error:
            best = fit
    if best is None:
        raise SolveError("Figure-2 Kabsch fit could not produce a rotation")
    return best


def _assignment_rms(fits: Sequence[_Fit]) -> float:
    squared_error = sum(fit.squared_error for fit in fits)
    vertex_count = sum(fit.vertex_count for fit in fits)
    return float(np.sqrt(squared_error / max(vertex_count, 1)))


def _quality_limits(config: VisionConfig) -> Tuple[float, float, float]:
    tolerance = max(1.0, float(config.solver_collinear_tolerance_mm))
    total_rms = max(3.5, 1.5 * tolerance)
    piece_rms = max(4.0, 1.75 * tolerance)
    maximum_vertex = max(8.0, 3.0 * tolerance)
    return total_rms, piece_rms, maximum_vertex


def _assignment_is_usable(
    assignment: _Assignment, config: VisionConfig
) -> bool:
    total_limit, piece_limit, vertex_limit = _quality_limits(config)
    scales = [fit.scale_to_template for fit in assignment.fits]
    scale_spread = max(scales) / max(min(scales), 1e-9)
    return (
        assignment.rms_mm <= total_limit
        and all(fit.rms_mm <= piece_limit for fit in assignment.fits)
        and all(
            fit.max_error_mm <= vertex_limit for fit in assignment.fits
        )
        and all(0.85 <= scale <= 1.18 for scale in scales)
        and scale_spread <= 1.10
    )


def _fit_cache(
    pieces: Sequence[PieceObservation],
    template_hands: Sequence[Sequence[np.ndarray]],
    config: VisionConfig,
) -> Dict[Tuple[int, int, int], Optional[_Fit]]:
    cache: Dict[Tuple[int, int, int], Optional[_Fit]] = {}
    for piece_index, piece in enumerate(pieces):
        for template_index, template in enumerate(template_hands[0]):
            cleaned = _clean_polygon(
                piece.polygon_mm, len(template), config
            )
            for hand, templates in enumerate(template_hands):
                cache[(piece_index, template_index, hand)] = (
                    None
                    if cleaned is None
                    else _best_rotation_fit(
                        cleaned, templates[template_index]
                    )
                )
    return cache


def _global_assignments(
    pieces: Sequence[PieceObservation],
    cache: Dict[Tuple[int, int, int], Optional[_Fit]],
) -> Tuple[List[_Assignment], int]:
    assignments: List[_Assignment] = []
    explored = 0
    for hand in (0, 1):
        for permutation in itertools.permutations(range(4)):
            explored += 1
            fits = tuple(
                cache[(piece_index, template_index, hand)]
                for piece_index, template_index in enumerate(permutation)
            )
            if any(fit is None for fit in fits):
                continue
            concrete = tuple(fit for fit in fits if fit is not None)
            assignments.append(
                _Assignment(
                    template_indices=tuple(permutation),
                    hands=(hand, hand, hand, hand),
                    fits=concrete,
                    rms_mm=_assignment_rms(concrete),
                )
            )
    assignments.sort(key=lambda item: item.rms_mm)
    return assignments, explored


def _layout_scale(assignment: _Assignment) -> float:
    """Use one physical scale for all four rigid target poses."""
    return float(
        np.median(
            [
                1.0 / fit.scale_to_template
                for fit in assignment.fits
            ]
        )
    )


def _layout_transforms(
    pieces: Sequence[PieceObservation],
    assignment: _Assignment,
    templates: Sequence[Sequence[np.ndarray]],
) -> Dict[int, np.ndarray]:
    layout_scale = _layout_scale(assignment)
    transforms: Dict[int, np.ndarray] = {}
    for index, piece in enumerate(pieces):
        matrix = assignment.fits[index].matrix.copy()
        hand = assignment.hands[index]
        template_index = assignment.template_indices[index]
        target_center = templates[hand][template_index].mean(axis=0)
        matrix[:2, 2] += (layout_scale - 1.0) * target_center
        transforms[piece.piece_id] = matrix
    return transforms


def _piece_node(
    pieces: Sequence[PieceObservation],
    assignment: _Assignment,
    piece_index: int,
    node: str,
) -> np.ndarray:
    """Return the measured source vertex assigned to one topology node."""
    template_index = assignment.template_indices[piece_index]
    labels = _TEMPLATE_NODE_LABELS[template_index]
    if assignment.hands[piece_index] == 1:
        labels = tuple(reversed(labels))
    target_index = labels.index(node)
    source_index = (
        target_index - assignment.fits[piece_index].source_shift
    ) % len(pieces[piece_index].polygon_mm)
    return pieces[piece_index].polygon_mm[source_index]


def _align_piece_nodes(
    polygon: np.ndarray,
    source_start: np.ndarray,
    source_end: np.ndarray,
    target_start: np.ndarray,
    target_end: np.ndarray,
) -> np.ndarray:
    """Rigidly align a measured seam by its two corresponding endpoints."""
    source_vector = source_end - source_start
    target_vector = target_end - target_start
    source_angle = float(
        np.arctan2(source_vector[1], source_vector[0])
    )
    target_angle = float(
        np.arctan2(target_vector[1], target_vector[0])
    )
    angle = target_angle - source_angle
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    source_midpoint = 0.5 * (source_start + source_end)
    target_midpoint = 0.5 * (target_start + target_end)
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = target_midpoint - source_midpoint @ rotation.T
    return matrix


def _seam_locked_transforms(
    pieces: Sequence[PieceObservation],
    assignment: _Assignment,
    templates: Sequence[Sequence[np.ndarray]],
) -> Dict[int, np.ndarray]:
    """Assemble Figure-2 by its boundary topology and hierarchical seam loop.

    D is anchored to the known rectangle boundary.  C is locked to D on
    E--F, A is locked to C on D--C, and the triangular B piece closes the
    complete B--C--F--bottom-right cut.  This is bounded and deterministic:
    no pairwise brute-force search is needed.
    """
    piece_index_by_template = {
        template_index: piece_index
        for piece_index, template_index in enumerate(
            assignment.template_indices
        )
    }
    a_index = piece_index_by_template[0]
    b_index = piece_index_by_template[1]
    c_index = piece_index_by_template[2]
    d_index = piece_index_by_template[3]

    baseline = _layout_transforms(pieces, assignment, templates)
    transforms: Dict[int, np.ndarray] = {}

    d_piece = pieces[d_index]
    transforms[d_piece.piece_id] = baseline[d_piece.piece_id]

    def placed_node(piece_index: int, node: str) -> np.ndarray:
        piece = pieces[piece_index]
        return transform_points(
            _piece_node(
                pieces, assignment, piece_index, node
            ).reshape(1, 2),
            transforms[piece.piece_id],
        )[0]

    c_piece = pieces[c_index]
    transforms[c_piece.piece_id] = _align_piece_nodes(
        c_piece.polygon_mm,
        _piece_node(pieces, assignment, c_index, "e"),
        _piece_node(pieces, assignment, c_index, "f"),
        placed_node(d_index, "e"),
        placed_node(d_index, "f"),
    )

    a_piece = pieces[a_index]
    transforms[a_piece.piece_id] = _align_piece_nodes(
        a_piece.polygon_mm,
        _piece_node(pieces, assignment, a_index, "d"),
        _piece_node(pieces, assignment, a_index, "c"),
        placed_node(c_index, "d"),
        placed_node(c_index, "c"),
    )

    b_piece = pieces[b_index]
    transforms[b_piece.piece_id] = _align_piece_nodes(
        b_piece.polygon_mm,
        _piece_node(pieces, assignment, b_index, "b"),
        _piece_node(
            pieces, assignment, b_index, "bottom_right"
        ),
        placed_node(a_index, "b"),
        placed_node(d_index, "bottom_right"),
    )
    return transforms


def _layout_vertex_errors(
    pieces: Sequence[PieceObservation],
    assignment: _Assignment,
    templates: Sequence[Sequence[np.ndarray]],
) -> np.ndarray:
    """Measure the contest's corresponding-vertex rule in target millimetres."""
    layout_scale = _layout_scale(assignment)
    transforms = _layout_transforms(pieces, assignment, templates)
    errors: List[np.ndarray] = []
    for index, piece in enumerate(pieces):
        fit = assignment.fits[index]
        template_index = assignment.template_indices[index]
        hand = assignment.hands[index]
        placed = transform_points(
            piece.polygon_mm, transforms[piece.piece_id]
        )
        corresponding = np.roll(placed, fit.source_shift, axis=0)
        target = templates[hand][template_index] * layout_scale
        errors.append(np.linalg.norm(corresponding - target, axis=1))
    return np.concatenate(errors)


def _placed_topology_node(
    pieces: Sequence[PieceObservation],
    assignment: _Assignment,
    transforms: Dict[int, np.ndarray],
    piece_index: int,
    node: str,
) -> np.ndarray:
    piece = pieces[piece_index]
    return transform_points(
        _piece_node(
            pieces, assignment, piece_index, node
        ).reshape(1, 2),
        transforms[piece.piece_id],
    )[0]


def _topology_corresponding_gaps(
    pieces: Sequence[PieceObservation],
    assignment: _Assignment,
    templates: Sequence[Sequence[np.ndarray]],
    transforms: Dict[int, np.ndarray],
) -> np.ndarray:
    """Measure every shared Figure-2 junction after rigid placement.

    C and F lie inside the triangular piece's long edge rather than at
    triangle vertices.  Their positions are therefore interpolated from the
    confirmed cut topology.  This directly implements the contest's
    "adjacent corresponding vertices" check and never reflects a piece.
    """
    piece_index_by_template = {
        template_index: piece_index
        for piece_index, template_index in enumerate(
            assignment.template_indices
        )
    }
    a_index = piece_index_by_template[0]
    b_index = piece_index_by_template[1]
    c_index = piece_index_by_template[2]
    d_index = piece_index_by_template[3]

    def node(piece_index: int, label: str) -> np.ndarray:
        return _placed_topology_node(
            pieces,
            assignment,
            transforms,
            piece_index,
            label,
        )

    triangle_b = node(b_index, "b")
    triangle_bottom_right = node(b_index, "bottom_right")
    hand = assignment.hands[b_index]
    triangle_template = templates[hand][1]
    labels = _TEMPLATE_NODE_LABELS[1]
    if hand == 1:
        labels = tuple(reversed(labels))
    template_b = triangle_template[labels.index("b")]
    template_bottom_right = triangle_template[
        labels.index("bottom_right")
    ]
    diagonal = template_bottom_right - template_b
    diagonal_squared = max(float(np.dot(diagonal, diagonal)), 1e-9)

    def triangle_point(label: str) -> np.ndarray:
        reference = None
        for template_index, original_labels in enumerate(
            _TEMPLATE_NODE_LABELS
        ):
            labels_for_hand = (
                tuple(reversed(original_labels))
                if hand == 1
                else original_labels
            )
            if label in labels_for_hand:
                reference = templates[hand][template_index][
                    labels_for_hand.index(label)
                ]
                break
        if reference is None:
            raise SolveError(
                f"Figure-2 topology node {label!r} is undefined"
            )
        fraction = float(
            np.dot(reference - template_b, diagonal)
            / diagonal_squared
        )
        return (
            triangle_b
            + fraction * (triangle_bottom_right - triangle_b)
        )

    groups = (
        (node(a_index, "b"), triangle_b),
        (
            node(a_index, "c"),
            node(c_index, "c"),
            triangle_point("c"),
        ),
        (node(a_index, "d"), node(c_index, "d")),
        (node(c_index, "e"), node(d_index, "e")),
        (
            node(c_index, "f"),
            node(d_index, "f"),
            triangle_point("f"),
        ),
        (
            triangle_bottom_right,
            node(d_index, "bottom_right"),
        ),
    )
    gaps: List[float] = []
    for group in groups:
        for first in range(len(group)):
            for second in range(first + 1, len(group)):
                gaps.append(
                    float(np.linalg.norm(group[first] - group[second]))
                )
    return np.asarray(gaps, dtype=np.float64)


def _assignment_has_confirmed_topology(
    pieces: Sequence[PieceObservation],
    assignment: _Assignment,
    templates: Sequence[Sequence[np.ndarray]],
    config: VisionConfig,
) -> bool:
    scales = [fit.scale_to_template for fit in assignment.fits]
    scale_spread = max(scales) / max(min(scales), 1e-9)
    if (
        not all(0.85 <= scale <= 1.18 for scale in scales)
        or scale_spread > 1.10
    ):
        return False
    strong_limit = max(
        _TOPOLOGY_STRONG_PIECE_RMS_MM,
        1.25 * float(config.solver_collinear_tolerance_mm),
    )
    if (
        sum(fit.rms_mm <= strong_limit for fit in assignment.fits)
        < _TOPOLOGY_MIN_STRONG_PIECES
        or any(
            fit.max_error_mm > _CORRESPONDING_VERTEX_LIMIT_MM
            for fit in assignment.fits
        )
    ):
        return False

    layout_scale = _layout_scale(assignment)
    measured_area = sum(piece.area_mm2 for piece in pieces)
    target_area = (
        _RECTANGLE_WIDTH_MM
        * _RECTANGLE_HEIGHT_MM
        * layout_scale
        * layout_scale
    )
    area_ratio = measured_area / max(target_area, 1e-9)
    return 0.88 <= area_ratio <= 1.12


def _assignment_meets_vertex_rule(
    pieces: Sequence[PieceObservation],
    assignment: _Assignment,
    templates: Sequence[Sequence[np.ndarray]],
) -> bool:
    """Accept a noisy Figure-2 only when one proper-rotation hand is legal."""
    scales = [fit.scale_to_template for fit in assignment.fits]
    scale_spread = max(scales) / max(min(scales), 1e-9)
    if (
        not all(0.85 <= scale <= 1.18 for scale in scales)
        or scale_spread > 1.10
    ):
        return False
    errors = _layout_vertex_errors(pieces, assignment, templates)
    return float(np.max(errors)) <= _CORRESPONDING_VERTEX_LIMIT_MM


def _template_metrics_are_valid(metrics, config: VisionConfig) -> bool:
    """Validate a confirmed Figure-2 without requiring zero raster seam gap."""
    from .solver import _metrics_are_valid

    if _metrics_are_valid(metrics, config):
        return True
    short, long = metrics.rectangle_size_mm
    return (
        1 <= metrics.connected_components <= 4
        and config.target_short_min_mm
        <= short
        <= config.target_short_max_mm
        and config.target_long_min_mm
        <= long
        <= config.target_long_max_mm
        and metrics.fill_ratio
        >= min(
            config.min_rectangle_fill_ratio,
            _TEMPLATE_FILL_RATIO_MIN,
        )
        and metrics.overlap_ratio
        <= max(
            config.max_overlap_ratio,
            _TEMPLATE_OVERLAP_RATIO_MAX,
        )
        and metrics.hole_ratio <= config.max_hole_ratio
        and metrics.boundary_p95_mm
        <= _CORRESPONDING_VERTEX_LIMIT_MM
        and metrics.boundary_p99_mm
        <= _CORRESPONDING_VERTEX_LIMIT_MM
        and metrics.side_coverage_min
        >= min(
            config.min_rectangle_side_coverage,
            _TEMPLATE_SIDE_COVERAGE_MIN,
        )
        and metrics.convexity_ratio
        >= config.min_rectangle_convexity_ratio
        and metrics.corner_error_max_mm
        <= max(
            config.max_rectangle_corner_error_mm,
            _TEMPLATE_CORNER_ERROR_MAX_MM,
        )
    )


def _trained_assignments(
    pieces: Sequence[PieceObservation],
    template: _TrainedTemplate,
    config: VisionConfig,
) -> Tuple[List[_Assignment], int]:
    count = len(pieces)
    cache: Dict[Tuple[int, int], Optional[_Fit]] = {}
    for piece_index, piece in enumerate(pieces):
        for template_index, target in enumerate(template.polygons):
            cleaned = _clean_polygon(
                piece.polygon_mm, len(target), config
            )
            cache[(piece_index, template_index)] = (
                None
                if cleaned is None
                else _best_rotation_fit(cleaned, target)
            )

    total_limit = max(
        template.match_rms_limit_mm,
        0.65 * float(config.solver_collinear_tolerance_mm),
    )
    piece_limit = max(
        template.match_piece_rms_limit_mm,
        0.80 * float(config.solver_collinear_tolerance_mm),
    )
    vertex_limit = max(
        template.match_vertex_limit_mm,
        1.50 * float(config.solver_collinear_tolerance_mm),
    )
    assignments: List[_Assignment] = []
    explored = 0
    for permutation in itertools.permutations(range(count)):
        explored += 1
        fits = tuple(
            cache[(piece_index, template_index)]
            for piece_index, template_index in enumerate(permutation)
        )
        if any(fit is None for fit in fits):
            continue
        concrete = tuple(fit for fit in fits if fit is not None)
        scales = [fit.scale_to_template for fit in concrete]
        scale_spread = max(scales) / max(min(scales), 1e-9)
        rms = _assignment_rms(concrete)
        if (
            rms <= total_limit
            and all(fit.rms_mm <= piece_limit for fit in concrete)
            and all(
                fit.max_error_mm <= vertex_limit for fit in concrete
            )
            and all(
                template.match_scale_min
                <= scale
                <= template.match_scale_max
                for scale in scales
            )
            and scale_spread <= template.match_scale_spread
        ):
            assignments.append(
                _Assignment(
                    template_indices=tuple(permutation),
                    hands=(0,) * count,
                    fits=concrete,
                    rms_mm=rms,
                )
            )
    assignments.sort(key=lambda item: item.rms_mm)
    return assignments, explored


def _piece_dark_ratio(
    paper_bgr: np.ndarray, piece: PieceObservation
) -> float:
    """Measure black rank ink while excluding the hand-cut boundary."""
    gray = cv2.cvtColor(paper_bgr, cv2.COLOR_BGR2GRAY)
    safe_mask = cv2.erode(
        (piece.mask > 0).astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
    )
    valid = safe_mask > 0
    if int(np.count_nonzero(valid)) < 25:
        return 0.0
    return float(np.mean(gray[valid] < 150))


def _center_symmetric_rank_is_valid(
    pieces: Sequence[PieceObservation],
    assignment: _Assignment,
    transforms: Dict[int, np.ndarray],
    paper_bgr: np.ndarray,
    px_per_mm: float,
    template: _TrainedTemplate,
) -> Tuple[bool, float, float, float]:
    """Bind the K-bearing small triangle to its opposite-corner slot.

    The template metadata identifies which of the two nearly isosceles small
    triangles carries one K.  The other K lies on a geometrically unique large
    piece at the 180-degree opposite card corner.  Both measured rank marks
    must land near their respective outer card corners.
    """
    piece_by_template = {
        template_index: piece_index
        for piece_index, template_index in enumerate(
            assignment.template_indices
        )
    }
    required_indices = (
        template.rank_triangle_index,
        template.plain_triangle_index,
        template.opposite_rank_piece_index,
    )
    if any(index not in piece_by_template for index in required_indices):
        return False, 0.0, float("inf"), float("inf")
    plain_ratio = _piece_dark_ratio(
        paper_bgr,
        pieces[piece_by_template[template.plain_triangle_index]],
    )
    marked_piece = pieces[
        piece_by_template[template.rank_triangle_index]
    ]
    marked_ratio = _piece_dark_ratio(paper_bgr, marked_piece)
    separation = (
        marked_ratio - plain_ratio
        if template.rank_is_darker
        else plain_ratio - marked_ratio
    )
    gray = cv2.cvtColor(paper_bgr, cv2.COLOR_BGR2GRAY)
    all_target_points = np.vstack(
        [
            transform_points(
                piece.polygon_mm, transforms[piece.piece_id]
            )
            for piece in pieces
        ]
    )
    lower = np.min(all_target_points, axis=0)
    upper = np.max(all_target_points, axis=0)

    def corner_error(
        piece: PieceObservation,
        corner_sign: Tuple[int, int],
        percentile: float,
    ) -> float:
        safe_mask = cv2.erode(
            (piece.mask > 0).astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
        )
        ink_y, ink_x = np.nonzero(
            (safe_mask > 0) & (gray < 150)
        )
        if len(ink_x) < 25:
            return float("inf")
        source_mm = (
            np.column_stack([ink_x, ink_y]).astype(np.float64)
            / px_per_mm
        )
        target_mm = transform_points(
            source_mm, transforms[piece.piece_id]
        )
        expected = np.array(
            [
                upper[0] if corner_sign[0] > 0 else lower[0],
                upper[1] if corner_sign[1] > 0 else lower[1],
            ],
            dtype=np.float64,
        )
        return float(
            np.percentile(
                np.linalg.norm(target_mm - expected, axis=1),
                percentile,
            )
        )

    corner_error_mm = corner_error(
        marked_piece, template.rank_corner, 25.0
    )
    opposite_piece = pieces[
        piece_by_template[template.opposite_rank_piece_index]
    ]
    opposite_corner_error_mm = corner_error(
        opposite_piece,
        (-template.rank_corner[0], -template.rank_corner[1]),
        10.0,
    )
    return (
        marked_ratio >= template.rank_min_dark_ratio
        and separation >= 0.06
        and corner_error_mm <= template.rank_corner_limit_mm
        and opposite_corner_error_mm
        <= template.opposite_rank_corner_limit_mm,
        separation,
        corner_error_mm,
        opposite_corner_error_mm,
    )


def _trained_metrics_are_valid(
    metrics,
    template: _TrainedTemplate,
    config: VisionConfig,
) -> bool:
    """Use a template-only hand-cut tolerance without weakening generic mode."""
    short, long = metrics.rectangle_size_mm
    return (
        1 <= metrics.connected_components <= len(template.polygons)
        and config.target_short_min_mm
        <= short
        <= config.target_short_max_mm
        and config.target_long_min_mm
        <= long
        <= config.target_long_max_mm
        and metrics.fill_ratio
        >= min(config.min_rectangle_fill_ratio, 0.89)
        and metrics.overlap_ratio
        <= max(config.max_overlap_ratio, 0.08)
        and metrics.hole_ratio <= max(config.max_hole_ratio, 0.03)
        and metrics.boundary_p95_mm
        <= max(
            config.max_boundary_p95_mm,
            template.maximum_boundary_p95_mm,
        )
        and metrics.boundary_p99_mm
        <= max(
            config.max_boundary_p99_mm,
            template.maximum_boundary_p99_mm,
        )
        and metrics.side_coverage_min
        >= min(
            config.min_rectangle_side_coverage,
            template.minimum_side_coverage,
        )
        and metrics.convexity_ratio
        >= min(config.min_rectangle_convexity_ratio, 0.93)
        and metrics.corner_error_max_mm
        <= max(config.max_rectangle_corner_error_mm, 7.0)
    )


def solve_trained_template_bank(
    pieces: Sequence[PieceObservation],
    paper_bgr: np.ndarray,
    config: VisionConfig,
    use_texture: bool = True,
) -> AssemblySolution:
    """Match the four user-trained cuts before entering generic enumeration."""
    matching_templates = [
        template
        for template in trained_template_bank()
        if len(template.polygons) == len(pieces)
    ]
    if not matching_templates:
        raise SolveError(
            "trained template bank has no layout for this piece count"
        )

    from .solver import _geometry_score, _raster_metrics

    candidates = []
    explored = 0
    for template in matching_templates:
        assignments, template_explored = _trained_assignments(
            pieces, template, config
        )
        explored += template_explored
        for assignment in assignments:
            rank_separation = 0.0
            rank_corner_error = 0.0
            rank_opposite_corner_error = 0.0
            transforms = _layout_transforms(
                pieces, assignment, (template.polygons,)
            )
            rank_semantics_required = (
                template.center_symmetric_rank and use_texture
            )
            if rank_semantics_required:
                (
                    rank_valid,
                    rank_separation,
                    rank_corner_error,
                    rank_opposite_corner_error,
                ) = (
                    _center_symmetric_rank_is_valid(
                        pieces,
                        assignment,
                        transforms,
                        paper_bgr,
                        config.px_per_mm,
                        template,
                    )
                )
                if not rank_valid:
                    continue
            polygons = [
                transform_points(
                    piece.polygon_mm,
                    transforms[piece.piece_id],
                )
                for piece in pieces
            ]
            metrics = _raster_metrics(polygons, config)
            if not _trained_metrics_are_valid(
                metrics, template, config
            ):
                continue
            candidates.append(
                (
                    assignment.rms_mm,
                    template.name,
                    rank_semantics_required,
                    rank_separation,
                    rank_corner_error,
                    rank_opposite_corner_error,
                    assignment,
                    transforms,
                    metrics,
                )
            )
    if not candidates:
        raise SolveError(
            "observed pieces do not match a trained layout with all "
            "template-specific constraints"
        )

    candidates.sort(key=lambda item: item[0])
    (
        best_rms,
        template_name,
        center_symmetric_rank,
        rank_separation,
        rank_corner_error,
        rank_opposite_corner_error,
        best_assignment,
        transforms,
        metrics,
    ) = candidates[0]
    geometry_score = _geometry_score(metrics, config)
    margin = (
        candidates[1][0] - best_rms
        if len(candidates) > 1
        else None
    )
    semantic_detail = (
        f", K-centre-symmetry=yes, ink-separation={rank_separation:.3f}, "
        f"rank-corners={rank_corner_error:.1f}/"
        f"{rank_opposite_corner_error:.1f} mm"
        if center_symmetric_rank
        else ""
    )
    print(
        f"[vision] trained template {template_name} selected "
        f"(RMS={best_rms:.2f} mm, fill={metrics.fill_ratio:.3f}, "
        f"side={metrics.side_coverage_min:.3f}{semantic_detail})",
        flush=True,
    )
    center_symmetric_piece_pair = None
    if center_symmetric_rank:
        piece_by_template = {
            template_index: pieces[piece_index].piece_id
            for piece_index, template_index in enumerate(
                best_assignment.template_indices
            )
        }
        selected_template = next(
            template
            for template in matching_templates
            if template.name == template_name
        )
        center_symmetric_piece_pair = (
            piece_by_template[selected_template.rank_triangle_index],
            piece_by_template[
                selected_template.opposite_rank_piece_index
            ],
        )
    return AssemblySolution(
        transforms=transforms,
        metrics=metrics,
        geometry_score=geometry_score,
        texture_score=0.0,
        total_score=geometry_score,
        explored_nodes=explored,
        valid_candidate_count=len(candidates),
        search_complete=True,
        score_margin=margin,
        preserve_center_symmetry=center_symmetric_rank,
        center_symmetric_piece_pair=center_symmetric_piece_pair,
        trained_template_name=template_name,
    )


def solve_figure2_template(
    pieces: Sequence[PieceObservation],
    config: VisionConfig,
) -> AssemblySolution:
    """Solve the confirmed four-piece Figure-2 without geometric enumeration.

    The two possible manufacturing cut hands are classified, but all four
    pieces must use one uniform hand.  Every emitted pose is a proper rotation
    plus translation; classifying the drawing's opposite hand never reflects
    a physical piece.  The outer rectangle and corner topology are fixed
    first, then noisy internal seams use the contest's 20 mm rule.
    """
    if len(pieces) != 4:
        raise SolveError("Figure-2 template solver requires exactly 4 pieces")
    piece_ids = [piece.piece_id for piece in pieces]
    if len(set(piece_ids)) != 4:
        raise SolveError("Figure-2 piece ids must be unique")

    templates = figure2_template_polygons()
    template_hands = (
        templates,
        _opposite_cut_hand_templates(templates),
    )
    cache = _fit_cache(pieces, template_hands, config)
    assignments, explored = _global_assignments(pieces, cache)
    strict_usable = [
        assignment
        for assignment in assignments
        if _assignment_is_usable(assignment, config)
    ]
    usable = strict_usable
    if not strict_usable:
        topology_candidates = [
            assignment
            for assignment in assignments
            if _assignment_meets_vertex_rule(
                pieces, assignment, template_hands
            )
            and _assignment_has_confirmed_topology(
                pieces, assignment, template_hands, config
            )
        ]
        if topology_candidates:
            candidate = topology_candidates[0]
            candidate_transforms = _seam_locked_transforms(
                pieces, candidate, template_hands
            )
            from .solver import _raster_metrics

            candidate_metrics = _raster_metrics(
                [
                    transform_points(
                        piece.polygon_mm,
                        candidate_transforms[piece.piece_id],
                    )
                    for piece in pieces
                ],
                config,
            )
            maximum_gap = float(
                np.max(
                    _topology_corresponding_gaps(
                        pieces,
                        candidate,
                        template_hands,
                        candidate_transforms,
                    )
                )
            )
            raise SolveError(
                "same-face Figure-2 seam topology was found "
                f"(max adjacent gap={maximum_gap:.1f} mm), but its outer "
                "boundary is not a valid rectangle: "
                f"fill={candidate_metrics.fill_ratio:.3f}, "
                f"side={candidate_metrics.side_coverage_min:.3f}, "
                f"corner={candidate_metrics.corner_error_max_mm:.1f} mm; "
                "no piece reflection was attempted"
            )
    if not usable:
        detail = (
            "no vertex-compatible assignment"
            if not assignments
            else f"best rotation-only RMS={assignments[0].rms_mm:.2f} mm"
        )
        raise SolveError(
            "observed pieces do not match the Figure-2 template; " + detail
        )

    from .solver import _geometry_score, _raster_metrics

    best: Optional[_Assignment] = None
    best_index: Optional[int] = None
    transforms: Dict[int, np.ndarray] = {}
    metrics = None
    rejected_metrics = None
    rejected_score = float("inf")
    rejected_variant = ""
    selected_variant = ""
    for assignment_index, assignment in enumerate(usable):
        valid_variants = []
        variants = (
            (
                "independent",
                _layout_transforms(
                    pieces, assignment, template_hands
                ),
            ),
            (
                "seam-locked",
                _seam_locked_transforms(
                    pieces, assignment, template_hands
                ),
            ),
        )
        for variant_name, candidate_transforms in variants:
            candidate_polygons = [
                transform_points(
                    piece.polygon_mm,
                    candidate_transforms[piece.piece_id],
                )
                for piece in pieces
            ]
            candidate_metrics = _raster_metrics(
                candidate_polygons, config
            )
            candidate_score = _geometry_score(
                candidate_metrics, config
            )
            if _template_metrics_are_valid(
                candidate_metrics, config
            ):
                valid_variants.append(
                    (
                        candidate_score,
                        variant_name,
                        candidate_transforms,
                        candidate_metrics,
                    )
                )
            elif candidate_score < rejected_score:
                rejected_score = candidate_score
                rejected_metrics = candidate_metrics
                rejected_variant = variant_name
        if valid_variants:
            (
                _,
                selected_variant,
                transforms,
                metrics,
            ) = min(valid_variants, key=lambda item: item[0])
            best = assignment
            best_index = assignment_index
            break
    if best is None or best_index is None or metrics is None:
        if rejected_metrics is None:
            detail = "no rigid layout could be measured"
        else:
            short, long = rejected_metrics.rectangle_size_mm
            detail = (
                f"best={rejected_variant}, "
                f"fill={rejected_metrics.fill_ratio:.3f}, "
                f"overlap={rejected_metrics.overlap_ratio:.3f}, "
                f"hole={rejected_metrics.hole_ratio:.3f}, "
                f"side={rejected_metrics.side_coverage_min:.3f}, "
                f"convexity={rejected_metrics.convexity_ratio:.3f}, "
                f"corner={rejected_metrics.corner_error_max_mm:.1f} mm, "
                f"boundary={rejected_metrics.boundary_p95_mm:.1f}/"
                f"{rejected_metrics.boundary_p99_mm:.1f} mm, "
                f"components={rejected_metrics.connected_components}, "
                f"size={short:.1f}x{long:.1f} mm"
            )
        raise SolveError(
            "Figure-2 shapes matched, but the rigid target layout failed "
            "the rectangle checks: " + detail
        )
    if any(
        float(np.linalg.det(matrix[:2, :2])) <= 0.0
        for matrix in transforms.values()
    ):
        raise SolveError("Figure-2 fit attempted an internal reflection")

    print(
        "[vision] Figure-2 selected "
        f"{selected_variant} layout: "
        f"fill={metrics.fill_ratio:.3f}, "
        f"side={metrics.side_coverage_min:.3f}, "
        f"boundary={metrics.boundary_p95_mm:.1f}/"
        f"{metrics.boundary_p99_mm:.1f} mm",
        flush=True,
    )

    geometry_score = _geometry_score(metrics, config)
    margin = (
        usable[best_index + 1].rms_mm - best.rms_mm
        if best_index + 1 < len(usable)
        else None
    )
    return AssemblySolution(
        transforms=transforms,
        metrics=metrics,
        geometry_score=geometry_score,
        texture_score=0.0,
        total_score=geometry_score,
        explored_nodes=explored,
        valid_candidate_count=len(usable),
        search_complete=True,
        score_margin=margin,
        placement_spread_scale=1.25,
    )
