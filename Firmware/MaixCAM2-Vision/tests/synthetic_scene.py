from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from puzzle_vision.config import VisionConfig
from puzzle_vision.geometry import (
    PieceObservation,
    ensure_positive_winding,
    polygon_area,
    rigid_matrix,
    transform_points,
)


def figure2_target_polygons() -> List[np.ndarray]:
    # Origin is the upper-left corner of the 10 cm x 6 cm target rectangle.
    a = np.array([0.0, 0.0])
    b = np.array([20.0, 0.0])
    top_right = np.array([100.0, 0.0])
    bottom_right = np.array([100.0, 60.0])
    bottom_left = np.array([0.0, 60.0])
    d = np.array([0.0, 20.0])
    e = np.array([0.0, 30.0])
    # B -> bottom-right is a 6-8-10 line. The marked seam lengths are 2 cm
    # from B and 3 cm from the bottom-right endpoint.
    c = np.array([36.0, 12.0])
    f = np.array([76.0, 42.0])
    return [
        ensure_positive_winding(np.array([a, b, c, d])),
        ensure_positive_winding(np.array([b, top_right, bottom_right])),
        ensure_positive_winding(np.array([d, c, f, e])),
        ensure_positive_winding(
            np.array([e, f, bottom_right, bottom_left])
        ),
    ]


def _pose_in_source(
    polygon: np.ndarray, center: Tuple[float, float], angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    centroid = polygon.mean(axis=0)
    angle = np.radians(angle_deg)
    rotation = rigid_matrix(angle, np.zeros(2))
    centered = polygon - centroid
    rotated = transform_points(centered, rotation)
    translation = np.asarray(center, dtype=np.float64) - rotated.mean(axis=0)
    matrix = rigid_matrix(angle, translation - rotation[:2, :2] @ centroid)
    return transform_points(polygon, matrix), matrix


def make_rectified_scene(
    config: VisionConfig,
    textured: bool = False,
    divider_bgr: Tuple[int, int, int] = (20, 20, 20),
    paper_bgr: Tuple[int, int, int] = (170, 105, 45),
) -> Tuple[np.ndarray, List[PieceObservation]]:
    width = int(round(config.paper_short_mm * config.px_per_mm))
    height = int(round(config.paper_long_mm * config.px_per_mm))
    paper_bgr_image = np.full(
        (height, width, 3), paper_bgr, dtype=np.uint8
    )
    divider_y = height // 2
    cv2.line(
        paper_bgr_image,
        (0, divider_y),
        (width - 1, divider_y),
        divider_bgr,
        max(2, int(round(2.0 * config.px_per_mm))),
    )

    targets = figure2_target_polygons()
    centers = [(35.0, 34.0), (137.0, 42.0), (45.0, 105.0), (148.0, 112.0)]
    angles = [25.0, 0.0, -5.0, 3.0]
    observations: List[PieceObservation] = []

    for piece_id, (target, center, angle) in enumerate(
        zip(targets, centers, angles)
    ):
        source, _ = _pose_in_source(target, center, angle)
        minimum = source.min(axis=0)
        maximum = source.max(axis=0)
        correction = np.zeros(2)
        if minimum[0] < 6.0:
            correction[0] += 6.0 - minimum[0]
        if maximum[0] > config.paper_short_mm - 6.0:
            correction[0] -= maximum[0] - (config.paper_short_mm - 6.0)
        if minimum[1] < 6.0:
            correction[1] += 6.0 - minimum[1]
        source_limit = 0.5 * config.paper_long_mm - 8.0
        if maximum[1] > source_limit:
            correction[1] -= maximum[1] - source_limit
        source = source + correction

        mask = np.zeros((height, width), dtype=np.uint8)
        source_px = np.round(source * config.px_per_mm).astype(np.int32)
        cv2.fillPoly(mask, [source_px], 255)
        color = (238, 238, 238) if not textured else (245, 245, 245)
        cv2.fillPoly(paper_bgr_image, [source_px], color)
        if textured:
            center_px = tuple(
                np.round(source.mean(axis=0) * config.px_per_mm).astype(int)
            )
            cv2.putText(
                paper_bgr_image,
                "A" if piece_id % 2 == 0 else "K",
                (center_px[0] - 14, center_px[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 210 if piece_id % 2 == 0 else 20),
                3,
                cv2.LINE_AA,
            )
        moments = cv2.moments(mask, binaryImage=True)
        center_mm = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        ) / config.px_per_mm
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, _, _, pickup_px = cv2.minMaxLoc(distance)
        observations.append(
            PieceObservation(
                piece_id=piece_id,
                contour_px=source_px.astype(np.float64),
                polygon_mm=ensure_positive_winding(source),
                mask=mask,
                source_center_mm=center_mm,
                pickup_mm=np.array(pickup_px, dtype=np.float64)
                / config.px_per_mm,
                area_mm2=polygon_area(source),
                approximation_error_mm=0.0,
            )
        )

    return paper_bgr_image, observations


def make_single_piece_orientation_scene(
    config: VisionConfig,
    *,
    piece_half: str = "bottom",
    balanced_shadows: bool = False,
    upper_shadow: bool = False,
) -> np.ndarray:
    """Make a non-white A4 scene for testing source-half orientation."""
    if piece_half not in ("top", "bottom"):
        raise ValueError("piece_half must be 'top' or 'bottom'")

    width = int(round(config.paper_short_mm * config.px_per_mm))
    height = int(round(config.paper_long_mm * config.px_per_mm))
    paper = np.full((height, width, 3), (170, 105, 45), dtype=np.uint8)

    yy, xx = np.mgrid[0:height, 0:width]
    shadow = np.zeros((height, width), dtype=np.float32)
    if balanced_shadows:
        for center_y in (0.25 * height, 0.75 * height):
            shadow += np.exp(
                -0.5
                * (
                    ((xx - 0.28 * width) / (0.18 * width)) ** 2
                    + ((yy - center_y) / (0.10 * height)) ** 2
                )
            )
    if upper_shadow:
        shadow += np.exp(
            -0.5
            * (
                ((xx - 0.50 * width) / (0.31 * width)) ** 2
                + ((yy - 0.25 * height) / (0.16 * height)) ** 2
            )
        )
    paper = np.clip(
        paper.astype(np.float32) - 20.0 * shadow[:, :, None],
        0,
        255,
    ).astype(np.uint8)

    divider_y = height // 2
    cv2.line(
        paper,
        (0, divider_y),
        (width - 1, divider_y),
        (20, 20, 20),
        max(2, int(round(2.0 * config.px_per_mm))),
    )

    center_y_mm = 70.0 if piece_half == "top" else 227.0
    center_mm = np.array([105.0, center_y_mm])
    half_size_mm = np.array([7.0, 5.0])
    corner_min = np.round(
        (center_mm - half_size_mm) * config.px_per_mm
    ).astype(int)
    corner_max = np.round(
        (center_mm + half_size_mm) * config.px_per_mm
    ).astype(int)
    cv2.rectangle(
        paper,
        tuple(corner_min),
        tuple(corner_max),
        (245, 245, 245),
        -1,
    )
    return paper


def perspective_camera_frame(
    rectified_bgr: np.ndarray,
    frame_size: Tuple[int, int] = (1280, 720),
    landscape: bool = False,
    destination: np.ndarray | None = None,
    background_bgr: Tuple[int, int, int] = (62, 68, 73),
    outer_quad: np.ndarray | None = None,
) -> np.ndarray:
    frame_width, frame_height = frame_size
    frame = np.full(
        (frame_height, frame_width, 3), background_bgr, dtype=np.uint8
    )
    if outer_quad is not None:
        cv2.polylines(
            frame,
            [np.round(outer_quad).astype(np.int32)],
            True,
            (28, 32, 36),
            9,
            cv2.LINE_AA,
        )
    if landscape:
        rectified_bgr = cv2.rotate(
            rectified_bgr, cv2.ROTATE_90_CLOCKWISE
        )
    source = np.array(
        [
            [0, 0],
            [rectified_bgr.shape[1] - 1, 0],
            [rectified_bgr.shape[1] - 1, rectified_bgr.shape[0] - 1],
            [0, rectified_bgr.shape[0] - 1],
        ],
        dtype=np.float32,
    )
    if destination is None:
        if landscape:
            destination = np.array(
                [[185, 92], [1095, 76], [1120, 648], [160, 665]],
                dtype=np.float32,
            )
        else:
            destination = np.array(
                [[445, 18], [830, 32], [902, 692], [370, 677]],
                dtype=np.float32,
            )
    else:
        destination = np.asarray(destination, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        rectified_bgr, matrix, (frame_width, frame_height)
    )
    mask = cv2.warpPerspective(
        np.full(rectified_bgr.shape[:2], 255, dtype=np.uint8),
        matrix,
        (frame_width, frame_height),
    )
    frame[mask > 0] = warped[mask > 0]
    return frame


def add_paper_edge_artifacts(
    frame_bgr: np.ndarray,
    paper_quad: np.ndarray,
    background_bgr: Tuple[int, int, int],
) -> np.ndarray:
    """Add a faint parallel edge and short occlusions to two paper sides."""
    result = frame_bgr.copy()
    quad = np.asarray(paper_quad, dtype=np.float64).reshape(4, 2)

    for edge_index in (0, 3):
        start = quad[edge_index]
        end = quad[(edge_index + 1) % 4]
        direction = end - start
        length = float(np.linalg.norm(direction))
        tangent = direction / length
        outward = np.array([tangent[1], -tangent[0]])
        duplicate_start = np.round(start + 4.0 * outward).astype(int)
        duplicate_end = np.round(end + 4.0 * outward).astype(int)
        cv2.line(
            result,
            tuple(duplicate_start),
            tuple(duplicate_end),
            (38, 42, 46),
            2,
            cv2.LINE_AA,
        )

        midpoint = start + 0.58 * direction
        half_gap = 0.045 * length * tangent
        cv2.line(
            result,
            tuple(np.round(midpoint - half_gap).astype(int)),
            tuple(np.round(midpoint + half_gap).astype(int)),
            background_bgr,
            11,
            cv2.LINE_AA,
        )

    return result


def make_ambiguous_textured_pair(
    config: VisionConfig,
) -> Tuple[np.ndarray, List[PieceObservation], Dict[int, np.ndarray]]:
    """Two identical rectangles whose left/right order is decided by texture."""
    width = int(round(config.paper_short_mm * config.px_per_mm))
    height = int(round(config.paper_long_mm * config.px_per_mm))
    paper_bgr = np.full((height, width, 3), (170, 105, 45), dtype=np.uint8)

    scale = config.px_per_mm
    card_width = int(round(100.0 * scale)) + 1
    card_height = int(round(60.0 * scale)) + 1
    card = np.full((card_height, card_width, 3), 245, dtype=np.uint8)
    # Only about 10% of the 60 mm seam carries useful artwork.  This guards
    # against texture evidence being diluted by the much longer blank seam.
    cv2.circle(
        card,
        (int(round(50.0 * scale)), int(round(17.0 * scale))),
        int(round(3.0 * scale)),
        (210, 45, 35),
        -1,
        cv2.LINE_AA,
    )

    target_polygons = [
        np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 60.0], [0.0, 60.0]]),
        np.array(
            [[50.0, 0.0], [100.0, 0.0], [100.0, 60.0], [50.0, 60.0]]
        ),
    ]
    centers = [(55.0, 61.0), (151.0, 79.0)]
    angles = [18.0, -23.0]
    observations: List[PieceObservation] = []
    source_matrices: Dict[int, np.ndarray] = {}
    pixel_scale = np.diag([scale, scale, 1.0])

    for piece_id, (target, center, angle) in enumerate(
        zip(target_polygons, centers, angles)
    ):
        source, source_matrix = _pose_in_source(target, center, angle)
        source_matrices[piece_id] = source_matrix
        target_mask = np.zeros(card.shape[:2], dtype=np.uint8)
        cv2.fillPoly(
            target_mask,
            [np.round(target * scale).astype(np.int32)],
            255,
        )
        warp_matrix = pixel_scale @ source_matrix @ np.linalg.inv(pixel_scale)
        source_texture = cv2.warpPerspective(
            card,
            warp_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
        )
        source_mask = cv2.warpPerspective(
            target_mask,
            warp_matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
        paper_bgr[source_mask > 0] = source_texture[source_mask > 0]

        moments = cv2.moments(source_mask, binaryImage=True)
        center_mm = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        ) / scale
        distance = cv2.distanceTransform(source_mask, cv2.DIST_L2, 5)
        _, _, _, pickup_px = cv2.minMaxLoc(distance)
        observations.append(
            PieceObservation(
                piece_id=piece_id,
                contour_px=np.round(source * scale).astype(np.float64),
                polygon_mm=ensure_positive_winding(source),
                mask=source_mask,
                source_center_mm=center_mm,
                pickup_mm=np.asarray(pickup_px, dtype=np.float64) / scale,
                area_mm2=polygon_area(source),
                approximation_error_mm=0.0,
            )
        )
    return paper_bgr, observations, source_matrices


def make_quartered_textured_card(
    config: VisionConfig,
    *,
    center_mark: bool = True,
    piece_id_order: Sequence[int] = (0, 1, 2, 3),
    strip_heights_mm: Sequence[float] | None = None,
) -> Tuple[np.ndarray, List[PieceObservation], Dict[int, np.ndarray]]:
    """Four card parts whose geometry has many rectangle solutions."""
    width = int(round(config.paper_short_mm * config.px_per_mm))
    height = int(round(config.paper_long_mm * config.px_per_mm))
    paper_bgr = np.full(
        (height, width, 3), (170, 105, 45), dtype=np.uint8
    )

    scale = config.px_per_mm
    card_width_mm = 60.0
    card_height_mm = 90.0
    card = np.full(
        (
            int(round(card_height_mm * scale)) + 1,
            int(round(card_width_mm * scale)) + 1,
            3,
        ),
        245,
        dtype=np.uint8,
    )
    ink = np.full_like(card, 245)
    cv2.putText(
        ink,
        "7",
        (int(3 * scale), int(14 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(
        ink,
        (int(46 * scale), int(20 * scale)),
        int(4 * scale),
        (30, 30, 210),
        -1,
        cv2.LINE_AA,
    )
    printed = np.any(ink < 235, axis=2)
    card[printed] = ink[printed]
    opposite = cv2.rotate(ink, cv2.ROTATE_180)
    printed = np.any(opposite < 235, axis=2)
    card[printed] = opposite[printed]
    if center_mark:
        center_x = int(round(0.5 * card_width_mm * scale))
        center_y = int(round(0.5 * card_height_mm * scale))
        cv2.fillPoly(
            card,
            [
                np.array(
                    [
                        [center_x, center_y - int(8 * scale)],
                        [center_x + int(7 * scale), center_y],
                        [center_x + int(2 * scale), center_y],
                        [center_x + int(2 * scale), center_y + int(8 * scale)],
                        [center_x - int(2 * scale), center_y + int(8 * scale)],
                        [center_x - int(2 * scale), center_y],
                        [center_x - int(7 * scale), center_y],
                    ],
                    dtype=np.int32,
                )
            ],
            (25, 25, 25),
            cv2.LINE_AA,
        )

    if strip_heights_mm is None:
        target_polygons = [
            np.array(
                [[0.0, 0.0], [30.0, 0.0], [30.0, 45.0], [0.0, 45.0]]
            ),
            np.array(
                [[30.0, 0.0], [60.0, 0.0], [60.0, 45.0], [30.0, 45.0]]
            ),
            np.array(
                [
                    [30.0, 45.0],
                    [60.0, 45.0],
                    [60.0, 90.0],
                    [30.0, 90.0],
                ]
            ),
            np.array(
                [[0.0, 45.0], [30.0, 45.0], [30.0, 90.0], [0.0, 90.0]]
            ),
        ]
    else:
        heights = np.asarray(strip_heights_mm, dtype=np.float64)
        if heights.shape != (4,) or not np.isclose(
            float(np.sum(heights)), card_height_mm
        ):
            raise ValueError(
                "strip_heights_mm must contain four values summing to 90"
            )
        boundaries = np.concatenate([[0.0], np.cumsum(heights)])
        target_polygons = [
            np.array(
                [
                    [0.0, boundaries[index]],
                    [card_width_mm, boundaries[index]],
                    [card_width_mm, boundaries[index + 1]],
                    [0.0, boundaries[index + 1]],
                ]
            )
            for index in range(4)
        ]
    centers = [(25.0, 32.0), (78.0, 37.0), (130.0, 31.0), (181.0, 36.0)]
    angles = [13.0, -19.0, 24.0, -11.0]
    observations: List[PieceObservation] = []
    source_matrices: Dict[int, np.ndarray] = {}
    pixel_scale = np.diag([scale, scale, 1.0])

    for target_index, (target, center, angle) in enumerate(
        zip(target_polygons, centers, angles)
    ):
        piece_id = int(piece_id_order[target_index])
        source, source_matrix = _pose_in_source(target, center, angle)
        source_matrices[piece_id] = source_matrix
        target_mask = np.zeros(card.shape[:2], dtype=np.uint8)
        cv2.fillPoly(
            target_mask,
            [np.round(target * scale).astype(np.int32)],
            255,
        )
        warp_matrix = (
            pixel_scale @ source_matrix @ np.linalg.inv(pixel_scale)
        )
        source_texture = cv2.warpPerspective(
            card,
            warp_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
        )
        source_mask = cv2.warpPerspective(
            target_mask,
            warp_matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
        paper_bgr[source_mask > 0] = source_texture[source_mask > 0]

        moments = cv2.moments(source_mask, binaryImage=True)
        center_mm = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        ) / scale
        distance = cv2.distanceTransform(source_mask, cv2.DIST_L2, 5)
        _, _, _, pickup_px = cv2.minMaxLoc(distance)
        observations.append(
            PieceObservation(
                piece_id=piece_id,
                contour_px=np.round(source * scale).astype(np.float64),
                polygon_mm=ensure_positive_winding(source),
                mask=source_mask,
                source_center_mm=center_mm,
                pickup_mm=np.asarray(pickup_px, dtype=np.float64) / scale,
                area_mm2=polygon_area(source),
                approximation_error_mm=0.0,
            )
        )
    return paper_bgr, observations, source_matrices
