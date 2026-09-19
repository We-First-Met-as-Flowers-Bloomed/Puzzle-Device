from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import VisionConfig
from .geometry import (
    DetectionError,
    PaperObservation,
    PieceObservation,
    ensure_positive_winding,
    order_quad,
    polygon_area,
)


@dataclass
class _PaperCandidate:
    quad: np.ndarray
    score: float
    divider_coverage: float
    landscape: bool
    reverse: bool
    preview_score: float
    preview_plausible: bool
    preview_piece_count: int
    preview_piece_area_mm2: float
    fixed: bool = False


@dataclass
class _PreviewEvidence:
    score: float
    plausible: bool
    landscape: bool
    reverse: bool
    piece_count: int
    piece_area_mm2: float


def _otsu_distance_threshold(values: np.ndarray) -> float:
    """Return an Otsu threshold while keeping Lab distances in native units."""
    if values.size == 0:
        return 0.0
    values_u8 = np.clip(values, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(
        values_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return float(threshold)


def _chroma_noise_threshold(
    lower_chroma_distance: np.ndarray, config: VisionConfig
) -> float:
    """Threshold colour contrast without being dominated by illumination."""
    baseline = float(np.median(lower_chroma_distance))
    mad = float(
        np.median(np.abs(lower_chroma_distance - baseline))
    )
    return max(
        4.0,
        baseline
        + config.foreground_mad_scale * max(mad, 0.2),
    )


def undistort_frame(frame_bgr: np.ndarray, config: VisionConfig) -> np.ndarray:
    if config.camera_matrix is None or config.distortion_coefficients is None:
        return frame_bgr
    matrix = np.asarray(config.camera_matrix, dtype=np.float64)
    distortion = np.asarray(config.distortion_coefficients, dtype=np.float64)
    return cv2.undistort(frame_bgr, matrix, distortion)


def _quad_geometry(quad: np.ndarray) -> Tuple[float, float, float]:
    q = order_quad(quad)
    horizontal = 0.5 * (
        np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
    )
    vertical = 0.5 * (
        np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])
    )
    short = min(horizontal, vertical)
    long = max(horizontal, vertical)
    return horizontal, vertical, long / max(short, 1e-6)


def _gradient_edges(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    channels = cv2.split(cv2.GaussianBlur(lab, (5, 5), 0))
    magnitude = np.zeros(image_bgr.shape[:2], dtype=np.float32)
    for channel in channels:
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.maximum(magnitude, cv2.magnitude(gx, gy))
    scale = 255.0 / max(float(np.percentile(magnitude, 99.5)), 1.0)
    magnitude_u8 = np.clip(magnitude * scale, 0, 255).astype(np.uint8)
    _, edges = cv2.threshold(
        magnitude_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )
    return edges


def _odd_size(value: float, minimum: int = 3, maximum: int = 31) -> int:
    size = int(round(value))
    size = max(minimum, min(maximum, size))
    if size % 2 == 0:
        size += 1
    return min(size, maximum if maximum % 2 == 1 else maximum - 1)


def _clean_region_mask(mask: np.ndarray) -> np.ndarray:
    """Join a divider-width gap without rounding away the A4 corners."""
    minimum_side = min(mask.shape[:2])
    close_size = _odd_size(0.018 * minimum_side, minimum=5, maximum=21)
    open_size = _odd_size(0.004 * minimum_side, minimum=3, maximum=7)
    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_size, close_size)
        ),
    )
    return cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (open_size, open_size)
        ),
    )


def _color_region_masks(image_bgr: np.ndarray) -> List[np.ndarray]:
    """Return paper-region hypotheses independent of the paper's color.

    The competition permits any pure-color A4 sheet and any platform color.
    A border-derived Lab-distance mask handles the normal contrasting-platform
    setup.  Per-channel Otsu masks provide a fallback when the useful contrast
    appears mainly in only one Lab channel.
    """
    lab = cv2.cvtColor(
        cv2.GaussianBlur(image_bgr, (7, 7), 0), cv2.COLOR_BGR2LAB
    )
    h, w = lab.shape[:2]
    band = max(4, int(round(0.035 * min(h, w))))
    border_pixels = np.concatenate(
        [
            lab[:band, :].reshape(-1, 3),
            lab[h - band :, :].reshape(-1, 3),
            lab[band : h - band, :band].reshape(-1, 3),
            lab[band : h - band, w - band :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    background = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(lab.astype(np.float32) - background, axis=2)
    border_distance = np.linalg.norm(border_pixels - background, axis=1)
    adaptive_threshold = max(
        6.0, float(np.percentile(border_distance, 97.5)) + 2.0
    )
    distance_u8 = np.clip(distance, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(
        distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    masks: List[np.ndarray] = []
    thresholds = {
        int(round(adaptive_threshold)),
        int(round(max(6.0, 0.80 * float(otsu_threshold)))),
        int(round(max(6.0, float(otsu_threshold)))),
    }
    for threshold in sorted(thresholds):
        masks.append(
            _clean_region_mask(
                (distance > float(threshold)).astype(np.uint8) * 255
            )
        )

    for channel in cv2.split(lab):
        if int(np.percentile(channel, 95)) - int(np.percentile(channel, 5)) < 8:
            continue
        _, positive = cv2.threshold(
            channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        masks.append(_clean_region_mask(positive))
        masks.append(_clean_region_mask(cv2.bitwise_not(positive)))
    return masks


def _line_response(
    warped_bgr: np.ndarray,
    horizontal_line: bool,
    config: VisionConfig,
) -> Tuple[float, int]:
    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if not horizontal_line:
        lab = np.transpose(lab, (1, 0, 2))
    h, w = lab.shape[:2]
    border = max(3, int(round(min(h, w) * 0.04)))
    core = lab[border : h - border, border : w - border]
    background = np.median(core.reshape(-1, 3), axis=0)
    distance = np.linalg.norm(lab - background, axis=2)

    baseline = float(np.median(distance))
    mad = float(np.median(np.abs(distance - baseline)))
    threshold = max(7.0, baseline + 5.0 * max(mad, 0.5))
    start = int(config.divider_search_min * h)
    end = int(config.divider_search_max * h)
    strip_distance = distance[start:end, border : w - border]
    strip_lightness = lab[start:end, border : w - border, 0]
    if strip_distance.size == 0:
        return 0.0, (start + end) // 2

    # Candidate previews are intentionally warped at about 1 px/mm while the
    # final paper image normally uses config.px_per_mm.  Derive the local
    # scale from this image so a 2 mm divider is treated consistently in both.
    local_px_per_mm = min(
        min(w, h) / config.paper_short_mm,
        max(w, h) / config.paper_long_mm,
    )
    core_lightness = lab[
        border : h - border, border : w - border, 0
    ]
    # Use a per-column paper reference.  White pieces are brighter and cannot
    # create a false black-line response; a black hand-drawn divider remains
    # dark even under a smooth illumination gradient.
    column_reference = np.percentile(
        core_lightness, 60.0, axis=0
    ).astype(np.float32)
    dark_delta = column_reference[None, :] - strip_lightness
    dark_baseline = float(np.median(dark_delta))
    dark_mad = float(
        np.median(np.abs(dark_delta - dark_baseline))
    )
    dark_threshold = max(
        6.0, dark_baseline + 5.0 * max(dark_mad, 0.5)
    )

    local_window = max(3, int(round(8.0 * local_px_per_mm)))
    if local_window % 2 == 0:
        local_window += 1
    local_lightness = cv2.blur(
        lab[:, :, 0],
        (1, local_window),
        borderType=cv2.BORDER_REPLICATE,
    )[start:end, border : w - border]
    thin_dark = (local_lightness - strip_lightness) > 4.0
    line_mask = (
        (strip_distance > threshold)
        | (dark_delta > dark_threshold)
        | thin_dark
    ).astype(np.uint8)

    # Join small pen gaps along the expected line direction.  Coverage alone
    # is insufficient: a long fragment edge may cover half the paper width,
    # while the real divider spans almost the complete sheet even if broken.
    gap_width = max(3, int(round(8.0 * local_px_per_mm)))
    closed = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_CLOSE,
        np.ones((1, gap_width), dtype=np.uint8),
    )
    raw_coverage = line_mask.mean(axis=1).astype(np.float32)
    closed_coverage = closed.mean(axis=1).astype(np.float32)
    occupied = line_mask > 0
    has_pixel = occupied.any(axis=1)
    span = np.zeros(len(line_mask), dtype=np.float32)
    if np.any(has_pixel):
        first = np.argmax(occupied, axis=1)
        last = occupied.shape[1] - 1 - np.argmax(
            occupied[:, ::-1], axis=1
        )
        span[has_pixel] = (
            last[has_pixel] - first[has_pixel] + 1
        ) / max(occupied.shape[1], 1)
    coverage = (
        0.55 * raw_coverage
        + 0.30 * closed_coverage
        + 0.15 * span
    )

    smooth_width = max(1, int(round(2.0 * local_px_per_mm)))
    kernel = np.ones(smooth_width, dtype=np.float32) / smooth_width
    smoothed = np.convolve(coverage.astype(np.float32), kernel, mode="same")
    surround_width = max(
        smooth_width + 2, int(round(12.0 * local_px_per_mm))
    )
    surround_kernel = (
        np.ones(surround_width, dtype=np.float32) / surround_width
    )
    surround = np.convolve(
        smoothed, surround_kernel, mode="same"
    )
    prominence = np.maximum(0.0, smoothed - surround)

    rows = np.arange(start, end, dtype=np.float32)
    fractions = rows / max(float(h - 1), 1.0)
    center_prior = np.exp(
        -0.5 * ((fractions - 0.5) / 0.10) ** 2
    )
    score = (
        0.55 * smoothed + 0.45 * prominence
    ) * (0.45 + 0.55 * center_prior)
    index = int(np.argmax(score))
    return float(np.clip(score[index], 0.0, 1.0)), start + index


def _warp_candidate_preview(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    config: VisionConfig,
    landscape: bool,
) -> np.ndarray:
    q = order_quad(quad)
    short_px = max(160, int(round(config.paper_short_mm)))
    long_px = max(220, int(round(config.paper_long_mm)))
    width, height = (long_px, short_px) if landscape else (short_px, long_px)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(q, destination)
    return cv2.warpPerspective(frame_bgr, matrix, (width, height))


def _candidate_quads(binary: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    quads: List[Tuple[np.ndarray, float]] = []
    for contour in sorted(
        contours, key=lambda item: abs(cv2.contourArea(item)), reverse=True
    )[:100]:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 80:
            continue
        contour_area = abs(float(cv2.contourArea(contour)))
        bases = [contour]
        hull = cv2.convexHull(contour)
        if len(hull) != len(contour):
            bases.append(hull)
        found = False
        for base in bases:
            base_perimeter = cv2.arcLength(base, True)
            for epsilon_fraction in (0.008, 0.012, 0.02, 0.03, 0.05, 0.07):
                approx = cv2.approxPolyDP(
                    base, epsilon_fraction * base_perimeter, True
                )
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quads.append(
                        (
                            order_quad(approx.reshape(4, 2)),
                            contour_area,
                        )
                    )
                    found = True
                    break
            if found:
                break
    return quads


def _all_candidate_quads(
    image_bgr: np.ndarray, edges: np.ndarray
) -> List[Tuple[np.ndarray, float]]:
    raw = _candidate_quads(edges)
    for mask in _color_region_masks(image_bgr):
        raw.extend(_candidate_quads(mask))

    # The same border normally appears in several masks.  Coarse corner
    # quantization avoids doing multiple perspective warps for that border.
    unique = {}
    for quad, contour_area in raw:
        ordered = order_quad(quad)
        key = tuple(np.round(ordered / 5.0).astype(np.int32).reshape(-1))
        previous = unique.get(key)
        if previous is None or contour_area > previous[1]:
            unique[key] = (ordered, contour_area)
    return list(unique.values())


def _preview_piece_evidence(
    portrait_bgr: np.ndarray, config: VisionConfig
) -> Tuple[float, bool, int, float]:
    """Cheaply score whether the upper half contains official-like pieces.

    This runs at roughly 1 px/mm and intentionally avoids the full-resolution
    plane fit.  It is only a prefilter; the selected hypothesis is still
    checked by segment_pieces before any pose is emitted.
    """
    lab = cv2.cvtColor(portrait_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    local_scale = w / max(config.paper_short_mm, 1e-6)
    border = max(2, int(round(config.paper_border_exclusion_mm * local_scale)))
    divider = h // 2
    exclusion = max(
        2, int(round(config.divider_exclusion_mm * local_scale))
    )
    lower_start = min(h - border - 1, divider + exclusion)
    lower = lab[lower_start : h - border, border : w - border]
    if lower.size == 0:
        return -1e6, False, 0, 0.0

    paper_color = np.median(lower.reshape(-1, 3), axis=0)
    color_delta = lab - paper_color
    distance = np.linalg.norm(color_delta, axis=2)
    chroma_distance = np.linalg.norm(color_delta[:, :, 1:3], axis=2)
    lower_distance = distance[
        lower_start : h - border, border : w - border
    ]
    lower_chroma_distance = chroma_distance[
        lower_start : h - border, border : w - border
    ]
    baseline = float(np.median(lower_distance))
    mad = float(np.median(np.abs(lower_distance - baseline)))
    threshold = max(
        config.foreground_min_lab_distance,
        baseline + config.foreground_mad_scale * max(mad, 0.35),
    )

    source_end = max(border, divider - exclusion)
    source_distance = distance[
        border:source_end, border : w - border
    ]
    # A pure-color sheet can still have a strong exposure/color gradient.
    # The clean-half MAD threshold then labels most of the source half as
    # foreground.  Otsu on the source half separates the high-contrast
    # fragments without assuming that both paper halves have equal lighting.
    threshold = max(
        threshold,
        _otsu_distance_threshold(source_distance),
    )
    chroma_threshold = _chroma_noise_threshold(
        lower_chroma_distance, config
    )
    chroma_threshold = max(
        chroma_threshold,
        _otsu_distance_threshold(
            chroma_distance[
                border:source_end, border : w - border
            ]
        ),
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    source_foreground = (
        distance[border:source_end, border : w - border] > threshold
    ) | (
        chroma_distance[border:source_end, border : w - border]
        > chroma_threshold
    )
    mask[border:source_end, border : w - border] = (
        source_foreground.astype(np.uint8) * 255
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    piece_count = 0
    piece_area_mm2 = 0.0
    invalid_area_mm2 = 0.0
    area_scale = local_scale * local_scale
    for contour in contours:
        contour_area_px = abs(float(cv2.contourArea(contour)))
        area_mm2 = contour_area_px / max(area_scale, 1e-6)
        if area_mm2 < 0.65 * config.min_piece_area_mm2:
            continue
        if area_mm2 > 1.25 * config.max_piece_area_mm2:
            invalid_area_mm2 += area_mm2
            continue
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(1.0, config.polygon_epsilon_mm * local_scale)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        while (
            len(approx) > config.max_polygon_vertices
            and epsilon < 0.08 * perimeter
        ):
            epsilon *= 1.3
            approx = cv2.approxPolyDP(contour, epsilon, True)
        approximation_area = abs(float(cv2.contourArea(approx)))
        relative_error = abs(
            approximation_area - contour_area_px
        ) / max(contour_area_px, 1.0)
        valid_polygon = (
            3 <= len(approx) <= config.max_polygon_vertices
            and relative_error <= 0.22
        )
        effective_area_px = contour_area_px

        # Match the full-resolution textured-piece repair.  This preview is
        # only a paper-candidate prefilter; without the same narrow fallback,
        # a printed mark can reduce a true four-piece scene to three before
        # segment_pieces gets a chance to inspect it accurately.
        if not valid_polygon and config.texture_enabled:
            hull = cv2.convexHull(contour)
            hull_area_px = abs(float(cv2.contourArea(hull)))
            hull_growth = (
                (hull_area_px - contour_area_px)
                / max(contour_area_px, 1.0)
            )
            hull_perimeter = cv2.arcLength(hull, True)
            hull_epsilon = max(
                1.0, config.polygon_epsilon_mm * local_scale
            )
            hull_approx = cv2.approxPolyDP(
                hull, hull_epsilon, True
            )
            while (
                len(hull_approx) > config.max_polygon_vertices
                and hull_epsilon < 0.08 * hull_perimeter
            ):
                hull_epsilon *= 1.3
                hull_approx = cv2.approxPolyDP(
                    hull, hull_epsilon, True
                )
            hull_approximation_area = abs(
                float(cv2.contourArea(hull_approx))
            )
            hull_relative_error = abs(
                hull_approximation_area - hull_area_px
            ) / max(hull_area_px, 1.0)
            valid_polygon = (
                0.0 < hull_growth <= 0.20
                and 3
                <= len(hull_approx)
                <= config.max_polygon_vertices
                and hull_relative_error <= 0.22
            )
            if valid_polygon:
                effective_area_px = hull_area_px

        if not valid_polygon:
            invalid_area_mm2 += area_mm2
            continue
        piece_count += 1
        piece_area_mm2 += effective_area_px / max(area_scale, 1e-6)

    if config.expected_piece_count:
        count_ok = piece_count == config.expected_piece_count
        count_error = abs(piece_count - config.expected_piece_count)
    else:
        count_ok = 1 <= piece_count <= 4
        count_error = 0 if count_ok else min(abs(piece_count - 1), abs(piece_count - 4))

    minimum_total_area = (
        config.target_short_min_mm
        * config.target_long_min_mm
        * 0.75
    )
    maximum_total_area = (
        config.target_short_max_mm
        * config.target_long_max_mm
        * 1.25
    )
    area_ok = minimum_total_area <= piece_area_mm2 <= maximum_total_area
    if piece_area_mm2 < minimum_total_area:
        normalized_area_error = (
            minimum_total_area - piece_area_mm2
        ) / max(minimum_total_area, 1.0)
    elif piece_area_mm2 > maximum_total_area:
        normalized_area_error = (
            piece_area_mm2 - maximum_total_area
        ) / max(maximum_total_area, 1.0)
    else:
        normalized_area_error = 0.0

    lower_foreground_ratio = float(
        np.count_nonzero(
            (lower_distance > threshold)
            | (lower_chroma_distance > chroma_threshold)
        )
    ) / max(lower_distance.size, 1)
    invalid_ratio = invalid_area_mm2 / max(
        config.paper_short_mm * 0.5 * config.paper_long_mm, 1.0
    )
    score = (
        (12.0 if count_ok else -4.0 * count_error)
        + (8.0 if area_ok else -6.0 * normalized_area_error)
        + 0.5 * min(piece_count, 4)
        - 12.0 * lower_foreground_ratio
        - 4.0 * min(invalid_ratio, 2.0)
    )
    return score, bool(count_ok and area_ok), piece_count, piece_area_mm2


def _candidate_preview_evidence(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    config: VisionConfig,
) -> _PreviewEvidence:
    hypotheses: List[_PreviewEvidence] = []
    for landscape in (False, True):
        preview = _warp_candidate_preview(
            frame_bgr, quad, config, landscape
        )
        if landscape:
            preview = cv2.rotate(preview, cv2.ROTATE_90_CLOCKWISE)
        for reverse in (False, True):
            oriented = (
                cv2.rotate(preview, cv2.ROTATE_180)
                if reverse
                else preview
            )
            score, plausible, count, area = _preview_piece_evidence(
                oriented, config
            )
            hypotheses.append(
                _PreviewEvidence(
                    score=score,
                    plausible=plausible,
                    landscape=landscape,
                    reverse=reverse,
                    piece_count=count,
                    piece_area_mm2=area,
                )
            )
    return max(hypotheses, key=lambda hypothesis: hypothesis.score)


def detect_paper_and_pieces(
    frame_bgr: np.ndarray,
    config: VisionConfig,
) -> Tuple[
    PaperObservation,
    List[PieceObservation],
    np.ndarray,
    np.ndarray,
]:
    if frame_bgr is None or frame_bgr.ndim != 3:
        raise DetectionError("camera frame is empty or not a BGR image")

    frame = undistort_frame(frame_bgr, config)
    original_h, original_w = frame.shape[:2]
    if config.fixed_paper_quad_px is not None:
        fixed_quad = order_quad(
            np.asarray(config.fixed_paper_quad_px, dtype=np.float32)
        )
        if (
            float(fixed_quad[:, 0].min()) < 0.0
            or float(fixed_quad[:, 1].min()) < 0.0
            or float(fixed_quad[:, 0].max()) > original_w - 1
            or float(fixed_quad[:, 1].max()) > original_h - 1
        ):
            raise DetectionError(
                "fixed paper calibration lies outside the camera frame"
            )
        horizontal = 0.5 * (
            np.linalg.norm(fixed_quad[1] - fixed_quad[0])
            + np.linalg.norm(fixed_quad[2] - fixed_quad[3])
        )
        vertical = 0.5 * (
            np.linalg.norm(fixed_quad[3] - fixed_quad[0])
            + np.linalg.norm(fixed_quad[2] - fixed_quad[1])
        )
        area = abs(
            float(
                cv2.contourArea(
                    fixed_quad.astype(np.float32)
                )
            )
        )
        print(
            "[vision] using fixed paper calibration "
            f"area={area:.0f} px2",
            flush=True,
        )
        fixed_candidate = _PaperCandidate(
            quad=fixed_quad,
            score=10.0,
            divider_coverage=1.0,
            landscape=bool(horizontal >= vertical),
            reverse=False,
            preview_score=0.0,
            preview_plausible=True,
            preview_piece_count=0,
            preview_piece_area_mm2=0.0,
            fixed=True,
        )
        try:
            return _rectify_paper(
                frame, fixed_candidate, config
            )
        except DetectionError as error:
            raise DetectionError(
                "fixed paper calibration did not contain valid pieces: "
                f"{error}"
            ) from error

    max_detection_side = 1100
    scale = min(1.0, max_detection_side / max(original_h, original_w))
    if scale < 1.0:
        detection_image = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    else:
        detection_image = frame

    edges = _gradient_edges(detection_image)
    frame_area = float(detection_image.shape[0] * detection_image.shape[1])
    expected_ratio = config.paper_long_mm / config.paper_short_mm
    candidates: List[_PaperCandidate] = []
    raw_quads = _all_candidate_quads(detection_image, edges)
    frame_clear_pass_count = 0
    area_pass_count = 0
    aspect_pass_count = 0

    for quad_small, contour_area in raw_quads:
        detection_h, detection_w = detection_image.shape[:2]
        frame_margin = 2.0
        touches_frame = (
            float(quad_small[:, 0].min()) < frame_margin
            or float(quad_small[:, 1].min()) < frame_margin
            or float(quad_small[:, 0].max()) > detection_w - 1 - frame_margin
            or float(quad_small[:, 1].max()) > detection_h - 1 - frame_margin
        )
        if not touches_frame:
            frame_clear_pass_count += 1
        area = abs(float(cv2.contourArea(quad_small.astype(np.float32))))
        area_ratio = area / frame_area
        if not config.paper_min_area_ratio <= area_ratio <= config.paper_max_area_ratio:
            continue
        area_pass_count += 1
        _, _, ratio = _quad_geometry(quad_small)
        ratio_error = abs(np.log(max(ratio, 1e-6) / expected_ratio))
        if ratio_error > config.paper_aspect_tolerance:
            continue
        aspect_pass_count += 1

        orientation_responses = []
        for landscape in (False, True):
            preview = _warp_candidate_preview(
                detection_image, quad_small, config, landscape
            )
            divider_coverage, _ = _line_response(
                preview, horizontal_line=not landscape, config=config
            )
            orientation_responses.append(
                (divider_coverage, landscape)
            )
        divider_coverage, landscape = max(
            orientation_responses, key=lambda item: item[0]
        )
        # Build the cheap piece-count preview from the original capture.
        # ``detection_image`` may already be downscaled to 1100 px and the
        # preview is then warped again to roughly 1 px/mm.  That double
        # resampling erased narrow playing-card strips in a real 1280x720
        # capture: the full A4 candidate appeared to contain two pieces while
        # its divider-bounded half appeared to contain four, so the half-sheet
        # was selected and every measured contour was projectively distorted.
        # Candidate geometry still comes from the bounded detection image;
        # only this small evidence warp uses the source pixels.
        preview_evidence = _candidate_preview_evidence(
            frame, quad_small / scale, config
        )
        rectangularity = min(area, contour_area) / max(
            area, contour_area, 1.0
        )
        score = (
            1.5 * area_ratio
            + 2.0 * divider_coverage
            + 0.5 * rectangularity
            - 1.25 * ratio_error
            - (1.0 if touches_frame else 0.0)
        )
        candidates.append(
            _PaperCandidate(quad=quad_small / scale, score=score,
                            divider_coverage=divider_coverage,
                            landscape=preview_evidence.landscape,
                            reverse=preview_evidence.reverse,
                            preview_score=preview_evidence.score,
                            preview_plausible=preview_evidence.plausible,
                            preview_piece_count=preview_evidence.piece_count,
                            preview_piece_area_mm2=(
                                preview_evidence.piece_area_mm2
                            ))
        )

    if not candidates:
        raise DetectionError(
            "A4 contour not found "
            f"(quads={len(raw_quads)}, frame_ok={frame_clear_pass_count}, "
            f"area_ok={area_pass_count}, "
            f"shape_ok={aspect_pass_count}); keep all four paper edges visible, "
            "make the paper occupy at least 5% of the image, and use a "
            "contrasting platform"
        )

    # Edge/color masks can form a convincing quadrilateral from a platform,
    # support arm, or image border.  Do not trust the highest visual score
    # alone: a valid paper hypothesis must also yield the official 1..4 piece
    # components on exactly one side of its geometric centre line.
    plausible_candidates = [
        candidate
        for candidate in candidates
        if candidate.preview_plausible
    ]
    if not plausible_candidates:
        best_preview = max(
            candidates, key=lambda candidate: candidate.preview_score
        )
        raise DetectionError(
            "A4-like quadrilaterals were found, but the fast check found no "
            "candidate with the configured piece count and target area "
            f"(best pieces={best_preview.preview_piece_count}, "
            f"area={best_preview.preview_piece_area_mm2:.0f} mm2). Keep all "
            "four paper corners clear and separate the paper color from the "
            "platform."
        )

    failures = []
    preview_order = sorted(
        plausible_candidates,
        key=lambda candidate: (
            candidate.preview_score,
            candidate.score,
        ),
        reverse=True,
    )
    visual_order = sorted(
        candidates, key=lambda candidate: candidate.score, reverse=True
    )
    ordered_candidates = []
    seen_quads = set()
    for candidate in preview_order[:3] + visual_order[:2]:
        signature = tuple(
            np.round(order_quad(candidate.quad) / 5.0)
            .astype(np.int32)
            .reshape(-1)
        )
        if signature in seen_quads:
            continue
        seen_quads.add(signature)
        ordered_candidates.append(candidate)

    for candidate_index, candidate in enumerate(ordered_candidates[:4]):
        try:
            return _rectify_paper(frame, candidate, config)
        except DetectionError as error:
            failures.append(
                f"candidate {candidate_index + 1}: {error}"
            )

    detail = "; ".join(failures[:2])
    raise DetectionError(
        "A4-like quadrilaterals were found, but none contained 1..4 valid "
        f"pieces on one half. {detail}"
    )


def detect_paper(
    frame_bgr: np.ndarray, config: VisionConfig
) -> PaperObservation:
    paper, _, _, _ = detect_paper_and_pieces(frame_bgr, config)
    return paper


def _rotation_cw_matrix(width: int, height: int) -> np.ndarray:
    return np.array(
        [[0.0, -1.0, height - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rotation_180_matrix(width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [-1.0, 0.0, width - 1.0],
            [0.0, -1.0, height - 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _make_rectified_observation(
    frame_bgr: np.ndarray,
    candidate: _PaperCandidate,
    config: VisionConfig,
    landscape: bool,
    reverse: bool,
) -> Tuple[PaperObservation, float]:
    quad = order_quad(candidate.quad)
    short_px = int(round(config.paper_short_mm * config.px_per_mm))
    long_px = int(round(config.paper_long_mm * config.px_per_mm))
    width, height = (long_px, short_px) if landscape else (short_px, long_px)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    frame_to_warp = cv2.getPerspectiveTransform(quad, destination).astype(np.float64)
    warped = cv2.warpPerspective(frame_bgr, frame_to_warp, (width, height))
    transform = frame_to_warp

    if landscape:
        rotate = _rotation_cw_matrix(width, height)
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        transform = rotate @ transform
        width, height = height, width

    if reverse:
        rotate = _rotation_180_matrix(width, height)
        warped = cv2.rotate(warped, cv2.ROTATE_180)
        transform = rotate @ transform

    if candidate.fixed:
        # A fixed rig already defines the physical centre line in paper
        # coordinates.  Ignoring the observed black pixels prevents artwork,
        # shadows, or a partially covered line from moving the source/target
        # boundary between runs.
        divider_coverage = 1.0
        divider_y = 0.5 * float(height - 1)
    else:
        # The hand-drawn divider need not land on the exact geometric centre.
        # Locate its full-width colour/brightness contrast after rectification
        # and retain the geometric centre only when the line is not reliable.
        divider_coverage, measured_divider_y = _line_response(
            warped, horizontal_line=True, config=config
        )
        divider_y = (
            float(measured_divider_y)
            if divider_coverage >= config.divider_min_coverage
            else 0.5 * float(height - 1)
        )

    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    exclusion = int(round(config.divider_exclusion_mm * config.px_per_mm))
    lower_start = min(height - 1, int(round(divider_y)) + exclusion)
    lower = lab[lower_start:, :]
    paper_color = np.median(lower.reshape(-1, 3), axis=0)

    observation = PaperObservation(
        image_bgr=warped,
        frame_quad_px=quad,
        frame_to_paper_px=transform,
        paper_px_to_frame=np.linalg.inv(transform),
        divider_y_px=divider_y,
        paper_color_lab=paper_color,
        detection_score=candidate.score,
        px_per_mm=config.px_per_mm,
    )
    return observation, divider_coverage


def _rectify_paper(
    frame_bgr: np.ndarray, candidate: _PaperCandidate, config: VisionConfig
) -> Tuple[
    PaperObservation,
    List[PieceObservation],
    np.ndarray,
    np.ndarray,
]:
    failures = []
    orientation_order = (
        [(candidate.landscape, candidate.reverse)]
        if candidate.fixed
        else [
            (candidate.landscape, candidate.reverse),
            (candidate.landscape, not candidate.reverse),
            (not candidate.landscape, candidate.reverse),
            (not candidate.landscape, not candidate.reverse),
        ]
    )
    for landscape, reverse in orientation_order:
        label = (
            ("landscape" if landscape else "portrait")
            + ("+180" if reverse else "")
        )
        try:
            paper, divider_coverage = _make_rectified_observation(
                frame_bgr,
                candidate,
                config,
                landscape=landscape,
                reverse=reverse,
            )
            pieces, foreground_mask, residual = segment_pieces(
                paper, config
            )
            total_piece_area_mm2 = sum(
                piece.area_mm2 for piece in pieces
            )
            minimum_total_area = (
                config.target_short_min_mm
                * config.target_long_min_mm
                * 0.75
            )
            maximum_total_area = (
                config.target_short_max_mm
                * config.target_long_max_mm
                * 1.25
            )
            if not (
                minimum_total_area
                <= total_piece_area_mm2
                <= maximum_total_area
            ):
                raise DetectionError(
                    f"piece total area {total_piece_area_mm2:.0f} mm2 is "
                    f"outside the target range "
                    f"{minimum_total_area:.0f}..{maximum_total_area:.0f}"
                )
            divider_source = (
                "fixed-calibration"
                if candidate.fixed
                else (
                    "measured"
                    if divider_coverage >= config.divider_min_coverage
                    else "center-fallback"
                )
            )
            print(
                "[vision] divider "
                f"y={paper.divider_y_px / config.px_per_mm:.1f} mm, "
                f"coverage={divider_coverage:.2f}, "
                f"source={divider_source}",
                flush=True,
            )
            return paper, pieces, foreground_mask, residual
        except DetectionError as error:
            failures.append(f"{label}: {error}")

    raise DetectionError(
        "no orientation produced valid pieces ("
        + "; ".join(failures[:2])
        + ")"
    )


def _fit_background_plane(
    lab: np.ndarray,
    divider_y: float,
    config: VisionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = lab.shape[:2]
    border = max(2, int(round(config.paper_border_exclusion_mm * config.px_per_mm)))
    start_y = min(
        h - border - 1,
        int(round(divider_y + config.divider_exclusion_mm * config.px_per_mm)),
    )
    step = max(3, int(round(config.px_per_mm * 2.0)))
    ys, xs = np.mgrid[start_y : h - border : step, border : w - border : step]
    samples = lab[ys, xs].reshape(-1, 3).astype(np.float64)
    coordinates = np.column_stack(
        [
            xs.reshape(-1) / max(w - 1, 1),
            ys.reshape(-1) / max(h - 1, 1),
            np.ones(xs.size),
        ]
    )
    median = np.median(samples, axis=0)
    distance = np.linalg.norm(samples - median, axis=1)
    keep = distance <= np.percentile(distance, 85)
    if np.count_nonzero(keep) < 100:
        keep = np.ones(len(samples), dtype=bool)

    solved, coefficients = cv2.solve(
        coordinates[keep].astype(np.float32),
        samples[keep].astype(np.float32),
        flags=cv2.DECOMP_QR,
    )
    if not solved:
        raise DetectionError("could not fit the paper illumination model")

    x_axis = np.arange(w, dtype=np.float32) / max(w - 1, 1)
    y_axis = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    background = (
        x_axis[None, :, None] * coefficients[0][None, None, :]
        + y_axis[:, None, None] * coefficients[1][None, None, :]
        + coefficients[2][None, None, :]
    )
    lower_residual = np.linalg.norm(
        lab[start_y : h - border, border : w - border].astype(np.float64)
        - background[start_y : h - border, border : w - border],
        axis=2,
    )
    return background.astype(np.float32), lower_residual.astype(np.float32)


def _fit_supported_edge_line(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    px_per_mm: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    initial = np.asarray(end, dtype=np.float64) - np.asarray(
        start, dtype=np.float64
    )
    length = float(np.linalg.norm(initial))
    if length < max(8.0, 0.8 * px_per_mm):
        return None
    unit = initial / length
    relative = np.asarray(points, dtype=np.float64) - start
    projection = relative @ unit
    line_distance = np.abs(
        initial[0] * relative[:, 1]
        - initial[1] * relative[:, 0]
    ) / length
    trim = min(0.12 * length, max(2.0, 1.2 * px_per_mm))
    selected = (
        (projection >= trim)
        & (projection <= length - trim)
        & (line_distance <= max(3.0, 2.5 * px_per_mm))
    )
    samples = np.asarray(points, dtype=np.float64)[selected]
    minimum_samples = max(8, int(round(0.25 * length)))
    if len(samples) < minimum_samples:
        return None

    direction = unit
    center = samples.mean(axis=0)
    for _ in range(2):
        centered = samples - center
        covariance = centered.T @ centered
        _, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, -1]
        if float(direction @ unit) < 0.0:
            direction = -direction
        normal = np.array([-direction[1], direction[0]])
        residual = np.abs(centered @ normal)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        threshold = min(
            1.5 * px_per_mm,
            max(1.0, median + 3.0 * 1.4826 * max(mad, 0.1)),
        )
        inliers = residual <= threshold
        if np.count_nonzero(inliers) < minimum_samples:
            return None
        samples = samples[inliers]
        center = samples.mean(axis=0)

    normal = np.array([-direction[1], direction[0]])
    rms = float(
        np.sqrt(np.mean(np.square((samples - center) @ normal)))
    )
    span = float(np.ptp((samples - center) @ direction))
    if (
        rms > max(1.0, 0.75 * px_per_mm)
        or span < 0.55 * length
    ):
        return None
    return center, direction


def _polygon_boundary_error(
    boundary: np.ndarray, polygon: np.ndarray
) -> float:
    distances = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        vector = end - start
        denominator = float(vector @ vector)
        if denominator < 1e-9:
            continue
        projection = np.clip(
            ((boundary - start) @ vector) / denominator,
            0.0,
            1.0,
        )
        closest = start + projection[:, None] * vector
        distances.append(np.linalg.norm(boundary - closest, axis=1))
    if not distances:
        return float("inf")
    return float(np.percentile(np.min(np.stack(distances), axis=0), 70))


def _refit_polygon_edges(
    contour: np.ndarray,
    polygon_px: np.ndarray,
    config: VisionConfig,
) -> np.ndarray:
    """Restore virtual vertices from dense straight-edge support.

    RDP places a vertex on a rounded or chipped physical corner.  The cut
    geometry is better represented by intersecting robustly fitted adjacent
    edge lines.  Any weak, unstable, or topology-changing fit falls back to
    the original polygon.
    """
    polygon = ensure_positive_winding(polygon_px)
    boundary = ensure_positive_winding(
        np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    )
    if len(boundary) < 3 * len(polygon):
        return polygon

    start_index = int(
        np.argmin(np.linalg.norm(boundary - polygon[0], axis=1))
    )
    boundary = np.roll(boundary, -start_index, axis=0)
    vertex_indices = [
        int(np.argmin(np.linalg.norm(boundary - vertex, axis=1)))
        for vertex in polygon
    ]
    if (
        vertex_indices[0] > 1
        or any(
            following <= current
            for current, following in zip(
                vertex_indices, vertex_indices[1:]
            )
        )
    ):
        return polygon

    lines = []
    for index, start in enumerate(polygon):
        following = (index + 1) % len(polygon)
        begin = vertex_indices[index]
        finish = (
            vertex_indices[following]
            if following
            else len(boundary)
        )
        arc = boundary[begin : finish + 1]
        if following == 0:
            arc = np.vstack([arc, boundary[:1]])
        fitted = _fit_supported_edge_line(
            arc,
            start,
            polygon[following],
            config.px_per_mm,
        )
        if fitted is None:
            return polygon
        lines.append(fitted)

    refitted = []
    for index, (current_origin, current_direction) in enumerate(lines):
        previous_origin, previous_direction = lines[index - 1]
        denominator = float(
            previous_direction[0] * current_direction[1]
            - previous_direction[1] * current_direction[0]
        )
        if abs(denominator) < 0.08:
            return polygon
        delta = current_origin - previous_origin
        distance = float(
            (
                delta[0] * current_direction[1]
                - delta[1] * current_direction[0]
            )
            / denominator
        )
        refitted.append(previous_origin + distance * previous_direction)
    fitted_vertices = np.asarray(refitted, dtype=np.float64)
    candidate = polygon.copy()
    rounded_corner_count = 0
    for index, fitted_vertex in enumerate(fitted_vertices):
        incoming = polygon[index - 1] - polygon[index]
        outgoing = polygon[(index + 1) % len(polygon)] - polygon[index]
        denominator = float(
            np.linalg.norm(incoming) * np.linalg.norm(outgoing)
        )
        if denominator < 1e-9:
            continue
        angle = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        float(incoming @ outgoing) / denominator,
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        boundary_distance = float(
            np.min(np.linalg.norm(boundary - fitted_vertex, axis=1))
        )
        displacement = float(
            np.linalg.norm(fitted_vertex - polygon[index])
        )
        # A virtual playing-card corner is approximately square and lies
        # outside the rounded physical contour.  Sharp hand-cut vertices stay
        # on the measured boundary and must not be extrapolated.
        if (
            75.0 <= angle <= 105.0
            and boundary_distance >= 0.60 * config.px_per_mm
            and displacement >= 0.40 * config.px_per_mm
        ):
            candidate[index] = fitted_vertex
            rounded_corner_count += 1
    if rounded_corner_count == 0:
        return polygon

    displacement = np.linalg.norm(candidate - polygon, axis=1)
    if float(displacement.max()) > 4.0 * config.px_per_mm:
        return polygon
    edge_lengths = np.linalg.norm(
        np.roll(candidate, -1, axis=0) - candidate, axis=1
    )
    if float(edge_lengths.min()) < 0.75 * (
        config.min_polygon_edge_mm * config.px_per_mm
    ):
        return polygon
    for index in range(len(polygon)):
        old_in = polygon[index] - polygon[index - 1]
        old_out = polygon[(index + 1) % len(polygon)] - polygon[index]
        new_in = candidate[index] - candidate[index - 1]
        new_out = candidate[(index + 1) % len(candidate)] - candidate[index]
        old_turn = float(
            old_in[0] * old_out[1] - old_in[1] * old_out[0]
        )
        new_turn = float(
            new_in[0] * new_out[1] - new_in[1] * new_out[0]
        )
        if abs(old_turn) > 1.0 and old_turn * new_turn <= 0.0:
            return polygon

    contour_area = abs(
        float(
            cv2.contourArea(
                np.asarray(contour, dtype=np.float32)
            )
        )
    )
    if (
        abs(polygon_area(candidate) - contour_area)
        / max(contour_area, 1.0)
        > 0.22
    ):
        return polygon
    if _polygon_boundary_error(
        boundary, candidate
    ) > _polygon_boundary_error(boundary, polygon) + 0.25 * config.px_per_mm:
        return polygon
    return ensure_positive_winding(candidate)


def _complete_clipped_polygon_corner(
    polygon_px: np.ndarray,
    config: VisionConfig,
) -> Optional[np.ndarray]:
    """Replace a short false bevel by the adjacent-edge intersection.

    Dense card artwork can have nearly the same colour as the red paper.  A
    threshold mask then cuts one real corner off and turns a triangle into a
    quadrilateral, or a quadrilateral into a pentagon.  Only accept the very
    specific clipped-corner signature: one edge is much shorter than both
    neighbours, their intersection is nearby, and restoring it adds little
    area while preserving a convex valid polygon.
    """
    polygon = ensure_positive_winding(polygon_px)
    count = len(polygon)
    if count not in (4, 5):
        return None

    lengths = np.linalg.norm(
        np.roll(polygon, -1, axis=0) - polygon,
        axis=1,
    )
    edge_index = int(np.argmin(lengths))
    previous_length = float(lengths[(edge_index - 1) % count])
    following_length = float(lengths[(edge_index + 1) % count])
    short_length = float(lengths[edge_index])
    if (
        short_length > 12.0 * config.px_per_mm
        or short_length
        > 0.58 * min(previous_length, following_length)
    ):
        return None

    previous = polygon[(edge_index - 1) % count]
    start = polygon[edge_index]
    end = polygon[(edge_index + 1) % count]
    following = polygon[(edge_index + 2) % count]
    previous_direction = start - previous
    following_direction = following - end
    denominator = float(
        previous_direction[0] * following_direction[1]
        - previous_direction[1] * following_direction[0]
    )
    if abs(denominator) < 0.08 * (
        np.linalg.norm(previous_direction)
        * np.linalg.norm(following_direction)
    ):
        return None
    delta = end - previous
    distance = float(
        (
            delta[0] * following_direction[1]
            - delta[1] * following_direction[0]
        )
        / denominator
    )
    intersection = previous + distance * previous_direction
    if max(
        float(np.linalg.norm(intersection - start)),
        float(np.linalg.norm(intersection - end)),
    ) > 12.0 * config.px_per_mm:
        return None

    candidate = polygon.copy()
    candidate[edge_index] = intersection
    candidate = np.delete(
        candidate, (edge_index + 1) % count, axis=0
    )
    candidate = ensure_positive_winding(candidate)
    if not cv2.isContourConvex(
        candidate.astype(np.float32).reshape(-1, 1, 2)
    ):
        return None

    original_area = polygon_area(polygon)
    candidate_area = polygon_area(candidate)
    area_ratio = candidate_area / max(original_area, 1.0)
    if not 1.005 <= area_ratio <= 1.18:
        return None
    candidate_lengths = np.linalg.norm(
        np.roll(candidate, -1, axis=0) - candidate,
        axis=1,
    )
    if float(candidate_lengths.min()) < (
        config.min_polygon_edge_mm * config.px_per_mm
    ):
        return None
    return candidate


def _piece_polygon(
    contour: np.ndarray,
    config: VisionConfig,
    maximum_area_error: float = 0.22,
) -> Tuple[np.ndarray, float]:
    epsilon_px = config.polygon_epsilon_mm * config.px_per_mm
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_px, True)
    while len(approx) > config.max_polygon_vertices and epsilon_px < 0.08 * perimeter:
        epsilon_px *= 1.25
        approx = cv2.approxPolyDP(contour, epsilon_px, True)
    if not 3 <= len(approx) <= config.max_polygon_vertices:
        raise DetectionError(
            f"piece contour has {len(approx)} stable vertices, expected 3.."
            f"{config.max_polygon_vertices}"
        )
    polygon_px = ensure_positive_winding(approx.reshape(-1, 2))
    contour_area = abs(float(cv2.contourArea(contour)))
    minimum_edge_px = config.min_polygon_edge_mm * config.px_per_mm
    while len(polygon_px) > 3:
        lengths = np.linalg.norm(
            np.roll(polygon_px, -1, axis=0) - polygon_px,
            axis=1,
        )
        edge_index = int(np.argmin(lengths))
        if lengths[edge_index] >= minimum_edge_px:
            break
        remove_options = [edge_index, (edge_index + 1) % len(polygon_px)]
        candidates = [
            np.delete(polygon_px, remove_index, axis=0)
            for remove_index in remove_options
        ]
        polygon_px = min(
            candidates,
            key=lambda candidate: abs(
                polygon_area(candidate) - contour_area
            ),
        )
        polygon_px = ensure_positive_winding(polygon_px)

    # Rough hand-cut edges and perspective sampling can leave a point a
    # millimetre or two away from an otherwise straight seam.  Such a point
    # is not a real polygon corner, but keeping it splits a long T-junction
    # mate into two unrelated solver edges (for example 98 mm into 70+28 mm).
    # Remove only shallow turns that also lie close to the neighbour chord;
    # the distance gate preserves genuine narrow polygon corners.
    maximum_turn_deg = 25.0
    maximum_chord_distance_px = max(
        1.5 * config.px_per_mm,
        3.0 * epsilon_px,
    )
    while len(polygon_px) > 3:
        collinear_candidates = []
        count = len(polygon_px)
        for index in range(count):
            previous = polygon_px[(index - 1) % count]
            current = polygon_px[index]
            following = polygon_px[(index + 1) % count]
            incoming = current - previous
            outgoing = following - current
            chord = following - previous
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            chord_length = float(np.linalg.norm(chord))
            if min(incoming_length, outgoing_length, chord_length) < 1e-6:
                continue
            cosine = float(
                np.dot(incoming, outgoing)
                / (incoming_length * outgoing_length)
            )
            turn_deg = float(
                np.degrees(
                    np.arccos(np.clip(cosine, -1.0, 1.0))
                )
            )
            chord_distance = abs(
                float(
                    chord[0] * (current[1] - previous[1])
                    - chord[1] * (current[0] - previous[0])
                )
            ) / chord_length
            projection = float(
                np.dot(current - previous, chord)
                / (chord_length * chord_length)
            )
            if (
                turn_deg <= maximum_turn_deg
                and chord_distance <= maximum_chord_distance_px
                and 0.0 <= projection <= 1.0
            ):
                collinear_candidates.append(
                    (turn_deg, chord_distance, index)
                )
        if not collinear_candidates:
            break
        _, _, remove_index = min(collinear_candidates)
        polygon_px = ensure_positive_winding(
            np.delete(polygon_px, remove_index, axis=0)
        )

    approximation_area = polygon_area(polygon_px)
    relative_area_error = abs(approximation_area - contour_area) / max(contour_area, 1.0)
    if relative_area_error > maximum_area_error:
        raise DetectionError(
            f"polygon approximation changes piece area by "
            f"{relative_area_error * 100:.1f}%"
        )
    return polygon_px, epsilon_px / config.px_per_mm


def _textured_piece_contour(
    contour: np.ndarray, config: VisionConfig
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Repair a small print-colour notch without changing normal polygons.

    A red/black mark on a white playing-card fragment can be classified as
    paper when the A4 sheet has a similar colour.  If that mark is connected
    to an under-segmented edge, the external contour gains several false
    vertices.  Normal 3..5-sided fragments are always tried unchanged first.
    A rejected contour, or one that needs more than four times the configured
    approximation error, may use its convex envelope in texture mode.  The
    envelope must add at most 20% area and produce a tighter polygon fit.  This
    keeps the fallback local and prevents broad shadows or genuine polygon
    corners from being silently converted into pieces.
    """
    original_polygon = None
    original_approximation_error_mm = float("inf")
    original_failure = None
    try:
        (
            original_polygon,
            original_approximation_error_mm,
        ) = _piece_polygon(contour, config)
        if (
            not config.texture_enabled
            or original_approximation_error_mm
            <= 4.0 * config.polygon_epsilon_mm
        ):
            return (
                contour,
                original_polygon,
                original_approximation_error_mm,
            )
    except DetectionError as error:
        original_failure = error
        if not config.texture_enabled:
            raise

    contour_area = abs(float(cv2.contourArea(contour)))
    hull = cv2.convexHull(contour)
    hull_area = abs(float(cv2.contourArea(hull)))
    if (
        contour_area > 0.0
        and hull_area > contour_area
        and (hull_area - contour_area) / contour_area <= 0.20
    ):
        try:
            small_textured_fragment = (
                contour_area
                <= 400.0 * config.px_per_mm * config.px_per_mm
            )
            polygon_px, approximation_error_mm = _piece_polygon(
                hull,
                config,
                maximum_area_error=(
                    0.35 if small_textured_fragment else 0.22
                ),
            )
            if (
                original_polygon is None
                or approximation_error_mm
                < original_approximation_error_mm
            ):
                return hull, polygon_px, approximation_error_mm
        except DetectionError:
            pass

    if original_polygon is not None:
        return (
            contour,
            original_polygon,
            original_approximation_error_mm,
        )
    assert original_failure is not None
    raise original_failure


def _divider_top_edge(
    foreground_mask: np.ndarray,
    divider_y: float,
    config: VisionConfig,
) -> int:
    """Locate the real top edge of the drawn divider.

    ``divider_exclusion_mm`` is an analysis/search margin, not a placement
    rule.  A piece may legally sit inside that margin as long as it does not
    touch the actual line.  The divider is the only foreground feature that
    spans most of the A4 width, so use row coverage near the geometric centre
    to estimate its physical edge.  If its colour is too close to the paper
    for this mask, the geometric centre remains the conservative boundary.
    """
    height, width = foreground_mask.shape
    center = int(round(divider_y))
    radius = max(
        2, int(round(config.divider_exclusion_mm * config.px_per_mm))
    )
    y0 = max(0, center - radius)
    y1 = min(height, center + radius + 1)
    sample_border = max(
        1,
        int(round(config.paper_border_exclusion_mm * config.px_per_mm)),
    )
    x0 = min(sample_border, max(width - 1, 0))
    x1 = max(x0 + 1, width - sample_border)
    coverage = np.count_nonzero(
        foreground_mask[y0:y1, x0:x1], axis=1
    ) / max(x1 - x0, 1)

    # A fragment can occupy over half of a row, but not about two thirds of
    # the full A4 width.  Requiring broad support prevents a near-divider
    # fragment from being mistaken for the divider itself.
    line_rows = np.flatnonzero(coverage >= 0.65)
    if line_rows.size == 0:
        return max(1, min(height - 1, center))

    absolute_rows = line_rows + y0
    groups = np.split(
        absolute_rows,
        np.flatnonzero(np.diff(absolute_rows) > 1) + 1,
    )
    line_group = min(
        groups,
        key=lambda group: min(
            abs(int(group[0]) - center),
            abs(int(group[-1]) - center),
        ),
    )
    return max(1, min(height - 1, int(line_group[0])))


def _clean_piece_mask(mask: np.ndarray, config: VisionConfig) -> np.ndarray:
    # Remove sub-millimetre bridges between legally non-overlapping pieces.
    # At the default 3 px/mm this is a 5x5 opening, still far smaller than the
    # official 20 mm minimum edge while separating point-touching card stock.
    open_size = max(3, int(round(1.2 * config.px_per_mm)))
    if open_size % 2 == 0:
        open_size += 1
    close_size = max(3, int(round(1.1 * config.px_per_mm)))
    if close_size % 2 == 0:
        close_size += 1
    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_size, open_size)
        ),
    )
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        ),
    )
    if open_size <= 3:
        return cleaned

    # A dark rank glyph can divide a very small card triangle into strokes
    # that a 5x5 opening removes completely.  Recover only isolated, verified
    # small polygons from a 3x3 pass; large components still use the stronger
    # bridge-removing opening above.
    fine_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )
    fine = cv2.morphologyEx(mask, cv2.MORPH_OPEN, fine_kernel)
    fine = cv2.morphologyEx(fine, cv2.MORPH_CLOSE, fine_kernel)
    area_scale = config.px_per_mm * config.px_per_mm
    rescue_area_max = min(
        config.max_piece_area_mm2,
        6.0 * config.min_piece_area_mm2,
    )
    for contour in cv2.findContours(
        fine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )[0]:
        area_mm2 = abs(float(cv2.contourArea(contour))) / area_scale
        if not config.min_piece_area_mm2 <= area_mm2 <= rescue_area_max:
            continue
        component = np.zeros_like(mask)
        cv2.drawContours(component, [contour], -1, 255, thickness=-1)
        try:
            _textured_piece_contour(contour, config)
        except DetectionError:
            continue
        cleaned = cv2.bitwise_or(cleaned, component)
    return cleaned


def _cyclic_polygon_rms(
    first: np.ndarray, second: np.ndarray
) -> float:
    if len(first) != len(second):
        return float("inf")
    return min(
        float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(np.roll(first, shift, axis=0) - second),
                        axis=1,
                    )
                )
            )
        )
        for shift in range(len(first))
    )


def _refine_print_clipped_components(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    paper: PaperObservation,
    config: VisionConfig,
) -> Tuple[np.ndarray, int]:
    """Recover only components with a verified print-clipped corner.

    The current mask remains hard foreground.  A fitted red-paper background
    supplies probable foreground seeds, while GrabCut's edge term decides the
    narrow unknown ring.  The result is accepted only if it agrees with the
    independent adjacent-line completion and grows the component by at most
    20 percent.
    """
    if not config.texture_enabled:
        return mask, 0

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    suspects = []
    area_scale = config.px_per_mm * config.px_per_mm
    for contour in contours:
        area_mm2 = abs(float(cv2.contourArea(contour))) / area_scale
        if not (
            config.min_piece_area_mm2
            <= area_mm2
            <= config.max_piece_area_mm2
        ):
            continue
        try:
            _, polygon_px, _ = _textured_piece_contour(
                contour, config
            )
        except DetectionError:
            continue
        completion = _complete_clipped_polygon_corner(
            polygon_px, config
        )
        if completion is not None:
            suspects.append((contour, completion))
    if not suspects:
        return mask, 0

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    background, lower_residual = _fit_background_plane(
        lab, paper.divider_y_px, config
    )
    residual = np.linalg.norm(lab - background, axis=2)
    baseline = float(np.median(lower_residual))
    mad = float(np.median(np.abs(lower_residual - baseline)))
    border = max(
        1,
        int(round(config.paper_border_exclusion_mm * config.px_per_mm)),
    )
    source_end = max(
        border + 1,
        int(round(paper.divider_y_px))
        - int(round(config.divider_exclusion_mm * config.px_per_mm)),
    )
    source_residual = residual[
        border:source_end,
        border : residual.shape[1] - border,
    ]
    probable_threshold = max(
        config.foreground_min_lab_distance,
        baseline
        + config.foreground_mad_scale * max(mad, 0.35),
        0.85 * _otsu_distance_threshold(source_residual),
    )

    refined = mask.copy()
    recovered_count = 0
    pad = max(4, int(round(12.0 * config.px_per_mm)))
    grow_radius = max(2, int(round(6.0 * config.px_per_mm)))
    grow_size = 2 * grow_radius + 1
    grow_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (grow_size, grow_size)
    )
    for contour, completion in suspects:
        component = np.zeros_like(mask)
        cv2.drawContours(component, [contour], -1, 255, thickness=-1)
        ys, xs = np.nonzero(component)
        if xs.size == 0:
            continue
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
        seed = component[y0:y1, x0:x1]
        roi = image_bgr[y0:y1, x0:x1]
        roi_residual = residual[y0:y1, x0:x1]

        grab_mask = np.full(
            seed.shape, cv2.GC_PR_BGD, dtype=np.uint8
        )
        nearby = cv2.dilate(seed, grow_kernel)
        grab_mask[
            (nearby > 0) & (roi_residual > probable_threshold)
        ] = cv2.GC_PR_FGD
        grab_mask[seed > 0] = cv2.GC_FGD
        other = cv2.bitwise_and(
            mask[y0:y1, x0:x1], cv2.bitwise_not(seed)
        )
        grab_mask[other > 0] = cv2.GC_BGD
        grab_mask[:2, :] = cv2.GC_BGD
        grab_mask[-2:, :] = cv2.GC_BGD
        grab_mask[:, :2] = cv2.GC_BGD
        grab_mask[:, -2:] = cv2.GC_BGD

        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(
                roi,
                grab_mask,
                None,
                background_model,
                foreground_model,
                2,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            continue
        candidate = np.where(
            (grab_mask == cv2.GC_FGD)
            | (grab_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
        component_count, labels, _, _ = cv2.connectedComponentsWithStats(
            candidate
        )
        if component_count <= 1:
            continue
        label = max(
            range(1, component_count),
            key=lambda value: int(
                np.count_nonzero(
                    (labels == value) & (seed > 0)
                )
            ),
        )
        candidate = np.where(labels == label, 255, 0).astype(np.uint8)
        candidate_contours, _ = cv2.findContours(
            candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not candidate_contours:
            continue
        candidate_contour = max(
            candidate_contours, key=cv2.contourArea
        )
        original_area = abs(float(cv2.contourArea(contour)))
        candidate_area = abs(float(cv2.contourArea(candidate_contour)))
        if not 1.005 <= candidate_area / max(original_area, 1.0) <= 1.20:
            continue
        try:
            _, candidate_polygon, _ = _textured_piece_contour(
                candidate_contour, config
            )
        except DetectionError:
            continue
        candidate_polygon = candidate_polygon + np.array([x0, y0])
        if (
            len(candidate_polygon) != len(completion)
            or _cyclic_polygon_rms(
                candidate_polygon, completion
            )
            > 3.5 * config.px_per_mm
        ):
            continue
        refined[y0:y1, x0:x1] = cv2.bitwise_or(
            refined[y0:y1, x0:x1], candidate
        )
        recovered_count += 1
    return refined, recovered_count


def _fast_white_piece_mask(
    lab: np.ndarray,
    paper: PaperObservation,
    config: VisionConfig,
) -> Tuple[np.ndarray, np.ndarray, bool] | None:
    """Use a cheap white-on-colour mask only when geometry confirms it.

    This is the useful part of the senior Maix script's fixed Lab threshold,
    but made relative to the clean lower paper half.  A saturated paper is
    required, then the resulting components must independently satisfy the
    official count, area, and 3..5-sided polygon rules.  Otherwise the caller
    falls back to the illumination-plane method.
    """
    height, width = lab.shape[:2]
    border = max(
        2,
        int(round(config.paper_border_exclusion_mm * config.px_per_mm)),
    )
    lower_start = min(
        height - border - 1,
        int(
            round(
                paper.divider_y_px
                + config.divider_exclusion_mm * config.px_per_mm
            )
        ),
    )
    lower = lab[
        lower_start : height - border,
        border : width - border,
    ]
    if lower.size == 0:
        return None
    paper_lab = np.median(lower.reshape(-1, 3), axis=0)
    paper_chroma = float(np.linalg.norm(paper_lab[1:3] - 128.0))
    if paper_chroma < 24.0:
        return None

    lightness_min = max(145.0, float(paper_lab[0]) + 18.0)
    mask = cv2.inRange(
        lab,
        np.array([lightness_min, 96.0, 96.0], dtype=np.float32),
        np.array([255.0, 160.0, 160.0], dtype=np.float32),
    )
    mask[max(1, int(round(paper.divider_y_px))) :, :] = 0
    mask[:, :1] = 0
    mask[:, width - 1 :] = 0
    cleaned = _clean_piece_mask(mask, config)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    valid_areas = []
    repaired_dimensions = []
    repaired_boxes = []
    high_error_count = 0
    rejected_piece_sized_count = 0
    area_scale = config.px_per_mm * config.px_per_mm
    for contour in contours:
        area_mm2 = abs(float(cv2.contourArea(contour))) / area_scale
        if not (
            config.min_piece_area_mm2
            <= area_mm2
            <= config.max_piece_area_mm2
        ):
            continue
        original_error = float("inf")
        try:
            _, original_error = _piece_polygon(contour, config)
        except DetectionError:
            pass
        try:
            repaired_contour, _, _ = _textured_piece_contour(
                contour, config
            )
        except DetectionError:
            rejected_piece_sized_count += 1
            continue
        valid_areas.append(area_mm2)
        repaired_rectangle = cv2.minAreaRect(repaired_contour)
        rect_width, rect_height = repaired_rectangle[1]
        short, long = sorted(
            (
                float(rect_width) / config.px_per_mm,
                float(rect_height) / config.px_per_mm,
            )
        )
        repaired_dimensions.append((short, long))
        repaired_boxes.append(
            cv2.boxPoints(repaired_rectangle).astype(np.float64)
        )
        if original_error > 4.0 * config.polygon_epsilon_mm:
            high_error_count += 1

    count = len(valid_areas)
    if rejected_piece_sized_count:
        return None
    if not 1 <= count <= 4:
        return None
    if config.expected_piece_count and count != config.expected_piece_count:
        return None
    total_area = sum(valid_areas)
    minimum_total_area = (
        0.75 * config.target_short_min_mm * config.target_long_min_mm
    )
    maximum_total_area = (
        1.25 * config.target_short_max_mm * config.target_long_max_mm
    )
    if not minimum_total_area <= total_area <= maximum_total_area:
        return None
    # A fast white-only mask deliberately excludes dark printing.  It remains
    # reliable when at least one fragment keeps a stable raw outline, but if
    # every component needs a coarse polygon fit, artwork is likely cutting
    # notches into all measured seams.  Let the illumination-plane path retain
    # those dark pixels instead of guessing every edge from convex envelopes.
    all_notched_card_strips = False
    if count == 4 and len(repaired_dimensions) == 4:
        dimensions = np.asarray(repaired_dimensions, dtype=np.float64)
        common_width = float(np.median(dimensions[:, 1]))
        assembled_aspect = float(
            np.sum(dimensions[:, 0]) / max(common_width, 1e-6)
        )
        all_notched_card_strips = bool(
            float(np.min(dimensions[:, 0])) > 1e-6
            and float(np.min(dimensions[:, 1])) > 1e-6
            and float(np.max(dimensions[:, 1]))
            / float(np.min(dimensions[:, 1]))
            <= 1.15
            and 1.25 <= assembled_aspect <= 2.20
        )
    reconstruct_notched_card_strips = bool(
        high_error_count == count and all_notched_card_strips
    )
    if high_error_count == count and not reconstruct_notched_card_strips:
        return None
    if reconstruct_notched_card_strips:
        mask = np.zeros_like(mask)
        for box in repaired_boxes:
            cv2.fillConvexPoly(
                mask,
                np.round(box).astype(np.int32),
                255,
            )

    residual = np.abs(lab[:, :, 0] - float(paper_lab[0]))
    return (
        mask,
        residual.astype(np.float32),
        reconstruct_notched_card_strips,
    )


def segment_pieces(
    paper: PaperObservation, config: VisionConfig
) -> Tuple[List[PieceObservation], np.ndarray, np.ndarray]:
    image = paper.image_bgr
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    analysis_border = max(
        1,
        int(round(config.paper_border_exclusion_mm * config.px_per_mm)),
    )
    divider_exclusion = int(
        round(config.divider_exclusion_mm * config.px_per_mm)
    )
    threshold_source_end = max(
        analysis_border,
        int(round(paper.divider_y_px)) - divider_exclusion,
    )
    fast_white = _fast_white_piece_mask(lab, paper, config)
    boundary_reconstructed = False
    if fast_white is not None:
        mask, residual, boundary_reconstructed = fast_white
    else:
        background, lower_residual = _fit_background_plane(
            lab, paper.divider_y_px, config
        )
        color_delta = lab - background
        residual = np.linalg.norm(color_delta, axis=2)
        chroma_residual = np.linalg.norm(
            color_delta[:, :, 1:3], axis=2
        )
        baseline = float(np.median(lower_residual))
        mad = float(np.median(np.abs(lower_residual - baseline)))
        threshold = max(
            config.foreground_min_lab_distance,
            baseline + config.foreground_mad_scale * max(mad, 0.35),
        )
        source_residual = residual[
            analysis_border:threshold_source_end,
            analysis_border : w - analysis_border,
        ]
        threshold = max(
            threshold,
            _otsu_distance_threshold(source_residual),
        )
        lower_start = min(
            h - analysis_border - 1,
            int(
                round(
                    paper.divider_y_px
                    + config.divider_exclusion_mm * config.px_per_mm
                )
            ),
        )
        lower_chroma_residual = chroma_residual[
            lower_start : h - analysis_border,
            analysis_border : w - analysis_border,
        ]
        source_chroma_residual = chroma_residual[
            analysis_border:threshold_source_end,
            analysis_border : w - analysis_border,
        ]
        chroma_threshold = _chroma_noise_threshold(
            lower_chroma_residual, config
        )
        plane_foreground_ratio = float(
            np.count_nonzero(
                source_chroma_residual > chroma_threshold
            )
        ) / max(source_chroma_residual.size, 1)
        if plane_foreground_ratio > 0.45:
            lower_chroma = lab[
                lower_start : h - analysis_border,
                analysis_border : w - analysis_border,
                1:3,
            ]
            paper_chroma = np.median(
                lower_chroma.reshape(-1, 2), axis=0
            )
            chroma_residual = np.linalg.norm(
                lab[:, :, 1:3] - paper_chroma, axis=2
            )
            lower_chroma_residual = chroma_residual[
                lower_start : h - analysis_border,
                analysis_border : w - analysis_border,
            ]
            source_chroma_residual = chroma_residual[
                analysis_border:threshold_source_end,
                analysis_border : w - analysis_border,
            ]
            chroma_threshold = max(
                _chroma_noise_threshold(
                    lower_chroma_residual, config
                ),
                _otsu_distance_threshold(source_chroma_residual),
            )
        fixed_saturated_paper = bool(
            config.fixed_paper_quad_px is not None
            and np.linalg.norm(paper.paper_color_lab[1:3] - 128.0)
            >= 24.0
        )
        mask = (
            (chroma_residual > chroma_threshold)
            if fixed_saturated_paper
            else (
                (residual > threshold)
                | (chroma_residual > chroma_threshold)
            )
        ).astype(np.uint8) * 255
    source_end = _divider_top_edge(mask, paper.divider_y_px, config)

    # Only the real paper boundary and the physical divider clip a contour.
    # The wider configured margins above are retained for robust background
    # and threshold estimation, but are no longer imposed on legal placement.
    physical_border = 1
    valid_region = np.zeros_like(mask)
    valid_region[
        physical_border:source_end,
        physical_border : w - physical_border,
    ] = 255
    mask = cv2.bitwise_and(mask, valid_region)

    mask = _clean_piece_mask(mask, config)
    refinement_seed = np.zeros_like(mask)
    px_area_scale = config.px_per_mm * config.px_per_mm
    for contour in cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )[0]:
        area_mm2 = abs(float(cv2.contourArea(contour))) / px_area_scale
        if not (
            config.min_piece_area_mm2
            <= area_mm2
            <= config.max_piece_area_mm2
        ):
            continue
        try:
            piece_contour, _, _ = _textured_piece_contour(
                contour, config
            )
        except DetectionError:
            continue
        cv2.drawContours(
            refinement_seed,
            [piece_contour],
            -1,
            255,
            thickness=-1,
        )
    refined_seed, recovered_print_corners = (
        _refine_print_clipped_components(
            image, refinement_seed, paper, config
        )
    )
    if recovered_print_corners:
        boundary_reconstructed = True
        mask = _clean_piece_mask(refined_seed, config)
        print(
            "[vision] recovered "
            f"{recovered_print_corners} print-covered corner(s) with "
            "background/edge refinement",
            flush=True,
        )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filtered: List[Tuple[np.ndarray, np.ndarray, float]] = []
    rejected_non_polygon_count = 0
    rejected_border_sliver_count = 0
    for contour in contours:
        area_mm2 = abs(float(cv2.contourArea(contour))) / px_area_scale
        if config.min_piece_area_mm2 <= area_mm2 <= config.max_piece_area_mm2:
            try:
                (
                    piece_contour,
                    polygon_px,
                    approximation_error_mm,
                ) = _textured_piece_contour(
                    contour, config
                )
            except DetectionError:
                # Broad illumination patches and soft shadows can have a
                # piece-sized connected area, but the official fragments have
                # 3..5 straight sides.  Ignore only components that fail that
                # explicit shape rule; valid polygonal components remain
                # subject to all boundary/count safety checks below.
                rejected_non_polygon_count += 1
                continue
            area_mm2 = (
                abs(float(cv2.contourArea(piece_contour)))
                / px_area_scale
            )
            if not (
                config.min_piece_area_mm2
                <= area_mm2
                <= config.max_piece_area_mm2
            ):
                rejected_non_polygon_count += 1
                continue
            points = piece_contour.reshape(-1, 2)
            touches_outer_border = bool(
                np.min(points[:, 0]) <= physical_border
                or np.max(points[:, 0]) >= w - 1 - physical_border
                or np.min(points[:, 1]) <= physical_border
            )
            if touches_outer_border:
                rect_size = cv2.minAreaRect(
                    piece_contour.astype(np.float32)
                )[1]
                visible_thickness_mm = (
                    min(float(rect_size[0]), float(rect_size[1]))
                    / config.px_per_mm
                )
                rect_area_mm2 = (
                    float(rect_size[0])
                    * float(rect_size[1])
                    / px_area_scale
                )
                rect_occupancy = area_mm2 / max(rect_area_mm2, 1e-6)
                if (
                    visible_thickness_mm < config.min_polygon_edge_mm
                    or rect_occupancy < 0.25
                ):
                    rejected_border_sliver_count += 1
                    continue
            # The rules guarantee that each fragment remains wholly in the
            # source half, but a judge may place an edge flush with the paper
            # edge or the drawn divider.  In that case the crop boundary is
            # itself the missing straight contour segment.  Keep the valid
            # 3..5-sided polygon and let the fixed-template/global rectangle
            # checks reject inconsistent geometry instead of aborting the
            # entire run before solving.
            filtered.append(
                (
                    piece_contour,
                    polygon_px,
                    approximation_error_mm,
                )
            )

    if rejected_border_sliver_count:
        print(
            "[vision] ignored "
            f"{rejected_border_sliver_count} thin component(s) clipped by "
            "the fixed paper border",
            flush=True,
        )

    if len(filtered) > 4:
        interior = []
        for item in filtered:
            points = item[0].reshape(-1, 2)
            touches_outer_border = bool(
                np.min(points[:, 0]) <= physical_border
                or np.max(points[:, 0]) >= w - 1 - physical_border
                or np.min(points[:, 1]) <= physical_border
            )
            if not touches_outer_border:
                interior.append(item)
        if 1 <= len(interior) <= 4:
            filtered = interior

    if not 1 <= len(filtered) <= 4:
        raise DetectionError(
            f"detected {len(filtered)} valid piece components; expected 1..4. "
            f"Ignored {rejected_non_polygon_count} non-polygon illumination "
            "components. Pieces that touch each other must be separated "
            "before the run."
        )
    if config.expected_piece_count and len(filtered) != config.expected_piece_count:
        raise DetectionError(
            f"detected {len(filtered)} pieces, configured for "
            f"{config.expected_piece_count}"
        )

    observations: List[PieceObservation] = []
    piece_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
    for contour, polygon_px, approximation_error_mm in filtered:
        component = np.zeros_like(mask)
        cv2.drawContours(component, [contour], -1, 255, thickness=-1)
        moments = cv2.moments(component, binaryImage=True)
        if moments["m00"] <= 0:
            continue
        center_px = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float64,
        )
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        _, max_distance, _, max_location = cv2.minMaxLoc(distance)
        if max_distance < max(1.0, config.px_per_mm):
            raise DetectionError("a piece is too narrow for a stable pickup point")
        pickup_px = np.array(max_location, dtype=np.float64)
        area_mm2 = abs(float(cv2.contourArea(contour))) / px_area_scale
        piece_data.append(
            (
                contour,
                polygon_px,
                component,
                area_mm2,
                approximation_error_mm,
            )
        )

    piece_data.sort(
        key=lambda item: (
            float(cv2.moments(item[2], binaryImage=True)["m01"])
            / max(float(cv2.moments(item[2], binaryImage=True)["m00"]), 1.0),
            float(cv2.moments(item[2], binaryImage=True)["m10"])
            / max(float(cv2.moments(item[2], binaryImage=True)["m00"]), 1.0),
        )
    )

    filled_mask = np.zeros_like(mask)
    for piece_id, (
        contour,
        polygon_px,
        component,
        area_mm2,
        approximation_error_mm,
    ) in enumerate(piece_data):
        moments = cv2.moments(component, binaryImage=True)
        center_px = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float64,
        )
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        _, _, _, max_location = cv2.minMaxLoc(distance)
        pickup_px = np.array(max_location, dtype=np.float64)
        filled_mask = cv2.bitwise_or(filled_mask, component)
        observations.append(
            PieceObservation(
                piece_id=piece_id,
                contour_px=contour.reshape(-1, 2).astype(np.float64),
                polygon_mm=ensure_positive_winding(
                    polygon_px / config.px_per_mm
                ),
                mask=component,
                source_center_mm=center_px / config.px_per_mm,
                pickup_mm=pickup_px / config.px_per_mm,
                area_mm2=area_mm2,
                approximation_error_mm=approximation_error_mm,
                boundary_reconstructed=boundary_reconstructed,
            )
        )

    return observations, filled_mask, residual
