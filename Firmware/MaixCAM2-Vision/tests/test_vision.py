from __future__ import annotations

import json
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import puzzle_vision.template_solver as template_solver  # noqa: E402
import puzzle_vision.solver as solver_module  # noqa: E402
import tools.uart_smoke_test as uart_smoke_test  # noqa: E402
from main import (  # noqa: E402
    _TouchButtonReader,
    _button_rectangles,
    _compose_touch_ui,
    _display_bgr,
    _fit_landscape,
    _make_error_view,
    _make_uart,
    _make_uart_payload,
    _send_uart,
)
from puzzle_vision.config import VisionConfig  # noqa: E402
from puzzle_vision.detector import (  # noqa: E402
    _complete_clipped_polygon_corner,
    _fast_white_piece_mask,
    _line_response,
    _piece_polygon,
    _refit_polygon_edges,
    _textured_piece_contour,
    detect_paper,
    detect_paper_and_pieces,
    segment_pieces,
)
from puzzle_vision.geometry import (  # noqa: E402
    AssemblyMetrics,
    AssemblySolution,
    DetectionError,
    MoveCommand,
    PaperObservation,
    PieceObservation,
    SolveError,
    apply_homography,
    ensure_positive_winding,
    order_quad,
    polygon_area,
    rigid_matrix,
)
from puzzle_vision.template_solver import (  # noqa: E402
    _template_metrics_are_valid,
)
from puzzle_vision.pipeline import (  # noqa: E402
    PuzzleVisionPipeline,
    _build_commands,
    _has_printed_texture,
    _recovery_piece_hypothesis,
    _robot_rotation,
    _solve_with_recovery,
)
from puzzle_vision.solver import (  # noqa: E402
    _candidate_transforms,
    _is_card_strip_partition,
    _pair_cycle_support,
    _pair_groups_incrementally_valid,
    _polygon_overlap_area,
    _pose_key,
    _transform_rigid,
    solve_puzzle,
)
from tests.synthetic_scene import (  # noqa: E402
    add_paper_edge_artifacts,
    make_ambiguous_textured_pair,
    make_quartered_textured_card,
    make_rectified_scene,
    make_single_piece_orientation_scene,
    perspective_camera_frame,
)


class PaperDetectionRegressionTests(unittest.TestCase):
    def setUp(self):
        self.config = VisionConfig(expected_piece_count=4)

    def assertPaperQuad(
        self,
        frame: np.ndarray,
        expected_quad: np.ndarray,
        max_corner_error_px: float = 12.0,
    ):
        paper = detect_paper(frame, self.config)
        actual = order_quad(paper.frame_quad_px)
        expected = order_quad(expected_quad)
        corner_errors = np.linalg.norm(actual - expected, axis=1)
        self.assertLess(
            float(corner_errors.max()),
            max_corner_error_px,
            msg=f"paper corner errors: {corner_errors.tolist()}",
        )
        self.assertAlmostEqual(paper.width_mm, 210.0, delta=1.0)
        self.assertAlmostEqual(paper.height_mm, 297.0, delta=1.0)
        self.assertAlmostEqual(
            paper.divider_y_px,
            0.5 * paper.image_bgr.shape[0],
            delta=12.0,
        )
        return paper

    def test_non_white_a4_is_detected(self):
        paper_bgr = (72, 148, 96)
        rectified, _ = make_rectified_scene(
            self.config, paper_bgr=paper_bgr
        )
        quad = np.array(
            [[445, 18], [830, 32], [902, 692], [370, 677]],
            dtype=np.float32,
        )
        paper = self.assertPaperQuad(
            perspective_camera_frame(rectified, destination=quad),
            quad,
        )
        expected_lab = cv2.cvtColor(
            np.array([[paper_bgr]], dtype=np.uint8), cv2.COLOR_BGR2LAB
        )[0, 0].astype(np.float32)
        self.assertLess(
            float(np.linalg.norm(paper.paper_color_lab - expected_lab)),
            8.0,
        )

    def test_weak_paper_to_platform_contrast(self):
        paper_bgr = (106, 116, 126)
        background_bgr = (99, 109, 119)
        rectified, _ = make_rectified_scene(
            self.config, paper_bgr=paper_bgr
        )
        quad = np.array(
            [[410, 42], [812, 52], [871, 686], [344, 674]],
            dtype=np.float32,
        )
        frame = perspective_camera_frame(
            rectified,
            destination=quad,
            background_bgr=background_bgr,
        )
        self.assertPaperQuad(frame, quad)

    def test_strong_perspective_a4_is_detected(self):
        rectified, _ = make_rectified_scene(self.config)
        quad = np.array(
            [[392, 24], [792, 116], [1018, 676], [254, 626]],
            dtype=np.float32,
        )
        self.assertPaperQuad(
            perspective_camera_frame(rectified, destination=quad),
            quad,
            max_corner_error_px=14.0,
        )

    def test_divider_selects_orientation_when_portrait_looks_wide(self):
        rectified, _ = make_rectified_scene(self.config)
        quad = np.array(
            [[300, 160], [970, 105], [910, 590], [370, 620]],
            dtype=np.float32,
        )
        horizontal, vertical = (
            0.5
            * (
                np.linalg.norm(quad[1] - quad[0])
                + np.linalg.norm(quad[2] - quad[3])
            ),
            0.5
            * (
                np.linalg.norm(quad[3] - quad[0])
                + np.linalg.norm(quad[2] - quad[1])
            ),
        )
        self.assertGreater(horizontal, vertical)
        self.assertPaperQuad(
            perspective_camera_frame(rectified, destination=quad),
            quad,
            max_corner_error_px=14.0,
        )

    def test_complete_paper_near_frame_edges(self):
        rectified, _ = make_rectified_scene(self.config)
        quad = np.array(
            [[9, 7], [405, 24], [475, 704], [4, 694]],
            dtype=np.float32,
        )
        self.assertPaperQuad(
            perspective_camera_frame(rectified, destination=quad),
            quad,
            max_corner_error_px=14.0,
        )

    def test_small_but_complete_a4_projection(self):
        rectified, _ = make_rectified_scene(self.config)
        quad = np.array(
            [[535, 185], [750, 175], [772, 510], [510, 520]],
            dtype=np.float32,
        )
        frame_area = 1280.0 * 720.0
        projected_area_ratio = abs(float(cv2.contourArea(quad))) / frame_area
        self.assertGreater(projected_area_ratio, 0.06)
        self.assertLess(projected_area_ratio, 0.11)
        self.assertPaperQuad(
            perspective_camera_frame(rectified, destination=quad),
            quad,
            max_corner_error_px=10.0,
        )

    def test_enclosing_outer_frame_does_not_replace_a4(self):
        rectified, _ = make_rectified_scene(self.config)
        paper_quad = np.array(
            [[430, 64], [750, 70], [795, 646], [365, 640]],
            dtype=np.float32,
        )
        outer_quad = np.array(
            [[258, 7], [850, 7], [850, 713], [258, 713]],
            dtype=np.float32,
        )
        frame = perspective_camera_frame(
            rectified,
            destination=paper_quad,
            outer_quad=outer_quad,
        )
        self.assertPaperQuad(frame, paper_quad)

    def test_short_edge_gaps_and_double_edges(self):
        background_bgr = (62, 68, 73)
        rectified, _ = make_rectified_scene(self.config)
        quad = np.array(
            [[420, 26], [824, 42], [895, 688], [348, 672]],
            dtype=np.float32,
        )
        frame = perspective_camera_frame(
            rectified,
            destination=quad,
            background_bgr=background_bgr,
        )
        degraded = add_paper_edge_artifacts(
            frame, quad, background_bgr
        )
        self.assertPaperQuad(degraded, quad, max_corner_error_px=15.0)

    def test_fixed_paper_calibration_bypasses_frame_and_divider_search(self):
        quad = np.array(
            [
                [398.55, 27.0],
                [1124.65, 27.35],
                [1100.8, 543.42],
                [392.73, 538.18],
            ],
            dtype=np.float32,
        )
        frame = cv2.imdecode(
            np.fromfile(
                PROJECT_DIR
                / "tests"
                / "data"
                / "face_queen_input.png",
                dtype=np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(frame)
        config = replace(
            self.config,
            fixed_paper_quad_px=quad.astype(float).tolist(),
        )
        paper = detect_paper(frame, config)
        np.testing.assert_allclose(
            paper.frame_quad_px, order_quad(quad), atol=1e-6
        )
        self.assertAlmostEqual(
            paper.divider_y_px,
            0.5 * (paper.image_bgr.shape[0] - 1),
            delta=1e-6,
        )

    def test_fixed_calibration_uses_preview_total_area_tolerance(self):
        width = int(round(self.config.paper_short_mm * self.config.px_per_mm))
        height = int(round(self.config.paper_long_mm * self.config.px_per_mm))
        frame = np.full((height, width, 3), (170, 105, 45), dtype=np.uint8)
        divider_y = height // 2
        cv2.line(frame, (0, divider_y), (width - 1, divider_y), (20, 20, 20), 6)
        for x0 in (30, 180, 330, 480):
            cv2.rectangle(frame, (x0, 30), (x0 + 105, 342), (245, 245, 245), -1)

        quad = [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ]
        config = replace(
            self.config,
            fixed_paper_quad_px=quad,
        )
        _, pieces, _, _ = detect_paper_and_pieces(frame, config)
        total_area = sum(piece.area_mm2 for piece in pieces)

        self.assertEqual(len(pieces), 4)
        self.assertGreater(
            total_area,
            1.15 * config.target_short_max_mm * config.target_long_max_mm,
        )
        self.assertLessEqual(
            total_area,
            1.25 * config.target_short_max_mm * config.target_long_max_mm,
        )

    def assertPieceHalfIsNormalizedToTop(self, rectified: np.ndarray):
        paper = detect_paper(
            perspective_camera_frame(rectified),
            VisionConfig(
                expected_piece_count=1,
                # This helper deliberately uses a 14 x 10 mm fragment to
                # isolate orientation behaviour from official target-size
                # validation.
                target_short_min_mm=5.0,
                target_long_min_mm=5.0,
            ),
        )
        piece_pixels = cv2.inRange(
            paper.image_bgr,
            np.array((230, 230, 230), dtype=np.uint8),
            np.array((255, 255, 255), dtype=np.uint8),
        )
        moments = cv2.moments(piece_pixels, binaryImage=True)
        self.assertGreater(moments["m00"], 0.0)
        piece_center_y = moments["m01"] / moments["m00"]
        self.assertLess(piece_center_y, paper.divider_y_px)

    def test_single_small_piece_orients_when_half_areas_differ_under_20_percent(
        self,
    ):
        rectified = make_single_piece_orientation_scene(
            self.config,
            piece_half="bottom",
            balanced_shadows=True,
        )
        lab = cv2.cvtColor(rectified, cv2.COLOR_BGR2LAB).astype(np.float32)
        paper_lab = cv2.cvtColor(
            np.array([[[170, 105, 45]]], dtype=np.uint8),
            cv2.COLOR_BGR2LAB,
        )[0, 0].astype(np.float32)
        foreground = (
            np.linalg.norm(lab - paper_lab, axis=2)
            > self.config.foreground_min_lab_distance
        )
        divider = rectified.shape[0] // 2
        exclusion = int(
            round(
                self.config.divider_exclusion_mm
                * self.config.px_per_mm
            )
        )
        top_area = np.count_nonzero(foreground[: divider - exclusion])
        bottom_area = np.count_nonzero(foreground[divider + exclusion :])
        area_difference = abs(top_area - bottom_area) / max(
            top_area, bottom_area
        )
        self.assertLess(area_difference, 0.20)
        self.assertPieceHalfIsNormalizedToTop(rectified)

    def test_upper_half_shadow_does_not_override_small_piece_side(self):
        rectified = make_single_piece_orientation_scene(
            self.config,
            piece_half="bottom",
            upper_shadow=True,
        )
        self.assertPieceHalfIsNormalizedToTop(rectified)

    def test_broken_black_divider_beats_a_long_fragment_edge(self):
        paper_bgr = (170, 105, 45)
        rectified, _ = make_rectified_scene(
            self.config, paper_bgr=paper_bgr
        )
        height, width = rectified.shape[:2]
        center = height // 2
        # Replace the ideal synthetic divider with a broken, slightly wavy
        # hand-drawn line.
        cv2.rectangle(
            rectified,
            (0, center - 8),
            (width - 1, center + 8),
            paper_bgr,
            -1,
        )
        segment_width = width // 7
        for segment in range(7):
            if segment in (2, 5):
                continue
            x0 = segment * segment_width
            x1 = min(width - 1, (segment + 1) * segment_width - 3)
            y = center + (-2 if segment % 2 else 2)
            cv2.line(
                rectified,
                (x0, y),
                (x1, y),
                (15, 15, 15),
                5,
            )
        # A long straight fragment-like edge closer to the search-window
        # boundary must not replace the full-width centre divider.
        decoy_y = int(round(0.40 * height))
        cv2.line(
            rectified,
            (int(0.08 * width), decoy_y),
            (int(0.68 * width), decoy_y),
            (15, 15, 15),
            5,
        )
        coverage, detected = _line_response(
            rectified, horizontal_line=True, config=self.config
        )
        self.assertGreaterEqual(
            coverage, self.config.divider_min_coverage
        )
        self.assertAlmostEqual(detected, center, delta=10)

    def test_thin_divider_beats_a_broad_central_shadow(self):
        rectified, _ = make_rectified_scene(self.config)
        height, width = rectified.shape[:2]
        shadow_y = int(round(0.43 * height))
        overlay = rectified.astype(np.int16)
        overlay[shadow_y - 16 : shadow_y + 17] -= 35
        rectified = np.clip(overlay, 0, 255).astype(np.uint8)
        coverage, detected = _line_response(
            rectified, horizontal_line=True, config=self.config
        )
        self.assertGreaterEqual(
            coverage, self.config.divider_min_coverage
        )
        self.assertAlmostEqual(detected, height // 2, delta=8)


class PiecePolygonRegressionTests(unittest.TestCase):
    def test_short_print_bevel_is_completed_from_adjacent_edges(self):
        config = VisionConfig()
        measured_mm = np.array(
            [
                [145.333333, 101.666667],
                [148.666667, 111.0],
                [135.0, 135.333333],
                [113.0, 129.333333],
                [120.333333, 100.0],
            ],
            dtype=np.float64,
        )
        completed = _complete_clipped_polygon_corner(
            measured_mm * config.px_per_mm,
            config,
        )
        self.assertIsNotNone(completed)
        completed_mm = completed / config.px_per_mm
        self.assertEqual(len(completed_mm), 4)
        self.assertLess(
            float(
                np.min(
                    np.linalg.norm(
                        completed_mm
                        - np.array([153.6, 102.2]),
                        axis=1,
                    )
                )
            ),
            0.5,
        )
        self.assertLess(
            polygon_area(completed_mm)
            / polygon_area(measured_mm),
            1.06,
        )

    def test_rounded_outer_corners_are_restored_from_straight_edge_support(
        self,
    ):
        config = VisionConfig()
        mask = np.zeros((120, 180), dtype=np.uint8)
        left, top, right, bottom = 20, 20, 150, 95
        radius = 10
        cv2.rectangle(
            mask,
            (left + radius, top),
            (right - radius, bottom),
            255,
            -1,
        )
        cv2.rectangle(
            mask,
            (left, top + radius),
            (right, bottom - radius),
            255,
            -1,
        )
        for x in (left + radius, right - radius):
            for y in (top + radius, bottom - radius):
                cv2.circle(mask, (x, y), radius, 255, -1)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        measured, _ = _piece_polygon(contours[0], config)
        polygon = _refit_polygon_edges(contours[0], measured, config)
        expected = np.array(
            [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ],
            dtype=np.float64,
        )
        corner_errors = [
            float(np.min(np.linalg.norm(polygon - corner, axis=1)))
            for corner in expected
        ]
        self.assertEqual(len(polygon), 4)
        self.assertLess(max(corner_errors), 1.5)

        measured_mm = measured / config.px_per_mm
        piece = PieceObservation(
            piece_id=0,
            contour_px=contours[0],
            polygon_mm=measured_mm,
            mask=mask,
            source_center_mm=measured_mm.mean(axis=0),
            pickup_mm=measured_mm.mean(axis=0),
            area_mm2=polygon_area(measured_mm),
            approximation_error_mm=0.0,
        )
        recovered = _recovery_piece_hypothesis([piece], config)[0]
        self.assertAlmostEqual(
            recovered.area_mm2,
            (
                piece.area_mm2
                + polygon_area(recovered.polygon_mm)
                - polygon_area(piece.polygon_mm)
            ),
            places=6,
        )


class PuzzleSolverTests(unittest.TestCase):
    def setUp(self):
        self.config = VisionConfig(
            expected_piece_count=4,
            ambiguity_margin=0.0,
            solver_max_nodes=80000,
        )

    def test_figure2_geometry_solver(self):
        paper, pieces = make_rectified_scene(self.config)
        solution = solve_puzzle(
            pieces, paper, self.config, use_texture=False
        )
        short, long = solution.metrics.rectangle_size_mm
        self.assertAlmostEqual(short, 60.0, delta=2.0)
        self.assertAlmostEqual(long, 100.0, delta=2.0)
        self.assertGreater(solution.metrics.fill_ratio, 0.96)
        self.assertLess(solution.metrics.overlap_ratio, 0.018)
        self.assertEqual(set(solution.transforms), {0, 1, 2, 3})
        self.assertEqual(solution.explored_nodes, 48)
        self.assertEqual(solution.placement_spread_scale, 1.25)

    def test_figure2_strict_match_also_checks_seam_locked_layout(self):
        paper, pieces = make_rectified_scene(self.config)
        with patch.object(
            template_solver,
            "_seam_locked_transforms",
            wraps=template_solver._seam_locked_transforms,
        ) as seam_locked:
            solve_puzzle(
                pieces, paper, self.config, use_texture=False
            )
        self.assertGreater(seam_locked.call_count, 0)

    def test_figure2_accepts_official_tolerance_seam_gaps(self):
        metrics = AssemblyMetrics(
            fill_ratio=0.926,
            overlap_ratio=0.025,
            hole_ratio=0.000,
            boundary_p95_mm=3.1,
            boundary_p99_mm=3.7,
            side_coverage_min=0.651,
            convexity_ratio=0.971,
            corner_error_max_mm=4.3,
            rectangle_center_mm=np.array([50.0, 30.0]),
            rectangle_size_mm=(63.7, 104.5),
            rectangle_box_mm=np.array(
                [[0.0, 0.0], [102.5, 0.0],
                 [102.5, 61.0], [0.0, 61.0]]
            ),
            connected_components=1,
        )
        self.assertTrue(
            _template_metrics_are_valid(metrics, self.config)
        )
        metrics.overlap_ratio = 0.031
        self.assertFalse(
            _template_metrics_are_valid(metrics, self.config)
        )

    def test_figure2_accepts_measured_outer_side_dropout(self):
        metrics = AssemblyMetrics(
            fill_ratio=0.941,
            overlap_ratio=0.002,
            hole_ratio=0.005,
            boundary_p95_mm=6.0,
            boundary_p99_mm=12.0,
            side_coverage_min=0.504,
            convexity_ratio=0.977,
            corner_error_max_mm=4.0,
            rectangle_center_mm=np.array([50.0, 30.0]),
            rectangle_size_mm=(63.0, 106.5),
            rectangle_box_mm=np.array(
                [[0.0, 0.0], [106.5, 0.0],
                 [106.5, 63.0], [0.0, 63.0]]
            ),
            connected_components=1,
        )
        self.assertTrue(
            _template_metrics_are_valid(metrics, self.config)
        )
        metrics.side_coverage_min = 0.399
        self.assertFalse(
            _template_metrics_are_valid(metrics, self.config)
        )

    def test_figure2_hand_cut_is_not_misreported_as_a_rectangle(self):
        # Measured from the submitted iron-backed white-piece photograph.
        # Three pieces identify the fixed cut within about 1 mm.  The
        # hand-cut triangle is much noisier, but the seam-locked layout keeps
        # every actual shared junction below the published 20 mm limit.
        polygons = [
            np.array(
                [
                    [122.67, 11.67],
                    [122.33, 41.00],
                    [46.67, 27.00],
                    [23.00, 7.67],
                ]
            ),
            np.array(
                [
                    [153.00, 82.00],
                    [144.33, 73.33],
                    [151.00, 42.67],
                    [191.00, 17.33],
                ]
            ),
            np.array(
                [
                    [114.00, 59.33],
                    [78.33, 104.67],
                    [15.67, 55.33],
                ]
            ),
            np.array(
                [
                    [136.00, 111.33],
                    [117.67, 113.33],
                    [136.33, 84.00],
                    [149.33, 96.00],
                ]
            ),
        ]
        pieces = [
            PieceObservation(
                piece_id=piece_id,
                contour_px=polygon.copy(),
                polygon_mm=ensure_positive_winding(polygon),
                mask=np.zeros((2, 2), dtype=np.uint8),
                source_center_mm=polygon.mean(axis=0),
                pickup_mm=polygon.mean(axis=0),
                area_mm2=polygon_area(polygon),
                approximation_error_mm=0.0,
            )
            for piece_id, polygon in enumerate(polygons)
        ]
        with self.assertRaisesRegex(
            SolveError, "no piece reflection was attempted"
        ):
            solve_puzzle(
                pieces,
                np.zeros((2, 2, 3), dtype=np.uint8),
                VisionConfig(solver_mode="figure2"),
                use_texture=False,
            )

    def test_generic_solver_uses_bounded_mating_graph(self):
        config = VisionConfig(
            expected_piece_count=4,
            solver_mode="generic",
            solver_time_limit_ms=3000,
            ambiguity_margin=0.0,
        )
        paper, pieces = make_rectified_scene(config)
        solution = solve_puzzle(
            pieces, paper, config, use_texture=False
        )
        self.assertLess(solution.explored_nodes, 4000)
        self.assertAlmostEqual(
            solution.metrics.rectangle_size_mm[0], 60.0, delta=2.0
        )
        self.assertAlmostEqual(
            solution.metrics.rectangle_size_mm[1], 100.0, delta=2.0
        )

    def test_global_graph_reuses_repeated_partial_pose_checks(self):
        square = np.array(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        )
        pieces = self._pieces_from_polygons([square.copy() for _ in range(4)])
        pieces_by_id = {piece.piece_id: piece for piece in pieces}
        pair_cache = {
            (first, second): [
                rigid_matrix(0.0, np.array([6.0 * rank, 2.0 * rank]))
                for rank in range(4)
            ]
            for first in pieces_by_id
            for second in pieces_by_id
            if first != second
        }
        with (
            patch.object(
                solver_module,
                "_incremental_valid",
                return_value=True,
            ) as incremental_valid,
            patch.object(
                solver_module,
                "_partial_priority",
                return_value=0.0,
            ),
        ):
            _, explored, timed_out = solver_module._spanning_tree_layouts(
                pieces_by_id,
                pair_cache,
                sum(piece.area_mm2 for piece in pieces),
                VisionConfig(solver_max_nodes=10000),
                0.0,
                None,
            )
        self.assertFalse(timed_out)
        self.assertEqual(explored, 1024)
        self.assertLess(incremental_valid.call_count, 2000)

    def test_printed_four_piece_prioritizes_refinement_search(self):
        config = VisionConfig(
            expected_piece_count=4,
            solver_mode="generic",
            solver_time_limit_ms=3000,
            ambiguity_margin=0.0,
        )
        paper, pieces = make_rectified_scene(config, textured=True)
        with patch.object(
            solver_module,
            "_spanning_tree_layouts",
            wraps=solver_module._spanning_tree_layouts,
        ) as spanning_tree:
            solution = solve_puzzle(
                pieces, paper, config, use_texture=True
            )
        self.assertEqual(spanning_tree.call_count, 0)
        self.assertGreater(solution.metrics.fill_ratio, 0.95)
        self.assertEqual(set(solution.transforms), {0, 1, 2, 3})

    def test_generic_solver_handles_t_junction(self):
        polygons = [
            np.array(
                [[0.0, 0.0], [37.0, 0.0], [37.0, 60.0], [0.0, 60.0]]
            ),
            np.array(
                [[37.0, 0.0], [100.0, 0.0],
                 [100.0, 22.0], [37.0, 22.0]]
            ),
            np.array(
                [[37.0, 22.0], [100.0, 22.0],
                 [100.0, 60.0], [37.0, 60.0]]
            ),
        ]
        pieces = [
            PieceObservation(
                piece_id=piece_id,
                contour_px=polygon.copy(),
                polygon_mm=ensure_positive_winding(polygon),
                mask=np.zeros((2, 2), dtype=np.uint8),
                source_center_mm=polygon.mean(axis=0),
                pickup_mm=polygon.mean(axis=0),
                area_mm2=polygon_area(polygon),
                approximation_error_mm=0.0,
            )
            for piece_id, polygon in enumerate(polygons)
        ]
        config = VisionConfig(
            solver_mode="generic",
            solver_time_limit_ms=3000,
            ambiguity_margin=0.0,
        )
        solution = solve_puzzle(
            pieces,
            np.zeros((2, 2, 3), dtype=np.uint8),
            config,
            use_texture=False,
        )
        self.assertLess(solution.explored_nodes, 1000)
        self.assertAlmostEqual(
            solution.metrics.rectangle_size_mm[0], 60.0, delta=1.0
        )
        self.assertAlmostEqual(
            solution.metrics.rectangle_size_mm[1], 100.0, delta=1.0
        )

    def test_pair_graph_keeps_interior_long_edge_offset(self):
        # The 30 mm middle piece occupies x=20..50 on a 100 mm seam.  Neither
        # endpoint alignment can produce its true pose; the 20 mm offset comes
        # from another measured edge in the same puzzle.
        polygons = [
            np.array(
                [[0.0, 40.0], [100.0, 40.0],
                 [100.0, 60.0], [0.0, 60.0]]
            ),
            np.array(
                [[20.0, 0.0], [50.0, 0.0],
                 [50.0, 40.0], [20.0, 40.0]]
            ),
            np.array(
                [[0.0, 0.0], [20.0, 0.0],
                 [20.0, 40.0], [0.0, 40.0]]
            ),
            np.array(
                [[50.0, 0.0], [100.0, 0.0],
                 [100.0, 40.0], [50.0, 40.0]]
            ),
        ]
        pieces = [
            PieceObservation(
                piece_id=piece_id,
                contour_px=polygon.copy(),
                polygon_mm=ensure_positive_winding(polygon),
                mask=np.zeros((2, 2), dtype=np.uint8),
                source_center_mm=polygon.mean(axis=0),
                pickup_mm=polygon.mean(axis=0),
                area_mm2=polygon_area(polygon),
                approximation_error_mm=0.0,
            )
            for piece_id, polygon in enumerate(polygons)
        ]
        pieces_by_id = {piece.piece_id: piece for piece in pieces}
        fixed_id = 0
        moving_id = 1
        candidates = _candidate_transforms(
            pieces_by_id[moving_id],
            pieces_by_id,
            {fixed_id: np.eye(3)},
            VisionConfig(solver_mode="generic"),
        )
        target_polygon = ensure_positive_winding(polygons[moving_id])
        errors = [
            float(
                np.mean(
                    np.linalg.norm(
                        _transform_rigid(
                            pieces_by_id[moving_id].polygon_mm, matrix
                        )
                        - target_polygon,
                        axis=1,
                    )
                )
            )
            for matrix in candidates
        ]
        self.assertLess(min(errors), 0.1)

    def test_recovery_pair_graph_expands_interior_offset_coverage(self):
        # The middle piece belongs at x=35..55 on the 100 mm seam.  With the
        # fixed strip enumerated first, the historical two derived offsets are
        # 20 and 60 mm; 35 mm is the fourth unique hypothesis.  The primary
        # graph remains bounded as before, while recovery must retain it.
        polygons = [
            np.array(
                [[0.0, 40.0], [100.0, 40.0],
                 [100.0, 60.0], [0.0, 60.0]]
            ),
            np.array(
                [[35.0, 0.0], [55.0, 0.0],
                 [55.0, 40.0], [35.0, 40.0]]
            ),
            np.array(
                [[0.0, 0.0], [35.0, 0.0],
                 [35.0, 40.0], [0.0, 40.0]]
            ),
            np.array(
                [[55.0, 0.0], [100.0, 0.0],
                 [100.0, 40.0], [55.0, 40.0]]
            ),
        ]
        pieces = [
            PieceObservation(
                piece_id=piece_id,
                contour_px=polygon.copy(),
                polygon_mm=ensure_positive_winding(polygon),
                mask=np.zeros((2, 2), dtype=np.uint8),
                source_center_mm=polygon.mean(axis=0),
                pickup_mm=polygon.mean(axis=0),
                area_mm2=polygon_area(polygon),
                approximation_error_mm=0.0,
            )
            for piece_id, polygon in enumerate(polygons)
        ]
        pieces_by_id = {piece.piece_id: piece for piece in pieces}
        config = VisionConfig(solver_mode="generic")

        primary = _candidate_transforms(
            pieces_by_id[1],
            pieces_by_id,
            {0: np.eye(3)},
            config,
        )
        recovery = _candidate_transforms(
            pieces_by_id[1],
            pieces_by_id,
            {0: np.eye(3)},
            config,
            recovery_mode=True,
        )
        target_polygon = ensure_positive_winding(polygons[1])

        def minimum_error(candidates):
            return min(
                float(
                    np.mean(
                        np.linalg.norm(
                            _transform_rigid(
                                pieces_by_id[1].polygon_mm, matrix
                            )
                            - target_polygon,
                            axis=1,
                        )
                    )
                )
                for matrix in candidates
            )

        self.assertGreater(minimum_error(primary), 1.0)
        self.assertLess(minimum_error(recovery), 0.1)
        self.assertLessEqual(
            len(recovery), config.solver_max_pair_candidates
        )

    def test_pair_cycle_support_prefers_consistent_relative_pose(self):
        pose_01 = rigid_matrix(0.0, np.array([12.0, -3.0]))
        pose_12 = rigid_matrix(
            np.deg2rad(8.0), np.array([4.0, 16.0])
        )
        pose_02 = pose_01 @ pose_12
        wrong_pose_01 = rigid_matrix(
            np.deg2rad(7.0), np.array([20.0, -3.0])
        )
        pair_cache = {
            (0, 2): [pose_02],
            (1, 2): [pose_12],
        }

        consistent = _pair_cycle_support(
            0, 1, pose_01, pair_cache, [0, 1, 2, 3]
        )
        inconsistent = _pair_cycle_support(
            0, 1, wrong_pose_01, pair_cache, [0, 1, 2, 3]
        )
        unsupported = _pair_cycle_support(
            0, 1, pose_01, {}, [0, 1, 2, 3]
        )

        self.assertGreater(consistent, 0.99)
        self.assertLess(inconsistent, 0.05)
        self.assertEqual(unsupported, 0.0)

    def test_auto_falls_back_for_non_figure2_cross_cut(self):
        top = np.array([35.0, 0.0])
        right = np.array([100.0, 22.0])
        bottom = np.array([65.0, 60.0])
        left = np.array([0.0, 38.0])
        vertical = bottom - top
        horizontal = right - left
        parameter = np.linalg.solve(
            np.column_stack([vertical, -horizontal]),
            left - top,
        )[0]
        center = top + parameter * vertical
        polygons = [
            np.array([[0.0, 0.0], top, center, left]),
            np.array([top, [100.0, 0.0], right, center]),
            np.array([center, right, [100.0, 60.0], bottom]),
            np.array([left, center, bottom, [0.0, 60.0]]),
        ]
        pieces = self._pieces_from_polygons(polygons)
        config = VisionConfig(
            expected_piece_count=4,
            solver_mode="auto",
            solver_time_limit_ms=3000,
            ambiguity_margin=0.0,
        )
        solution = solve_puzzle(
            pieces,
            np.zeros((2, 2, 3), dtype=np.uint8),
            config,
            use_texture=False,
        )
        self.assertAlmostEqual(
            solution.metrics.rectangle_size_mm[0], 60.0, delta=2.0
        )
        self.assertAlmostEqual(
            solution.metrics.rectangle_size_mm[1], 100.0, delta=2.0
        )
        self.assertGreater(solution.metrics.fill_ratio, 0.96)

    def test_generic_skips_loose_slanted_strip_false_rectangle(self):
        top_x = [0.0, 18.0, 45.0, 72.0, 100.0]
        bottom_x = [0.0, 30.0, 54.0, 83.0, 100.0]
        polygons = [
            np.array(
                [
                    [top_x[index], 0.0],
                    [top_x[index + 1], 0.0],
                    [bottom_x[index + 1], 60.0],
                    [bottom_x[index], 60.0],
                ]
            )
            for index in range(4)
        ]
        pieces = self._pieces_from_polygons(polygons)
        config = VisionConfig(
            expected_piece_count=4,
            solver_mode="generic",
            solver_time_limit_ms=3000,
            ambiguity_margin=0.0,
        )
        solution = solve_puzzle(
            pieces,
            np.zeros((2, 2, 3), dtype=np.uint8),
            config,
            use_texture=False,
        )
        self.assertGreater(solution.metrics.fill_ratio, 0.96)
        self.assertAlmostEqual(
            solution.metrics.rectangle_size_mm[0], 60.0, delta=2.0
        )
        self.assertAlmostEqual(
            solution.metrics.rectangle_size_mm[1], 100.0, delta=2.0
        )

    def test_generic_solver_handles_measured_playing_card_cut(self):
        # Measured from the submitted four-fragment five-of-hearts photograph.
        # Piece 0 has one long seam shared by pieces 2 and 3.  The correct
        # piece-2 mate ranks below several locally attractive C2 alternatives,
        # so a fixed top-three-per-edge graph incorrectly reports no solution.
        polygons = [
            np.array(
                [[99.0, 42.7], [57.0, 51.0], [47.7, 10.0]]
            ),
            np.array(
                [
                    [170.3, 52.7],
                    [169.0, 106.3],
                    [150.0, 106.7],
                    [124.7, 54.0],
                ]
            ),
            np.array(
                [
                    [101.7, 58.3],
                    [128.7, 110.7],
                    [96.7, 111.7],
                    [79.0, 82.0],
                ]
            ),
            np.array(
                [
                    [64.0, 114.0],
                    [30.0, 110.3],
                    [48.7, 84.3],
                    [60.3, 90.0],
                ]
            ),
        ]
        pieces = self._pieces_from_polygons(polygons)
        config = VisionConfig(
            expected_piece_count=4,
            solver_mode="generic",
            solver_time_limit_ms=3000,
            solver_max_nodes=30000,
            ambiguity_margin=0.0,
        )
        solution = solve_puzzle(
            pieces,
            np.zeros((2, 2, 3), dtype=np.uint8),
            config,
            use_texture=False,
        )
        self.assertLess(solution.explored_nodes, 10000)
        self.assertGreater(solution.metrics.fill_ratio, 0.95)
        self.assertLess(solution.metrics.boundary_p95_mm, 2.5)
        self.assertGreater(solution.metrics.side_coverage_min, 0.95)
        self.assertGreater(solution.metrics.convexity_ratio, 0.97)
        self.assertLess(solution.metrics.corner_error_max_mm, 3.0)
        printed_solution = solve_puzzle(
            pieces,
            np.zeros((2, 2, 3), dtype=np.uint8),
            config,
            use_texture=True,
        )
        self.assertGreater(printed_solution.metrics.fill_ratio, 0.94)
        self.assertEqual(set(printed_solution.transforms), {0, 1, 2, 3})
        self.assertAlmostEqual(
            solution.metrics.rectangle_size_mm[0], 54.5, delta=2.0
        )
        self.assertAlmostEqual(
            solution.metrics.rectangle_size_mm[1], 86.0, delta=2.0
        )

    def test_custom_three_white_template_accepts_short_outer_endpoint(self):
        # A repeated capture of custom template 3 matches its three piece
        # shapes at 0.68 mm RMS, but one hand-cut endpoint reduces the raster
        # side coverage to about 0.67.
        polygons = [
            np.array(
                [
                    [88.67, 25.00],
                    [127.33, 70.67],
                    [77.00, 72.67],
                    [59.33, 51.00],
                ]
            ),
            np.array(
                [
                    [187.00, 86.33],
                    [117.67, 100.33],
                    [161.00, 31.00],
                    [180.00, 28.00],
                ]
            ),
            np.array(
                [
                    [101.33, 105.67],
                    [43.00, 129.00],
                    [35.33, 108.00],
                    [109.67, 79.00],
                ]
            ),
        ]
        pieces = self._pieces_from_polygons(polygons)
        solution = template_solver.solve_trained_template_bank(
            pieces,
            np.zeros((2, 2, 3), dtype=np.uint8),
            VisionConfig(),
        )
        self.assertLess(solution.explored_nodes, 20)
        self.assertGreater(solution.metrics.side_coverage_min, 0.65)
        self.assertLess(solution.metrics.side_coverage_min, 0.72)

    def test_custom_three_white_template_accepts_shadow_clipped_endpoint(self):
        # A strong shadow hides both ends of one physical fragment.  The
        # remaining vertices still identify custom template 3 within the
        # published 20 mm corresponding-point tolerance.
        polygons = [
            np.array(
                [
                    [170.33, 31.67],
                    [184.33, 47.00],
                    [137.67, 85.67],
                    [92.00, 29.67],
                ]
            ),
            np.array(
                [
                    [48.67, 59.67],
                    [100.33, 87.33],
                    [54.67, 107.67],
                    [30.33, 94.67],
                ]
            ),
            np.array(
                [
                    [175.67, 122.33],
                    [123.33, 132.67],
                    [118.67, 110.00],
                    [167.33, 100.00],
                    [179.00, 103.00],
                ]
            ),
        ]
        pieces = self._pieces_from_polygons(polygons)
        solution = template_solver.solve_trained_template_bank(
            pieces,
            np.zeros((2, 2, 3), dtype=np.uint8),
            VisionConfig(),
        )
        self.assertEqual(solution.trained_template_name, "custom-3-white")
        self.assertEqual(solution.placement_spread_scale, 1.0)
        self.assertLess(solution.explored_nodes, 20)
        self.assertLessEqual(solution.metrics.boundary_p99_mm, 20.0)

    def test_card_templates_include_distinct_110_by_80_variants(self):
        bank = {
            template.name: template
            for template in template_solver.trained_template_bank()
        }
        for base_name in (
            "custom-1-card-k",
            "custom-2-card-k",
        ):
            with self.subTest(base_name=base_name):
                base = bank[base_name]
                enlarged = bank[base_name + "-110x80"]
                size = (
                    np.max(np.vstack(enlarged.polygons), axis=0)
                    - np.min(np.vstack(enlarged.polygons), axis=0)
                )
                np.testing.assert_allclose(
                    size, np.array([110.0, 80.0]), atol=1e-6
                )
                pieces = self._pieces_from_polygons(
                    enlarged.polygons
                )
                enlarged_assignments, _ = (
                    template_solver._trained_assignments(
                        pieces, enlarged, VisionConfig()
                    )
                )
                base_assignments, _ = (
                    template_solver._trained_assignments(
                        pieces, base, VisionConfig()
                    )
                )
                self.assertGreater(len(enlarged_assignments), 0)
                self.assertEqual(len(base_assignments), 0)
                self.assertTrue(
                    template_solver._trained_metrics_are_valid(
                        solver_module._raster_metrics(
                            enlarged.polygons, VisionConfig()
                        ),
                        enlarged,
                        VisionConfig(),
                    )
                )
                self.assertTrue(enlarged.center_symmetric_rank)
                self.assertEqual(
                    enlarged.rank_triangle_index,
                    base.rank_triangle_index,
                )
                self.assertEqual(
                    enlarged.opposite_rank_piece_index,
                    base.opposite_rank_piece_index,
                )
                white_solution = (
                    template_solver.solve_trained_template_bank(
                        pieces,
                        np.full((2, 2, 3), 255, dtype=np.uint8),
                        VisionConfig(),
                        use_texture=False,
                    )
                )
                self.assertEqual(
                    white_solution.trained_template_name,
                    enlarged.name,
                )
                self.assertFalse(
                    white_solution.preserve_center_symmetry
                )
                with self.assertRaises(SolveError):
                    template_solver.solve_trained_template_bank(
                        pieces,
                        np.full((2, 2, 3), 255, dtype=np.uint8),
                        VisionConfig(),
                        use_texture=True,
                    )

    @staticmethod
    def _pieces_from_polygons(polygons):
        centers = [
            np.array([30.0, 25.0]),
            np.array([150.0, 30.0]),
            np.array([45.0, 105.0]),
            np.array([150.0, 110.0]),
        ]
        angles = [0.4, -0.7, 1.1, -0.2]
        pieces = []
        for piece_id, (polygon, target_center, angle) in enumerate(
            zip(polygons, centers, angles)
        ):
            polygon = ensure_positive_winding(polygon)
            matrix = rigid_matrix(
                angle,
                target_center - polygon.mean(axis=0),
            )
            transformed = apply_homography(polygon, matrix)
            pieces.append(
                PieceObservation(
                    piece_id=piece_id,
                    contour_px=transformed.copy(),
                    polygon_mm=transformed,
                    mask=np.zeros((2, 2), dtype=np.uint8),
                    source_center_mm=transformed.mean(axis=0),
                    pickup_mm=transformed.mean(axis=0),
                    area_mm2=polygon_area(transformed),
                    approximation_error_mm=0.0,
                )
            )
        return pieces

    def test_touching_card_fragments_remain_four_components(self):
        frame_path = (
            PROJECT_DIR
            / "tests"
            / "data"
            / "card_touching_four_input.png"
        )
        frame = cv2.imdecode(
            np.fromfile(frame_path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(frame)
        result = PuzzleVisionPipeline(
            VisionConfig(ambiguity_margin=0.0)
        ).process(frame, use_texture=True)
        self.assertEqual(result.status, "OK")
        self.assertEqual(len(result.pieces), 4)
        self.assertEqual(
            sorted(len(piece.polygon_mm) for piece in result.pieces),
            [3, 3, 3, 4],
        )

    def test_printed_card_hairline_seams_are_not_outer_boundary(self):
        frame_path = (
            PROJECT_DIR
            / "tests"
            / "data"
            / "card_hairline_seams_four_input.png"
        )
        frame = cv2.imdecode(
            np.fromfile(frame_path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(frame)
        result = PuzzleVisionPipeline(
            VisionConfig(ambiguity_margin=0.0)
        ).process(frame, use_texture=True)
        self.assertEqual(result.status, "OK")
        self.assertEqual(len(result.pieces), 4)
        self.assertEqual(
            sorted(len(piece.polygon_mm) for piece in result.pieces),
            [3, 3, 3, 4],
        )
        self.assertLess(result.solution.metrics.boundary_p95_mm, 8.0)
        self.assertLess(result.solution.metrics.boundary_p99_mm, 10.0)
        self.assertGreater(
            result.solution.metrics.side_coverage_min, 0.97
        )
        self.assertLess(result.solution.explored_nodes, 4000)

    def test_repeated_card_capture_finishes_within_board_node_budget(self):
        frame_path = (
            PROJECT_DIR
            / "tests"
            / "data"
            / "card_repeat_capture_four_input.png"
        )
        frame = cv2.imdecode(
            np.fromfile(frame_path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(frame)
        result = PuzzleVisionPipeline(
            VisionConfig(ambiguity_margin=0.0)
        ).process(frame, use_texture=True)
        self.assertEqual(result.status, "OK")
        self.assertEqual(len(result.pieces), 4)
        self.assertTrue(result.solution.search_complete)
        self.assertLess(result.solution.explored_nodes, 4000)
        self.assertGreater(result.solution.metrics.fill_ratio, 0.94)
        self.assertGreater(
            result.solution.metrics.side_coverage_min, 0.97
        )

    def test_rotated_card_fragments_finish_within_board_node_budget(self):
        frame_path = (
            PROJECT_DIR
            / "tests"
            / "data"
            / "card_rotated_fragments_four_input.png"
        )
        frame = cv2.imdecode(
            np.fromfile(frame_path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(frame)
        result = PuzzleVisionPipeline(
            VisionConfig(ambiguity_margin=0.0)
        ).process(frame, use_texture=True)
        self.assertEqual(result.status, "OK")
        self.assertEqual(len(result.pieces), 4)
        self.assertTrue(result.solution.search_complete)
        self.assertLess(result.solution.explored_nodes, 4000)
        self.assertGreater(result.solution.metrics.fill_ratio, 0.94)
        self.assertGreater(
            result.solution.metrics.side_coverage_min, 0.97
        )

    def test_card_edge_print_notch_uses_safe_hull_repair(self):
        frame_path = (
            PROJECT_DIR
            / "tests"
            / "data"
            / "card_edge_print_notch_four_input.png"
        )
        frame = cv2.imdecode(
            np.fromfile(frame_path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(frame)
        result = PuzzleVisionPipeline(
            VisionConfig(ambiguity_margin=0.0)
        ).process(frame, use_texture=True)
        self.assertEqual(result.status, "OK")
        self.assertEqual(len(result.pieces), 4)
        self.assertEqual(
            sorted(len(piece.polygon_mm) for piece in result.pieces),
            [3, 4, 4, 4],
        )
        self.assertLess(
            max(
                piece.approximation_error_mm
                for piece in result.pieces
            ),
            3.0,
        )
        self.assertTrue(result.solution.search_complete)
        self.assertLess(result.solution.explored_nodes, 850)
        self.assertGreater(result.solution.metrics.fill_ratio, 0.94)
        self.assertLess(result.solution.metrics.hole_ratio, 0.01)

    def test_four_quadrilateral_card_finishes_within_board_budget(self):
        frame_path = (
            PROJECT_DIR
            / "tests"
            / "data"
            / "card_four_quadrilaterals_input.png"
        )
        frame = cv2.imdecode(
            np.fromfile(frame_path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(frame)
        result = PuzzleVisionPipeline(
            VisionConfig(ambiguity_margin=0.0)
        ).process(frame, use_texture=True)
        self.assertEqual(result.status, "OK")
        self.assertEqual(
            [len(piece.polygon_mm) for piece in result.pieces],
            [4, 4, 4, 4],
        )
        self.assertTrue(result.solution.search_complete)
        self.assertLess(result.solution.explored_nodes, 768)
        self.assertGreater(result.solution.metrics.fill_ratio, 0.94)
        self.assertGreater(
            result.solution.metrics.side_coverage_min, 0.98
        )
        self.assertLess(result.solution.metrics.hole_ratio, 0.01)

    def test_all_notched_card_uses_full_colour_segmentation(self):
        frame_path = (
            PROJECT_DIR
            / "tests"
            / "data"
            / "card_all_notched_four_input.png"
        )
        frame = cv2.imdecode(
            np.fromfile(frame_path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(frame)
        result = PuzzleVisionPipeline(
            VisionConfig(ambiguity_margin=0.0)
        ).process(frame, use_texture=True)
        self.assertEqual(result.status, "OK")
        self.assertEqual(
            [len(piece.polygon_mm) for piece in result.pieces],
            [4, 3, 4, 4],
        )
        self.assertLess(
            max(
                piece.approximation_error_mm
                for piece in result.pieces
            ),
            1.5,
        )
        self.assertTrue(result.solution.search_complete)
        self.assertLess(result.solution.explored_nodes, 500)
        self.assertGreater(result.solution.metrics.fill_ratio, 0.92)
        self.assertGreater(
            result.solution.metrics.side_coverage_min, 0.92
        )
        self.assertLess(result.solution.metrics.hole_ratio, 0.02)

    def test_card_strips_select_full_paper_and_whole_face_order(self):
        frame_path = (
            PROJECT_DIR
            / "tests"
            / "data"
            / "card_four_strips_paper_candidate_input.png"
        )
        frame = cv2.imdecode(
            np.fromfile(frame_path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(frame)
        config = VisionConfig()
        result = PuzzleVisionPipeline(config).process(
            frame, use_texture=True
        )
        self.assertEqual(result.status, "OK")
        self.assertEqual(len(result.pieces), 4)
        self.assertGreater(
            float(np.ptp(result.paper.frame_quad_px[:, 0])),
            650.0,
        )
        self.assertTrue(_is_card_strip_partition(result.pieces))
        self.assertTrue(result.solution.search_complete)
        self.assertLess(result.solution.explored_nodes, 500)
        self.assertGreater(result.solution.metrics.fill_ratio, 0.94)
        self.assertGreater(
            result.solution.metrics.side_coverage_min, 0.95
        )
        self.assertLess(result.solution.texture_score, 0.10)
        self.assertGreater(
            result.solution.score_margin, config.ambiguity_margin
        )

    def test_rich_face_cards_solve_from_fixed_camera_calibration(self):
        fixed_quad = [
            [398.55, 27.0],
            [1124.65, 27.35],
            [1100.8, 543.42],
            [392.73, 538.18],
        ]
        config = VisionConfig(
            fixed_paper_quad_px=fixed_quad,
            ambiguity_margin=0.0,
            solver_time_limit_ms=15000,
            solver_total_time_limit_ms=30000,
        )
        for filename in (
            "face_queen_input.png",
            "face_joker_input.png",
            "face_king_input.png",
            "face_jack_input.png",
        ):
            with self.subTest(filename=filename):
                frame_path = PROJECT_DIR / "tests" / "data" / filename
                frame = cv2.imdecode(
                    np.fromfile(frame_path, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                self.assertIsNotNone(frame)
                result = PuzzleVisionPipeline(config).process(
                    frame, use_texture=True
                )
                self.assertEqual(result.status, "OK")
                self.assertEqual(len(result.pieces), 4)
                np.testing.assert_allclose(
                    result.paper.frame_quad_px,
                    np.asarray(fixed_quad, dtype=np.float32),
                    atol=1e-5,
                )
                self.assertTrue(result.solution.search_complete)
                self.assertLess(result.solution.explored_nodes, 6000)
                self.assertGreaterEqual(
                    result.solution.metrics.fill_ratio, 0.90
                )
                self.assertGreaterEqual(
                    result.solution.metrics.side_coverage_min, 0.84
                )
                self.assertLessEqual(
                    result.solution.metrics.hole_ratio, 0.025
                )

    def test_failed_primary_solve_uses_distinct_recovery_budget(self):
        polygons = [
            np.array(
                [[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]]
            )
            for _ in range(4)
        ]
        pieces = self._pieces_from_polygons(polygons)
        metrics = AssemblyMetrics(
            fill_ratio=1.0,
            overlap_ratio=0.0,
            hole_ratio=0.0,
            boundary_p95_mm=0.0,
            boundary_p99_mm=0.0,
            side_coverage_min=1.0,
            convexity_ratio=1.0,
            corner_error_max_mm=0.0,
            rectangle_center_mm=np.array([50.0, 50.0]),
            rectangle_size_mm=(60.0, 90.0),
            rectangle_box_mm=np.array(
                [[20.0, 5.0], [80.0, 5.0], [80.0, 95.0], [20.0, 95.0]]
            ),
            connected_components=1,
        )
        recovered = AssemblySolution(
            transforms={
                piece.piece_id: np.eye(3, dtype=np.float64)
                for piece in pieces
            },
            metrics=metrics,
            geometry_score=0.0,
            texture_score=0.0,
            total_score=0.0,
            explored_nodes=10,
            valid_candidate_count=1,
            search_complete=True,
        )
        config = VisionConfig(
            solver_time_limit_ms=15000,
            solver_total_time_limit_ms=30000,
        )
        with (
            patch(
                "puzzle_vision.pipeline.py_time.monotonic",
                side_effect=[0.0, 0.0, 20.0],
            ),
            patch(
                "puzzle_vision.pipeline.solve_puzzle",
                side_effect=[
                    SolveError("primary timed out"),
                    recovered,
                ],
            ) as mocked_solve,
            patch(
                "puzzle_vision.pipeline._recovery_piece_hypothesis",
                return_value=pieces,
            ) as mocked_refit,
        ):
            result = _solve_with_recovery(
                pieces,
                np.zeros((10, 10, 3), dtype=np.uint8),
                config,
                use_texture=True,
            )

        self.assertIs(result, recovered)
        self.assertEqual(mocked_solve.call_count, 2)
        mocked_refit.assert_called_once()
        self.assertFalse(
            mocked_solve.call_args_list[0].kwargs["recovery_mode"]
        )
        self.assertTrue(
            mocked_solve.call_args_list[1].kwargs["recovery_mode"]
        )
        self.assertEqual(
            mocked_solve.call_args_list[0].args[2].solver_time_limit_ms,
            15000,
        )
        self.assertEqual(
            mocked_solve.call_args_list[1].args[2].solver_time_limit_ms,
            10000,
        )
        with (
            patch(
                "puzzle_vision.pipeline.py_time.monotonic",
                side_effect=[0.0, 0.0],
            ),
            patch(
                "puzzle_vision.pipeline.solve_puzzle",
                return_value=recovered,
            ) as successful_solve,
        ):
            result = _solve_with_recovery(
                pieces,
                np.zeros((10, 10, 3), dtype=np.uint8),
                config,
                use_texture=True,
            )
        self.assertIs(result, recovered)
        self.assertEqual(successful_solve.call_count, 1)
        self.assertFalse(
            successful_solve.call_args.kwargs["recovery_mode"]
        )

    def test_full_detection_and_pose_pipeline(self):
        rectified, _ = make_rectified_scene(self.config)
        frame = perspective_camera_frame(rectified)
        result = PuzzleVisionPipeline(self.config).process(
            frame, use_texture=False
        )
        self.assertEqual(len(result.pieces), 4)
        self.assertEqual(len(result.commands), 4)
        self.assertEqual(result.debug_bgr.shape, frame.shape)
        self.assertEqual(
            result.rectified_debug_bgr.shape,
            result.paper.image_bgr.shape,
        )
        changed = np.any(result.debug_bgr != frame, axis=2)
        self.assertGreater(np.count_nonzero(changed), 500)
        self.assertGreater(
            np.count_nonzero(~changed), int(0.50 * changed.size)
        )
        self.assertAlmostEqual(
            result.paper.width_mm, 210.0, delta=1.0
        )
        self.assertAlmostEqual(
            result.paper.height_mm, 297.0, delta=1.0
        )
        baseline_commands = _build_commands(
            result.paper,
            result.pieces,
            result.solution,
            replace(self.config, placement_spread_mm=0.0),
        )
        baseline_by_id = {
            command.piece_id: command for command in baseline_commands
        }
        baseline_points = np.vstack(
            [
                command.target_polygon_paper_mm
                for command in baseline_commands
            ]
        )
        assembly_center = 0.5 * (
            baseline_points.min(axis=0) + baseline_points.max(axis=0)
        )
        median_piece_area = float(
            np.median([piece.area_mm2 for piece in result.pieces])
        )
        area_by_id = {
            piece.piece_id: piece.area_mm2
            for piece in result.pieces
        }
        self.assertEqual(
            [command.piece_id for command in result.commands],
            [
                piece.piece_id
                for piece in sorted(
                    result.pieces,
                    key=lambda piece: (-piece.area_mm2, piece.piece_id),
                )
            ],
        )
        for command in result.commands:
            baseline = baseline_by_id[command.piece_id]
            spread_offset = (
                command.target_center_paper_mm
                - baseline.target_center_paper_mm
            )
            expected_spread = float(
                np.clip(
                    self.config.placement_spread_mm
                    * result.solution.placement_spread_scale
                    * np.sqrt(
                        median_piece_area
                        / area_by_id[command.piece_id]
                    ),
                    0.8
                    * self.config.placement_spread_mm
                    * result.solution.placement_spread_scale,
                    1.6
                    * self.config.placement_spread_mm
                    * result.solution.placement_spread_scale,
                )
            )
            self.assertAlmostEqual(
                float(np.linalg.norm(spread_offset)),
                expected_spread,
                delta=1e-6,
            )
            radial_direction = (
                baseline.target_center_paper_mm - assembly_center
            )
            self.assertGreater(
                float(np.dot(spread_offset, radial_direction)),
                0.0,
            )
            np.testing.assert_allclose(
                command.place_paper_mm - baseline.place_paper_mm,
                spread_offset,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                command.pickup_robot_mm,
                baseline.pickup_robot_mm,
                atol=1e-6,
            )
            self.assertAlmostEqual(
                command.rotation_robot_deg,
                baseline.rotation_robot_deg,
                delta=1e-6,
            )
            np.testing.assert_allclose(
                command.target_polygon_paper_mm
                - baseline.target_polygon_paper_mm,
                np.broadcast_to(
                    spread_offset,
                    command.target_polygon_paper_mm.shape,
                ),
                atol=1e-6,
            )
            polygon = command.target_polygon_paper_mm
            self.assertGreater(
                float(polygon[:, 1].min()),
                result.paper.divider_y_mm,
            )
            self.assertLess(
                float(polygon[:, 1].max()),
                result.paper.height_mm,
            )
            self.assertGreater(float(polygon[:, 0].min()), 0.0)
            self.assertLess(
                float(polygon[:, 0].max()), result.paper.width_mm
            )
            angle = np.radians(command.rotation_cw_deg)
            rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
            expected_offset = rotation @ (
                command.pickup_paper_mm
                - command.source_center_paper_mm
            )
            actual_offset = (
                command.place_paper_mm
                - command.target_center_paper_mm
            )
            np.testing.assert_allclose(
                actual_offset, expected_offset, atol=1e-6
            )
        json.dumps(result.to_dict())

    def test_landscape_camera_view_is_normalized(self):
        rectified, _ = make_rectified_scene(self.config)
        frame = perspective_camera_frame(rectified, landscape=True)
        result = PuzzleVisionPipeline(self.config).process(
            frame, use_texture=False
        )
        self.assertEqual(len(result.pieces), 4)
        self.assertAlmostEqual(
            result.paper.width_mm, 210.0, delta=1.0
        )
        self.assertAlmostEqual(
            result.paper.height_mm, 297.0, delta=1.0
        )
        self.assertLess(
            max(piece.source_center_mm[1] for piece in result.pieces),
            result.paper.divider_y_mm,
        )
        for command in result.commands:
            expected_frame_px = apply_homography(
                (
                    command.pickup_paper_mm
                    * result.paper.px_per_mm
                ).reshape(1, 2),
                result.paper.paper_px_to_frame,
            )[0]
            np.testing.assert_allclose(
                command.pickup_robot_mm,
                expected_frame_px,
                atol=1e-6,
            )

    def test_180_degree_view_and_colored_divider_are_normalized(self):
        rectified, _ = make_rectified_scene(
            self.config,
            divider_bgr=(20, 210, 210),
        )
        upside_down = cv2.rotate(rectified, cv2.ROTATE_180)
        frame = perspective_camera_frame(upside_down)
        result = PuzzleVisionPipeline(self.config).process(
            frame, use_texture=False
        )
        self.assertEqual(len(result.pieces), 4)
        self.assertLess(
            max(piece.source_center_mm[1] for piece in result.pieces),
            result.paper.divider_y_mm,
        )

    def test_largest_allowed_single_piece(self):
        polygon = np.array(
            [[0.0, 0.0], [120.0, 0.0], [120.0, 90.0], [0.0, 90.0]]
        )
        piece = PieceObservation(
            piece_id=0,
            contour_px=polygon.copy(),
            polygon_mm=polygon,
            mask=np.zeros((2, 2), dtype=np.uint8),
            source_center_mm=np.array([60.0, 45.0]),
            pickup_mm=np.array([60.0, 45.0]),
            area_mm2=10800.0,
            approximation_error_mm=0.0,
        )
        solution = solve_puzzle(
            [piece],
            np.zeros((2, 2, 3), dtype=np.uint8),
            VisionConfig(),
            use_texture=False,
        )
        self.assertEqual(solution.valid_candidate_count, 1)
        self.assertAlmostEqual(
            solution.metrics.rectangle_size_mm[0], 90.0, delta=1.0
        )
        self.assertAlmostEqual(
            solution.metrics.rectangle_size_mm[1], 120.0, delta=1.0
        )

    def test_cut_corner_is_not_accepted_as_a_rectangle(self):
        polygon = np.array(
            [[10.0, 0.0], [100.0, 0.0], [100.0, 60.0],
             [0.0, 60.0], [0.0, 10.0]]
        )
        piece = PieceObservation(
            piece_id=0,
            contour_px=polygon.copy(),
            polygon_mm=ensure_positive_winding(polygon),
            mask=np.zeros((2, 2), dtype=np.uint8),
            source_center_mm=polygon.mean(axis=0),
            pickup_mm=polygon.mean(axis=0),
            area_mm2=polygon_area(polygon),
            approximation_error_mm=0.0,
        )
        with self.assertRaises(SolveError):
            solve_puzzle(
                [piece],
                np.zeros((2, 2, 3), dtype=np.uint8),
                VisionConfig(),
                use_texture=False,
            )

    def test_nonconvex_shared_seam_is_not_counted_as_overlap(self):
        polygons = [
            np.array(
                [[0.0, 0.0], [50.0, 0.0], [40.0, 30.0],
                 [50.0, 60.0], [0.0, 60.0]]
            ),
            np.array(
                [[50.0, 0.0], [100.0, 0.0], [100.0, 60.0],
                 [50.0, 60.0], [40.0, 30.0]]
            ),
        ]
        pieces = [
            PieceObservation(
                piece_id=piece_id,
                contour_px=polygon.copy(),
                polygon_mm=ensure_positive_winding(polygon),
                mask=np.zeros((2, 2), dtype=np.uint8),
                source_center_mm=polygon.mean(axis=0),
                pickup_mm=polygon.mean(axis=0),
                area_mm2=polygon_area(polygon),
                approximation_error_mm=0.0,
            )
            for piece_id, polygon in enumerate(polygons)
        ]
        solution = solve_puzzle(
            pieces,
            np.zeros((2, 2, 3), dtype=np.uint8),
            VisionConfig(),
            use_texture=False,
        )
        self.assertTrue(solution.search_complete)
        self.assertLess(solution.metrics.overlap_ratio, 0.005)

    def test_two_pair_group_check_reuses_internal_seams(self):
        polygons = {
            0: np.array(
                [[0.0, 0.0], [50.0, 0.0], [50.0, 30.0], [0.0, 30.0]]
            ),
            1: np.array(
                [[0.0, 30.0], [50.0, 30.0], [50.0, 60.0], [0.0, 60.0]]
            ),
            2: np.array(
                [[50.0, 0.0], [100.0, 0.0], [100.0, 30.0], [50.0, 30.0]]
            ),
            3: np.array(
                [[50.0, 30.0], [100.0, 30.0], [100.0, 60.0], [50.0, 60.0]]
            ),
        }
        config = VisionConfig()
        self.assertTrue(
            _pair_groups_incrementally_valid(
                polygons, (2, 3), 6000.0, config
            )
        )
        overlapping = {
            piece_id: polygon.copy()
            for piece_id, polygon in polygons.items()
        }
        overlapping[2][:, 0] -= 10.0
        overlapping[3][:, 0] -= 10.0
        self.assertFalse(
            _pair_groups_incrementally_valid(
                overlapping, (2, 3), 6000.0, config
            )
        )

    def test_unfinished_priority_skips_final_rectangle_analysis(self):
        polygons = {
            0: np.array(
                [[0.0, 0.0], [50.0, 0.0], [50.0, 30.0], [0.0, 30.0]]
            ),
            1: np.array(
                [[50.0, 0.0], [100.0, 0.0], [100.0, 30.0], [50.0, 30.0]]
            ),
        }
        with (
            patch.object(
                solver_module,
                "_contacts_between",
                side_effect=AssertionError("final contact analysis was used"),
            ),
            patch.object(
                solver_module,
                "_rectangle_feature_scores",
                side_effect=AssertionError("final rectangle analysis was used"),
            ),
        ):
            score = solver_module._unfinished_priority(
                polygons,
                6000.0,
                VisionConfig(),
                0,
            )
        self.assertTrue(np.isfinite(score))

    def test_relative_pose_overlap_is_cached(self):
        polygons = [
            np.array(
                [[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]]
            ),
            np.array(
                [[30.0, 0.0], [60.0, 0.0], [60.0, 30.0], [30.0, 30.0]]
            ),
        ]
        pieces = [
            PieceObservation(
                piece_id=piece_id,
                contour_px=polygon.copy(),
                polygon_mm=ensure_positive_winding(polygon),
                mask=np.zeros((2, 2), dtype=np.uint8),
                source_center_mm=polygon.mean(axis=0),
                pickup_mm=polygon.mean(axis=0),
                area_mm2=polygon_area(polygon),
                approximation_error_mm=0.0,
            )
            for piece_id, polygon in enumerate(polygons)
        ]
        pieces_by_id = {piece.piece_id: piece for piece in pieces}
        transforms = {
            0: np.eye(3, dtype=np.float64),
            1: np.eye(3, dtype=np.float64),
        }
        placed = {
            piece_id: piece.polygon_mm.copy()
            for piece_id, piece in pieces_by_id.items()
        }
        cache = {}
        with patch.object(
            solver_module,
            "_polygon_overlap_area",
            wraps=solver_module._polygon_overlap_area,
        ) as overlap:
            first = solver_module._cached_incremental_valid(
                placed,
                transforms,
                1,
                pieces_by_id,
                1800.0,
                VisionConfig(),
                cache,
            )
            second = solver_module._cached_incremental_valid(
                placed,
                transforms,
                1,
                pieces_by_id,
                1800.0,
                VisionConfig(),
                cache,
            )
        self.assertEqual(first, second)
        self.assertEqual(overlap.call_count, 1)
        self.assertEqual(len(cache), 1)

    def test_nearly_collinear_nested_convex_overlap_is_not_zero(self):
        outer = np.array(
            [
                [176.93137833, 52.47402302],
                [118.06862167, 54.19264365],
                [131.00267366, 24.90595214],
                [165.39152360, 21.80285346],
            ]
        )
        inner = np.array(
            [
                [164.59128709, 52.83431765],
                [130.40871291, 53.83234901],
                [145.36045545, 25.53235185],
                [157.69017284, 29.55731040],
            ]
        )
        self.assertGreater(
            _polygon_overlap_area(outer, inner),
            500.0,
        )

    def test_measured_convex_shared_edge_is_not_full_overlap(self):
        # OpenCV 4.12 intersectConvexConvex can report the complete first
        # polygon area for these two measured card fragments even though their
        # separating-axis penetration is zero and they only share a seam.
        piece_a = np.array(
            [
                [124.57020034, 53.79945158],
                [150.09646633, 106.86721508],
                [118.08101708, 106.97214143],
                [101.25070220, 76.82313277],
            ]
        )
        piece_b = np.array(
            [
                [101.33147697, 76.96782877],
                [118.00024231, 106.82744542],
                [85.99389353, 106.63229837],
                [83.87095495, 93.83716828],
            ]
        )
        self.assertLess(_polygon_overlap_area(piece_a, piece_b), 1.0)

    def test_search_limit_is_never_reported_complete(self):
        config = VisionConfig(
            solver_max_nodes=1,
            solver_time_limit_ms=0,
        )
        paper, pieces, _ = make_ambiguous_textured_pair(config)
        with self.assertRaisesRegex(SolveError, "search limit reached"):
            solve_puzzle(pieces, paper, config, use_texture=False)

    def test_texture_mode_repairs_a_small_print_colour_notch(self):
        contour = (
            np.array(
                [
                    [101.7, 58.3],
                    [79.7, 83.3],
                    [96.7, 111.7],
                    [128.7, 110.7],
                    [113.0, 81.3],
                    [109.7, 89.3],
                    [99.0, 83.7],
                    [111.0, 77.3],
                ],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            * 3.0
        )
        with self.assertRaises(DetectionError):
            _textured_piece_contour(
                contour,
                VisionConfig(texture_enabled=False),
            )

        repaired, polygon, _ = _textured_piece_contour(
            contour,
            VisionConfig(texture_enabled=True),
        )
        self.assertEqual(len(polygon), 4)
        self.assertGreater(
            abs(float(cv2.contourArea(repaired))),
            abs(float(cv2.contourArea(contour))),
        )

    def test_nearly_collinear_point_does_not_split_a_long_seam(self):
        contour = np.array(
            [
                [15.67, 55.33],
                [44.00, 55.00],
                [114.00, 59.33],
                [78.33, 104.67],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        contour *= 3.0

        polygon, _ = _piece_polygon(
            contour,
            VisionConfig(),
        )
        self.assertEqual(len(polygon), 3)
        lengths_mm = np.linalg.norm(
            np.roll(polygon, -1, axis=0) - polygon,
            axis=1,
        ) / 3.0
        self.assertGreater(float(lengths_mm.max()), 95.0)

    def test_piece_aligned_with_paper_edge_is_segmented(self):
        config = VisionConfig(
            expected_piece_count=1,
            target_short_min_mm=40.0,
            target_long_min_mm=80.0,
        )
        paper = self._single_piece_paper(
            config,
            (0.0, 25.0),
            (100.0, 85.0),
        )
        pieces, _, _ = segment_pieces(paper, config)
        self.assertEqual(len(pieces), 1)
        self.assertLess(
            float(pieces[0].polygon_mm[:, 0].min()),
            1.0,
        )

    def test_piece_inside_old_paper_margin_is_not_clipped(self):
        config = VisionConfig(
            expected_piece_count=1,
            target_short_min_mm=40.0,
            target_long_min_mm=80.0,
        )
        paper = self._single_piece_paper(
            config,
            (1.0, 25.0),
            (101.0, 85.0),
        )
        pieces, _, _ = segment_pieces(paper, config)
        self.assertEqual(len(pieces), 1)
        self.assertGreater(float(pieces[0].polygon_mm[:, 0].min()), 0.0)

    def test_piece_inside_old_divider_margin_is_not_clipped(self):
        config = VisionConfig(
            expected_piece_count=1,
            target_short_min_mm=40.0,
            target_long_min_mm=80.0,
        )
        divider_y_mm = 0.5 * config.paper_long_mm
        paper = self._single_piece_paper(
            config,
            (55.0, divider_y_mm - 61.33),
            (155.0, divider_y_mm - 1.33),
        )
        pieces, _, _ = segment_pieces(paper, config)
        self.assertEqual(len(pieces), 1)
        gap_mm = (
            paper.divider_y_px / config.px_per_mm
            - float(pieces[0].polygon_mm[:, 1].max())
        )
        self.assertLess(gap_mm, config.divider_exclusion_mm)
        self.assertGreater(gap_mm, 0.0)

    def test_piece_touching_actual_divider_is_segmented(self):
        config = VisionConfig(
            expected_piece_count=1,
            target_short_min_mm=40.0,
            target_long_min_mm=80.0,
        )
        divider_y_mm = 0.5 * config.paper_long_mm
        paper = self._single_piece_paper(
            config,
            (55.0, divider_y_mm - 60.0),
            (155.0, divider_y_mm),
        )
        pieces, _, _ = segment_pieces(paper, config)
        self.assertEqual(len(pieces), 1)

    @staticmethod
    def _single_piece_paper(
        config: VisionConfig,
        corner_min_mm,
        corner_max_mm,
    ):
        width = int(round(config.paper_short_mm * config.px_per_mm))
        height = int(round(config.paper_long_mm * config.px_per_mm))
        image = np.full((height, width, 3), (170, 105, 45), dtype=np.uint8)
        divider_y = height // 2
        cv2.line(
            image,
            (0, divider_y),
            (width - 1, divider_y),
            (20, 20, 20),
            max(2, int(round(2.0 * config.px_per_mm))),
        )
        cv2.rectangle(
            image,
            tuple(
                np.round(
                    np.asarray(corner_min_mm) * config.px_per_mm
                ).astype(int)
            ),
            tuple(
                np.round(
                    np.asarray(corner_max_mm) * config.px_per_mm
                ).astype(int)
            ),
            (245, 245, 245),
            -1,
        )
        return PaperObservation(
            image_bgr=image,
            frame_quad_px=np.array(
                [
                    [0.0, 0.0],
                    [width - 1.0, 0.0],
                    [width - 1.0, height - 1.0],
                    [0.0, height - 1.0],
                ]
            ),
            frame_to_paper_px=np.eye(3),
            paper_px_to_frame=np.eye(3),
            divider_y_px=0.5 * (height - 1),
            paper_color_lab=np.zeros(3),
            detection_score=1.0,
            px_per_mm=config.px_per_mm,
        )

    def test_pose_key_wraps_equivalent_half_turns(self):
        positive = rigid_matrix(np.pi, np.array([12.0, 34.0]))
        negative = rigid_matrix(-np.pi, np.array([12.0, 34.0]))
        self.assertEqual(_pose_key(positive), _pose_key(negative))

    def test_reflected_robot_axes_reverse_rotation_sign(self):
        rectified, _ = make_rectified_scene(self.config)
        result = PuzzleVisionPipeline(self.config).process(
            perspective_camera_frame(rectified), use_texture=False
        )
        px_to_reflected_mm = np.array(
            [
                [1.0 / result.paper.px_per_mm, 0.0, 0.0],
                [0.0, -1.0 / result.paper.px_per_mm, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        frame_to_robot = (
            px_to_reflected_mm @ result.paper.frame_to_paper_px
        )
        angle = _robot_rotation(
            np.array([70.0, 70.0]),
            rigid_matrix(0.5 * np.pi, np.zeros(2)),
            result.paper,
            frame_to_robot,
        )
        self.assertAlmostEqual(angle, -90.0, delta=1e-5)

    def test_illumination_gradient_and_sensor_noise(self):
        rectified, _ = make_rectified_scene(self.config)
        frame = perspective_camera_frame(rectified).astype(np.float32)
        height, width = frame.shape[:2]
        x = np.linspace(0.78, 1.10, width, dtype=np.float32)
        y = np.linspace(0.96, 1.04, height, dtype=np.float32)
        illumination = y[:, None] * x[None, :]
        rng = np.random.default_rng(20260729)
        noise = rng.normal(0.0, 2.2, frame.shape).astype(np.float32)
        degraded = np.clip(
            frame * illumination[:, :, None] + noise, 0, 255
        ).astype(np.uint8)
        result = PuzzleVisionPipeline(self.config).process(
            degraded, use_texture=False
        )
        self.assertEqual(len(result.pieces), 4)
        self.assertGreater(result.solution.metrics.fill_ratio, 0.94)

    def test_detection_callback_receives_piece_contour_view(self):
        rectified, _ = make_rectified_scene(self.config)
        frame = perspective_camera_frame(rectified)
        frames = []
        pipeline = PuzzleVisionPipeline(self.config)
        result = pipeline.process(
            frame,
            use_texture=False,
            detection_callback=frames.append,
        )
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].shape, result.paper.image_bgr.shape)
        self.assertIsNotNone(pipeline.last_detection_debug_bgr)
        self.assertIsNotNone(
            pipeline.last_frame_detection_debug_bgr
        )
        self.assertEqual(
            pipeline.last_frame_detection_debug_bgr.shape,
            frame.shape,
        )
        self.assertGreater(
            int(np.count_nonzero(frames[0] != result.paper.image_bgr)),
            1000,
        )

    def test_solve_error_view_keeps_original_camera_frame(self):
        rectified, _ = make_rectified_scene(self.config, textured=True)
        frame = perspective_camera_frame(rectified, landscape=True)
        pipeline = PuzzleVisionPipeline(self.config)
        forced_error = SolveError("forced solve failure")
        with patch(
            "puzzle_vision.pipeline.solve_puzzle",
            side_effect=forced_error,
        ):
            with self.assertRaises(SolveError):
                pipeline.process(frame, use_texture=True)
        error_view = _make_error_view(
            forced_error, self.config, pipeline, frame
        )
        self.assertEqual(error_view.shape, frame.shape)
        self.assertEqual(
            pipeline.last_frame_detection_debug_bgr.shape,
            frame.shape,
        )

    def test_white_card_pieces_use_chroma_under_uneven_light(self):
        rectified, _ = make_rectified_scene(
            self.config,
            textured=True,
            paper_bgr=(205, 230, 205),
        )
        height, width = rectified.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        shadow = 0.28 * np.exp(
            -0.5
            * (
                ((xx - 0.30 * width) / (0.23 * width)) ** 2
                + ((yy - 0.22 * height) / (0.18 * height)) ** 2
            )
        )
        lab = cv2.cvtColor(rectified, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(
            lab[:, :, 0] - 70.0 * shadow, 0, 255
        )
        uneven = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        paper = PaperObservation(
            image_bgr=uneven,
            frame_quad_px=np.array(
                [
                    [0.0, 0.0],
                    [width - 1.0, 0.0],
                    [width - 1.0, height - 1.0],
                    [0.0, height - 1.0],
                ]
            ),
            frame_to_paper_px=np.eye(3),
            paper_px_to_frame=np.eye(3),
            divider_y_px=0.5 * (height - 1),
            paper_color_lab=np.zeros(3),
            detection_score=1.0,
            px_per_mm=self.config.px_per_mm,
        )
        pieces, _, _ = segment_pieces(paper, self.config)
        self.assertEqual(len(pieces), 4)
        self.assertGreater(sum(piece.area_mm2 for piece in pieces), 5000.0)

    def test_white_fragments_select_validated_fast_mask(self):
        rectified, _ = make_rectified_scene(
            self.config, textured=True
        )
        paper = detect_paper(
            perspective_camera_frame(rectified), self.config
        )
        lab = cv2.cvtColor(
            paper.image_bgr, cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        fast = _fast_white_piece_mask(lab, paper, self.config)
        self.assertIsNotNone(fast)
        pieces, _, _ = segment_pieces(paper, self.config)
        self.assertEqual(len(pieces), 4)

    def test_all_notched_equal_card_strips_reconstruct_white_boundaries(self):
        scale = self.config.px_per_mm
        width = int(round(self.config.paper_short_mm * scale))
        height = int(round(self.config.paper_long_mm * scale))
        image = np.full((height, width, 3), (170, 105, 45), dtype=np.uint8)
        divider_y = height // 2
        cv2.line(
            image,
            (0, divider_y),
            (width - 1, divider_y),
            (20, 20, 20),
            6,
        )
        placements = (
            ((45.0, 38.0), 12.0),
            ((150.0, 39.0), -15.0),
            ((52.0, 100.0), -8.0),
            ((151.0, 98.0), 18.0),
        )
        for center_mm, angle_deg in placements:
            center_px = np.asarray(center_mm) * scale
            rectangle = (
                tuple(center_px),
                (60.0 * scale, 20.0 * scale),
                angle_deg,
            )
            cv2.fillConvexPoly(
                image,
                np.round(cv2.boxPoints(rectangle)).astype(np.int32),
                (245, 245, 245),
            )
            angle = np.radians(angle_deg)
            rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
            notch_local_mm = np.array(
                [[-10.0, -10.0], [10.0, -10.0], [0.0, -2.0]]
            )
            notch_px = (
                notch_local_mm @ rotation.T
                + np.asarray(center_mm)
            ) * scale
            cv2.fillConvexPoly(
                image,
                np.round(notch_px).astype(np.int32),
                (20, 20, 20),
            )

        paper = PaperObservation(
            image_bgr=image,
            frame_quad_px=np.array(
                [
                    [0.0, 0.0],
                    [width - 1.0, 0.0],
                    [width - 1.0, height - 1.0],
                    [0.0, height - 1.0],
                ]
            ),
            frame_to_paper_px=np.eye(3),
            paper_px_to_frame=np.eye(3),
            divider_y_px=float(divider_y),
            paper_color_lab=np.zeros(3),
            detection_score=1.0,
            px_per_mm=scale,
        )
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        fast = _fast_white_piece_mask(lab, paper, self.config)

        self.assertIsNotNone(fast)
        self.assertTrue(fast[2])
        pieces, _, _ = segment_pieces(paper, self.config)
        self.assertEqual(len(pieces), 4)
        self.assertTrue(
            all(piece.boundary_reconstructed for piece in pieces)
        )
        self.assertEqual(
            [len(piece.polygon_mm) for piece in pieces],
            [4, 4, 4, 4],
        )

    def test_rejected_piece_sized_white_contour_uses_full_colour_mask(self):
        config = replace(self.config, expected_piece_count=0)
        scale = config.px_per_mm
        width = int(round(config.paper_short_mm * scale))
        height = int(round(config.paper_long_mm * scale))
        image = np.full((height, width, 3), (170, 105, 45), dtype=np.uint8)
        divider_y = height // 2
        cv2.line(
            image,
            (0, divider_y),
            (width - 1, divider_y),
            (20, 20, 20),
            6,
        )

        def fill_mm(points, color):
            cv2.fillPoly(
                image,
                [
                    np.round(
                        np.asarray(points, dtype=np.float64) * scale
                    ).astype(np.int32)
                ],
                color,
            )

        fill_mm(
            [[15, 40], [65, 52], [50, 105], [15, 95]],
            (245, 245, 245),
        )
        fill_mm(
            [[75, 45], [140, 45], [140, 85], [75, 85]],
            (245, 245, 245),
        )
        fill_mm(
            [[150, 42], [198, 60], [180, 118], [142, 100]],
            (245, 245, 245),
        )
        for x in (82, 102, 122):
            fill_mm(
                [[x, 45], [x + 14, 45], [x + 7, 62]],
                (20, 20, 20),
            )
        for x in (84, 104, 124):
            fill_mm(
                [[x, 85], [x + 14, 85], [x + 7, 68]],
                (20, 20, 20),
            )

        paper = PaperObservation(
            image_bgr=image,
            frame_quad_px=np.array(
                [
                    [0.0, 0.0],
                    [width - 1.0, 0.0],
                    [width - 1.0, height - 1.0],
                    [0.0, height - 1.0],
                ]
            ),
            frame_to_paper_px=np.eye(3),
            paper_px_to_frame=np.eye(3),
            divider_y_px=float(divider_y),
            paper_color_lab=np.zeros(3),
            detection_score=1.0,
            px_per_mm=scale,
        )
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        self.assertIsNone(
            _fast_white_piece_mask(lab, paper, config)
        )
        pieces, _, _ = segment_pieces(paper, config)
        self.assertEqual(len(pieces), 3)

    def test_colored_fragments_fall_back_from_white_mask(self):
        rectified, observations = make_rectified_scene(self.config)
        for observation in observations:
            rectified[observation.mask != 0] = (40, 220, 40)
        paper = detect_paper(
            perspective_camera_frame(rectified), self.config
        )
        lab = cv2.cvtColor(
            paper.image_bgr, cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        self.assertIsNone(
            _fast_white_piece_mask(lab, paper, self.config)
        )
        pieces, _, _ = segment_pieces(paper, self.config)
        self.assertEqual(len(pieces), 4)

    def test_texture_scoring_path(self):
        rectified, _ = make_rectified_scene(
            self.config, textured=True
        )
        frame = perspective_camera_frame(rectified)
        result = PuzzleVisionPipeline(self.config).process(
            frame, use_texture=True
        )
        self.assertTrue(np.isfinite(result.solution.texture_score))
        self.assertEqual(len(result.commands), 4)

    def test_texture_mode_is_selected_from_piece_content(self):
        plain_image, plain_pieces = make_rectified_scene(
            self.config, textured=False
        )
        card_image, card_pieces = make_rectified_scene(
            self.config, textured=True
        )
        self.assertFalse(
            _has_printed_texture(
                plain_image, plain_pieces, self.config.px_per_mm
            )
        )
        smooth_shadow = plain_image.copy()
        for piece in plain_pieces:
            y, x = np.nonzero(piece.mask)
            normalized_x = (
                x - np.min(x)
            ) / max(float(np.ptp(x)), 1.0)
            value = np.round(
                90.0 + 150.0 * normalized_x
            ).astype(np.uint8)
            smooth_shadow[y, x] = value[:, None]
        self.assertFalse(
            _has_printed_texture(
                smooth_shadow,
                plain_pieces,
                self.config.px_per_mm,
            )
        )
        self.assertTrue(
            _has_printed_texture(
                card_image, card_pieces, self.config.px_per_mm
            )
        )

    def test_texture_selects_the_continuous_card_seam(self):
        config = VisionConfig(ambiguity_margin=0.0)
        paper, pieces, source_matrices = make_ambiguous_textured_pair(
            config
        )
        solution = solve_puzzle(pieces, paper, config, use_texture=True)
        seam = np.array([[50.0, 0.0], [50.0, 60.0]])
        seam_a = (
            solution.transforms[0]
            @ source_matrices[0]
            @ np.column_stack([seam, np.ones(2)]).T
        ).T[:, :2]
        seam_b = (
            solution.transforms[1]
            @ source_matrices[1]
            @ np.column_stack([seam, np.ones(2)]).T
        ).T[:, :2]
        direct = np.max(np.linalg.norm(seam_a - seam_b, axis=1))
        reversed_order = np.max(
            np.linalg.norm(seam_a - seam_b[::-1], axis=1)
        )
        self.assertLess(min(direct, reversed_order), 1.0)
        self.assertLess(solution.texture_score, 0.0)

    def test_equal_card_quarters_use_face_features_not_only_rectangle(self):
        config = VisionConfig(
            solver_mode="generic",
            solver_max_nodes=180000,
        )
        paper, pieces, source_matrices = make_quartered_textured_card(
            config,
            piece_id_order=(2, 0, 3, 1),
        )
        solution = solve_puzzle(
            pieces, paper, config, use_texture=True
        )
        card_corners = np.array(
            [[0.0, 0.0], [60.0, 0.0], [60.0, 90.0], [0.0, 90.0]]
        )
        homogeneous = np.column_stack(
            [card_corners, np.ones(len(card_corners))]
        ).T
        assembled = [
            (
                solution.transforms[piece.piece_id]
                @ source_matrices[piece.piece_id]
                @ homogeneous
            ).T[:, :2]
            for piece in pieces
        ]
        maximum_disagreement = max(
            float(np.max(np.linalg.norm(points - assembled[0], axis=1)))
            for points in assembled[1:]
        )
        self.assertLess(maximum_disagreement, 1.0)
        self.assertGreater(solution.valid_candidate_count, 1)
        self.assertGreater(
            solution.score_margin, config.ambiguity_margin
        )

    def test_equal_card_quarters_reject_visually_indistinguishable_layouts(
        self,
    ):
        config = VisionConfig(solver_mode="generic")
        paper, pieces, _ = make_quartered_textured_card(
            config, center_mark=False
        )
        solution = solve_puzzle(
            pieces, paper, config, use_texture=True
        )
        self.assertIsNotNone(solution.score_margin)
        self.assertLess(
            solution.score_margin, config.ambiguity_margin
        )

    def test_unequal_card_strips_use_whole_face_features(self):
        config = VisionConfig(
            solver_mode="generic",
            solver_max_nodes=180000,
        )
        paper, pieces, source_matrices = make_quartered_textured_card(
            config,
            piece_id_order=(2, 0, 3, 1),
            strip_heights_mm=(20.0, 25.0, 18.0, 27.0),
        )
        self.assertFalse(
            solver_module._is_equal_card_quarters(pieces)
        )
        self.assertTrue(_is_card_strip_partition(pieces))
        solution = solve_puzzle(
            pieces, paper, config, use_texture=True
        )
        card_corners = np.array(
            [[0.0, 0.0], [60.0, 0.0], [60.0, 90.0], [0.0, 90.0]]
        )
        homogeneous = np.column_stack(
            [card_corners, np.ones(len(card_corners))]
        ).T
        assembled = [
            (
                solution.transforms[piece.piece_id]
                @ source_matrices[piece.piece_id]
                @ homogeneous
            ).T[:, :2]
            for piece in pieces
        ]
        maximum_disagreement = max(
            float(np.max(np.linalg.norm(points - assembled[0], axis=1)))
            for points in assembled[1:]
        )
        self.assertLess(maximum_disagreement, 1.0)
        self.assertGreater(
            solution.score_margin, config.ambiguity_margin
        )


class RuntimeDisplayTests(unittest.TestCase):
    def test_deployment_calibration_uses_top_left_origin_and_right_down_axes(
        self,
    ):
        config_data = json.loads(
            (PROJECT_DIR / "config.json").read_text(encoding="utf-8")
        )
        matrix = np.asarray(config_data["frame_to_robot"], dtype=np.float64)
        frame_corners = np.array(
            [
                [377.0251, 35.5374],
                [1118.5522, 34.9286],
                [1097.6113, 552.2358],
                [388.8529, 548.9881],
            ],
            dtype=np.float64,
        )
        expected_pulses = np.array(
            [
                [0.0, 0.0],
                [24000.0, 0.0],
                [24000.0, 16700.0],
                [0.0, 16700.0],
            ],
            dtype=np.float64,
        )

        actual_mm = apply_homography(frame_corners, matrix)
        actual_pulses = (
            actual_mm * float(config_data["robot_xy_pulses_per_mm"])
        )

        np.testing.assert_allclose(actual_pulses, expected_pulses, atol=1e-3)
        grid_centers = np.array(
            [
                [[525.985, 141.664], [671.605, 142.138],
                 [819.316, 141.705], [966.190, 141.389]],
                [[527.014, 245.716], [672.283, 246.386],
                 [817.470, 246.392], [962.971, 246.103]],
                [[528.249, 347.430], [671.268, 348.571],
                 [815.715, 348.424], [959.783, 348.599]],
                [[529.563, 448.088], [670.489, 449.138],
                 [814.436, 448.965], [956.957, 449.096]],
            ],
            dtype=np.float64,
        )
        expected_grid_mm = np.stack(
            np.meshgrid(
                [60.0, 120.0, 180.0, 240.0],
                [41.75, 83.5, 125.25, 167.0],
            ),
            axis=-1,
        )
        actual_grid_mm = apply_homography(
            grid_centers.reshape(-1, 2), matrix
        ).reshape(4, 4, 2)
        np.testing.assert_allclose(
            actual_grid_mm, expected_grid_mm, atol=1.25
        )
        self.assertEqual(config_data["robot_theta_sign"], 1.0)
        self.assertEqual(config_data["robot_x_max_pulse"], 24000)
        self.assertEqual(config_data["robot_y_max_pulse"], 16700)
        self.assertEqual(config_data["placement_spread_mm"], 5.0)

    def test_uart2_uses_u2t_u2r_header_and_ttys2(self):
        pin_calls = []
        uart_calls = []
        serial_object = object()

        fake_maix = types.ModuleType("maix")
        fake_maix.err = types.SimpleNamespace(
            check_raise=lambda result, message: self.assertEqual(
                result, 0, message
            )
        )
        fake_maix.pinmap = types.SimpleNamespace(
            set_pin_function=lambda pin, function: (
                pin_calls.append((pin, function)) or 0
            )
        )
        fake_maix.uart = types.SimpleNamespace(
            UART=lambda device, baudrate: (
                uart_calls.append((device, baudrate)) or serial_object
            )
        )

        config = VisionConfig(
            uart_enabled=True,
            robot_calibrated=True,
        )
        config.validate()
        with patch.dict(sys.modules, {"maix": fake_maix}):
            serial = _make_uart(config)

        self.assertIs(serial, serial_object)
        self.assertEqual(
            pin_calls,
            [
                ("B0", "UART2_TX"),
                ("B1", "UART2_RX"),
            ],
        )
        self.assertEqual(uart_calls, [("/dev/ttyS2", 115200)])

        with self.assertRaisesRegex(ValueError, "/dev/ttyS2"):
            VisionConfig(
                uart_enabled=True,
                robot_calibrated=True,
                uart_device="/dev/ttyS4",
            ).validate()
        with self.assertRaisesRegex(
            ValueError,
            "robot_xy_pulses_per_mm",
        ):
            VisionConfig(robot_xy_pulses_per_mm=0.0).validate()
        with self.assertRaisesRegex(ValueError, "pulse limits"):
            VisionConfig(robot_x_max_pulse=0).validate()
        with self.assertRaisesRegex(ValueError, "placement_spread_mm"):
            VisionConfig(placement_spread_mm=-0.1).validate()
        with self.assertRaisesRegex(ValueError, "fixed_paper_quad_px"):
            VisionConfig(
                fixed_paper_quad_px=[[0.0, 0.0], [1.0, 0.0]]
            ).validate()
        with self.assertRaisesRegex(
            ValueError, "solver_total_time_limit_ms"
        ):
            VisionConfig(solver_total_time_limit_ms=0).validate()
        with self.assertRaisesRegex(
            ValueError, "must not exceed"
        ):
            VisionConfig(
                solver_time_limit_ms=30001,
                solver_total_time_limit_ms=30000,
            ).validate()

    def test_uart2_smoke_script_sends_non_motion_test_frame(self):
        pin_calls = []
        sleep_calls = []

        class FakeSerial:
            def __init__(self):
                self.messages = []
                self.closed = False

            def write_str(self, message):
                self.messages.append(message)
                return len(message)

            def close(self):
                self.closed = True

        serial = FakeSerial()
        exit_states = iter((False, True))
        fake_maix = types.ModuleType("maix")
        fake_maix.app = types.SimpleNamespace(
            need_exit=lambda: next(exit_states)
        )
        fake_maix.err = types.SimpleNamespace(
            check_raise=lambda result, message: self.assertEqual(
                result, 0, message
            )
        )
        fake_maix.pinmap = types.SimpleNamespace(
            set_pin_function=lambda pin, function: (
                pin_calls.append((pin, function)) or 0
            )
        )
        fake_maix.time = types.SimpleNamespace(
            sleep_ms=sleep_calls.append
        )
        fake_maix.uart = types.SimpleNamespace(
            UART=lambda device, baudrate: serial
        )

        with patch.dict(sys.modules, {"maix": fake_maix}):
            self.assertEqual(uart_smoke_test.main(), 0)

        self.assertEqual(
            pin_calls,
            [
                ("B0", "UART2_TX"),
                ("B1", "UART2_RX"),
            ],
        )
        self.assertEqual(sleep_calls, [1000])
        self.assertTrue(serial.closed)
        self.assertEqual(len(serial.messages), 1)
        self.assertTrue(serial.messages[0].endswith("\n"))
        payload = json.loads(serial.messages[0])
        self.assertEqual(payload["status"], "UART_TEST")
        self.assertFalse(payload["robot_calibrated"])
        self.assertEqual(payload["commands"][0]["piece_id"], 0)
        self.assertEqual(payload["commands"][0]["pickup_x_pulse"], 800)

    def test_uart_plan_uses_minimal_robot_fields_and_jsonl_framing(self):
        command = MoveCommand(
            piece_id=2,
            source_center_paper_mm=np.array([1.0, 2.0]),
            pickup_paper_mm=np.array([3.0, 4.0]),
            target_center_paper_mm=np.array([5.0, 6.0]),
            place_paper_mm=np.array([7.0, 8.0]),
            rotation_cw_deg=9.0,
            pickup_robot_mm=np.array([120.4, 86.2]),
            place_robot_mm=np.array([156.8, 200.0]),
            rotation_robot_deg=31.5,
            target_polygon_paper_mm=np.zeros((4, 2)),
        )

        class Result:
            status = "OK"
            robot_calibrated = True
            commands = [command]

        config = VisionConfig()
        payload = _make_uart_payload(Result(), 5230, config)
        self.assertEqual(
            payload,
            {
                "status": "OK",
                "robot_calibrated": True,
                "elapsed_ms": 5230,
                "commands": [
                    {
                        "piece_id": 2,
                        "pickup_x_pulse": 9632,
                        "pickup_y_pulse": 6896,
                        "place_x_pulse": 12544,
                        "place_y_pulse": 16000,
                        "rotation_deg": 31.5,
                    }
                ],
            },
        )

        class FakeSerial:
            def __init__(self):
                self.message = None

            def write_str(self, message):
                self.message = message
                return len(message)

        serial = FakeSerial()
        _send_uart(serial, payload)
        self.assertEqual(
            serial.message,
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            + "\n",
        )

        command.rotation_robot_deg = -45.5
        clockwise_payload = _make_uart_payload(Result(), 5230, config)
        self.assertEqual(
            clockwise_payload["commands"][0]["rotation_deg"],
            314.5,
        )
        command.rotation_robot_deg = -1e-12
        zero_payload = _make_uart_payload(Result(), 5230, config)
        self.assertEqual(zero_payload["commands"][0]["rotation_deg"], 0.0)

        Result.status = "LOW_MARGIN"
        self.assertEqual(
            _make_uart_payload(Result(), 5230, config)["commands"],
            [],
        )

        Result.status = "OK"
        command.place_robot_mm = np.array([301.0, 100.0])
        with self.assertRaisesRegex(ValueError, "outside 0..24000"):
            _make_uart_payload(Result(), 5230, config)

    def test_landscape_camera_frame_keeps_exact_shape_and_orientation(self):
        source = np.zeros((480, 720, 3), dtype=np.uint8)
        source[:, :120] = (0, 0, 255)
        result = _fit_landscape(source, 720, 480)
        self.assertEqual(result.shape, (480, 720, 3))
        np.testing.assert_array_equal(result, source)

    def test_portrait_solver_view_is_contained_in_landscape_canvas(self):
        source = np.full((891, 630, 3), 180, dtype=np.uint8)
        result = _fit_landscape(source, 1280, 720)
        self.assertEqual(result.shape, (720, 1280, 3))
        self.assertTrue(np.all(result[:, 0] == 0))
        self.assertGreater(np.count_nonzero(result), 0)

    def test_touch_buttons_hit_after_display_letterboxing(self):
        canvas_width, canvas_height = 1280, 720
        display_width, display_height = 480, 640
        reader = _TouchButtonReader(
            display_width,
            display_height,
            canvas_width,
            canvas_height,
        )
        left, top, right, bottom = _button_rectangles(
            canvas_width, canvas_height
        )["start"]
        canvas_x = (left + right) * 0.5
        canvas_y = (top + bottom) * 0.5
        scale = min(
            display_width / canvas_height,
            display_height / canvas_width,
        )
        offset_x = (display_width - canvas_height * scale) * 0.5
        offset_y = (display_height - canvas_width * scale) * 0.5
        touch_x = int(round(offset_x + canvas_y * scale))
        touch_y = int(
            round(
                offset_y
                + (canvas_width - 1.0 - canvas_x) * scale
            )
        )
        self.assertIsNone(reader.feed(touch_x, touch_y, True))
        self.assertEqual(reader.feed(touch_x, touch_y, False), "start")

    def test_landscape_ui_is_pre_rotated_for_portrait_panel(self):
        class FakeImageModule:
            received = None

            @classmethod
            def cv2image(cls, image, *, bgr, copy):
                cls.received = image
                return image

        class PortraitDisplay:
            shown = None

            @staticmethod
            def width():
                return 480

            @staticmethod
            def height():
                return 640

            def show(self, image):
                self.shown = image

        source = np.zeros((480, 720, 3), dtype=np.uint8)
        source[20:80, 30:130] = (0, 0, 255)
        display = PortraitDisplay()
        _display_bgr(display, FakeImageModule, source)
        expected = cv2.rotate(
            source, cv2.ROTATE_90_COUNTERCLOCKWISE
        )
        self.assertEqual(FakeImageModule.received.shape, (720, 480, 3))
        np.testing.assert_array_equal(
            FakeImageModule.received, expected
        )

    def test_touch_ui_is_always_landscape(self):
        portrait = np.zeros((891, 630, 3), dtype=np.uint8)
        ui = _compose_touch_ui(
            portrait, 480, 720, "READY - tap START"
        )
        self.assertEqual(ui.shape, (480, 720, 3))

    def test_odd_debug_image_is_padded_before_maix_conversion(self):
        class FakeImageModule:
            received = None

            @classmethod
            def cv2image(cls, image, *, bgr, copy):
                cls.received = image
                self.assertTrue(bgr)
                self.assertTrue(copy)
                return image

        class FakeDisplay:
            shown = None

            def show(self, image):
                self.shown = image

        display = FakeDisplay()
        source = np.zeros((891, 630, 3), dtype=np.uint8)
        _display_bgr(display, FakeImageModule, source)
        self.assertEqual(FakeImageModule.received.shape, (892, 630, 3))
        self.assertIs(display.shown, FakeImageModule.received)


if __name__ == "__main__":
    unittest.main()
