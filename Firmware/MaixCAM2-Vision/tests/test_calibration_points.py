import unittest

from tools.collect_calibration_points import (
    display_to_frame,
    make_payload,
    pulse_to_mm,
)


class CalibrationPointToolTests(unittest.TestCase):
    def test_display_coordinates_map_back_to_original_frame(self):
        self.assertEqual(
            display_to_frame(320, 180, 0.5, 1280, 720),
            (640, 360),
        )
        self.assertEqual(
            display_to_frame(9999, -10, 0.5, 1280, 720),
            (1279, 0),
        )

    def test_pulse_position_converts_to_signed_millimetres(self):
        self.assertAlmostEqual(
            pulse_to_mm(8400, 400, 80, 1),
            100.0,
        )
        self.assertAlmostEqual(
            pulse_to_mm(8400, 400, 80, -1),
            -100.0,
        )

    def test_payload_is_accepted_by_existing_fitter_shape(self):
        payload = make_payload(
            [(100, 200), (300, 200), (300, 400), (100, 400)],
            [(0.0, 0.0), (50.0, 0.0), (50.0, 70.0), (0.0, 70.0)],
        )
        self.assertEqual(len(payload["frame_px"]), 4)
        self.assertEqual(len(payload["robot_mm"]), 4)
        self.assertEqual(payload["frame_px"][0], [100, 200])
        self.assertEqual(payload["robot_mm"][2], [50.0, 70.0])


if __name__ == "__main__":
    unittest.main()
