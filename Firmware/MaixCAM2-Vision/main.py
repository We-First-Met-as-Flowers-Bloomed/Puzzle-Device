"""MaixCAM2 touch-screen entry point for the E-problem puzzle vision pipeline."""

from __future__ import annotations

import json
import os
import sys
import threading
import time as py_time
import traceback
import warnings

import cv2
import numpy as np

from puzzle_vision import PuzzleVisionPipeline, load_config


warnings.filterwarnings(
    "ignore",
    message=r"The value of the smallest subnormal for .* type is zero\.",
    category=UserWarning,
    module=r"numpy\._core\.getlimits",
)

_BUTTON_SPECS = (
    ("start", "START", (42, 150, 55)),
    ("stop", "STOP", (45, 65, 210)),
    ("reset", "RESET", (190, 105, 35)),
)


def _sharpness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _fit_landscape(
    frame_bgr: np.ndarray,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Contain an image in an exact landscape canvas without rotating it."""
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output dimensions must be positive")
    if output_width < output_height:
        output_width, output_height = output_height, output_width
    source_height, source_width = frame_bgr.shape[:2]
    scale = min(
        output_width / float(source_width),
        output_height / float(source_height),
    )
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    if (
        resized_width == source_width
        and resized_height == source_height
    ):
        resized = frame_bgr
    else:
        interpolation = (
            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        )
        resized = cv2.resize(
            frame_bgr,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    x0 = (output_width - resized_width) // 2
    y0 = (output_height - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas


def _button_rectangles(width: int, height: int):
    margin = max(6, int(round(min(width, height) * 0.018)))
    gap = margin
    button_height = max(48, int(round(height * 0.15)))
    top = height - margin - button_height
    usable_width = width - 2 * margin - 2 * gap
    button_width = usable_width // 3
    rectangles = {}
    for index, (action, _label, _color) in enumerate(_BUTTON_SPECS):
        left = margin + index * (button_width + gap)
        right = (
            width - margin
            if index == len(_BUTTON_SPECS) - 1
            else left + button_width
        )
        rectangles[action] = (left, top, right, height - margin)
    return rectangles


def _compose_touch_ui(
    frame_bgr: np.ndarray,
    output_width: int,
    output_height: int,
    status: str,
) -> np.ndarray:
    canvas = _fit_landscape(frame_bgr, output_width, output_height)
    rectangles = _button_rectangles(canvas.shape[1], canvas.shape[0])

    overlay = canvas.copy()
    status_height = max(42, int(round(canvas.shape[0] * 0.085)))
    cv2.rectangle(
        overlay,
        (0, 0),
        (canvas.shape[1] - 1, status_height),
        (18, 18, 18),
        -1,
    )
    for action, _label, color in _BUTTON_SPECS:
        left, top, right, bottom = rectangles[action]
        cv2.rectangle(overlay, (left, top), (right, bottom), color, -1)
    canvas = cv2.addWeighted(overlay, 0.86, canvas, 0.14, 0.0)

    text_scale = max(0.55, min(1.35, canvas.shape[1] / 900.0))
    text_thickness = max(1, int(round(text_scale * 2.0)))
    cv2.putText(
        canvas,
        status,
        (18, int(round(status_height * 0.72))),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )
    for action, label, _color in _BUTTON_SPECS:
        left, top, right, bottom = rectangles[action]
        (text_width, text_height), _baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_thickness,
        )
        cv2.putText(
            canvas,
            label,
            (
                left + (right - left - text_width) // 2,
                top + (bottom - top + text_height) // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return canvas


def _map_touch_to_canvas(
    x: int,
    y: int,
    display_width: int,
    display_height: int,
    canvas_width: int,
    canvas_height: int,
):
    """Undo portrait-panel rotation and Display.show letterboxing."""
    rotate_ccw = (
        display_width < display_height
        and canvas_width > canvas_height
    )
    shown_canvas_width = (
        canvas_height if rotate_ccw else canvas_width
    )
    shown_canvas_height = (
        canvas_width if rotate_ccw else canvas_height
    )
    scale = min(
        display_width / float(shown_canvas_width),
        display_height / float(shown_canvas_height),
    )
    shown_width = shown_canvas_width * scale
    shown_height = shown_canvas_height * scale
    offset_x = (display_width - shown_width) * 0.5
    offset_y = (display_height - shown_height) * 0.5
    shown_x = (float(x) - offset_x) / scale
    shown_y = (float(y) - offset_y) / scale
    if not (
        0.0 <= shown_x < shown_canvas_width
        and 0.0 <= shown_y < shown_canvas_height
    ):
        return None
    if rotate_ccw:
        canvas_x = canvas_width - 1.0 - shown_y
        canvas_y = shown_x
    else:
        canvas_x = shown_x
        canvas_y = shown_y
    return int(canvas_x), int(canvas_y)


def _hit_test_button(x: int, y: int, width: int, height: int):
    for action, (left, top, right, bottom) in _button_rectangles(
        width, height
    ).items():
        if left <= x <= right and top <= y <= bottom:
            return action
    return None


class _TouchButtonReader:
    def __init__(
        self,
        display_width: int,
        display_height: int,
        canvas_width: int,
        canvas_height: int,
    ):
        self.display_width = display_width
        self.display_height = display_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self._pressed_action = None

    def feed(self, x: int, y: int, pressed: bool):
        point = _map_touch_to_canvas(
            x,
            y,
            self.display_width,
            self.display_height,
            self.canvas_width,
            self.canvas_height,
        )
        action = (
            None
            if point is None
            else _hit_test_button(
                point[0],
                point[1],
                self.canvas_width,
                self.canvas_height,
            )
        )
        if pressed:
            self._pressed_action = action
            return None
        released_action = (
            action if action == self._pressed_action else None
        )
        self._pressed_action = None
        return released_action

    def poll(self, touch):
        if not touch.available(0):
            return None
        x, y, pressed = touch.read()
        return self.feed(x, y, bool(pressed))


def _display_bgr(display_object, maix_image, frame_bgr: np.ndarray) -> None:
    try:
        display_width = int(display_object.width())
        display_height = int(display_object.height())
    except (AttributeError, TypeError):
        display_width = frame_bgr.shape[1]
        display_height = frame_bgr.shape[0]
    if (
        display_width < display_height
        and frame_bgr.shape[1] > frame_bgr.shape[0]
    ):
        # The MaixCAM2 panel reports portrait coordinates (480 x 640), while
        # the camera and UI are landscape.  The photographed device shows the
        # panel mounting rotates a landscape buffer clockwise, so pre-rotate
        # counter-clockwise here.  Touch mapping applies the exact inverse.
        frame_bgr = cv2.rotate(
            frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE
        )
    contiguous = np.ascontiguousarray(frame_bgr)
    height, width = contiguous.shape[:2]
    pad_bottom = height % 2
    pad_right = width % 2
    if pad_bottom or pad_right:
        # MaixCAM2 converts BGR888 to a YUV display buffer internally.  That
        # OpenCV path requires both dimensions to be even.  The rectified A4
        # debug image is normally 630 x 891, so replicate at most one edge
        # pixel instead of turning a successful vision run into a display
        # error.
        contiguous = cv2.copyMakeBorder(
            contiguous,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_REPLICATE,
        )
        contiguous = np.ascontiguousarray(contiguous)
    display_object.show(
        maix_image.cv2image(contiguous, bgr=True, copy=True)
    )


def _make_uart(config):
    if not config.uart_enabled:
        return None
    from maix import err, pinmap, uart

    err.check_raise(
        pinmap.set_pin_function("B0", "UART2_TX"),
        "set B0 UART2_TX failed",
    )
    err.check_raise(
        pinmap.set_pin_function("B1", "UART2_RX"),
        "set B1 UART2_RX failed",
    )
    return uart.UART(config.uart_device, config.uart_baudrate)


def _send_uart(serial, payload) -> None:
    if serial is None:
        return
    message = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
    )
    written = serial.write_str(message)
    if written != len(message):
        raise OSError(
            f"UART short write: sent {written} of {len(message)} bytes"
        )


def _make_uart_payload(result, elapsed_ms: int, config):
    commands = []
    if result.status == "OK":
        commands = [
            command.to_uart_dict(config.robot_xy_pulses_per_mm)
            for command in result.commands
        ]
        limits = {
            "pickup_x_pulse": config.robot_x_max_pulse,
            "place_x_pulse": config.robot_x_max_pulse,
            "pickup_y_pulse": config.robot_y_max_pulse,
            "place_y_pulse": config.robot_y_max_pulse,
        }
        for command in commands:
            for field, maximum in limits.items():
                if not 0 <= command[field] <= maximum:
                    raise ValueError(
                        f"piece {command['piece_id']} {field}="
                        f"{command[field]} outside 0..{maximum}"
                    )
    return {
        "status": result.status,
        "robot_calibrated": result.robot_calibrated,
        "elapsed_ms": elapsed_ms,
        "commands": commands,
    }


def _capture_best_frame(cam, maix_image, count: int) -> np.ndarray:
    frames = []
    for _ in range(max(1, count)):
        frame = cam.read()
        bgr = maix_image.image2cv(
            frame, ensure_bgr=False, copy=True
        )
        frames.append(np.ascontiguousarray(bgr))
    return max(frames, key=_sharpness)


def _wait_for_start(cam, disp, maix_app, maix_image, start_mode: str) -> None:
    if start_mode == "immediate":
        return

    from maix import key

    state = {"start": False}

    def on_key(_key_id, key_state):
        if key_state == key.State.KEY_PRESSED:
            print(f"start key pressed: id={_key_id}")
            state["start"] = True

    key_object = key.Key(on_key)
    try:
        while not maix_app.need_exit() and not state["start"]:
            frame = cam.read()
            preview = maix_image.image2cv(
                frame, ensure_bgr=False, copy=True
            )
            cv2.putText(
                preview,
                "PRESS FUNC TO START",
                (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            _display_bgr(disp, maix_image, preview)
    finally:
        # Key() removes MaixPy's default launcher-exit listener. Restore it
        # after this legacy start path.
        key_object.close()
        key.add_default_listener()


def _make_error_view(error, config, pipeline, frame_bgr):
    frame_detection_debug = (
        None
        if pipeline is None
        else pipeline.last_frame_detection_debug_bgr
    )
    detection_debug = (
        None if pipeline is None else pipeline.last_detection_debug_bgr
    )
    if frame_detection_debug is not None:
        error_image = frame_detection_debug.copy()
        height, width = error_image.shape[:2]
        panel_top = max(0, height - 190)
        panel_bottom = height - 1
    elif detection_debug is not None:
        error_image = detection_debug.copy()
        height, width = error_image.shape[:2]
        panel_top = max(0, height - 190)
        panel_bottom = height - 1
    elif frame_bgr is not None:
        error_image = frame_bgr.copy()
        height, width = error_image.shape[:2]
        panel_top = 0
        panel_bottom = min(height - 1, 190)
    else:
        width = config.camera_width
        height = config.camera_height
        error_image = np.zeros((height, width, 3), dtype=np.uint8)
        panel_top = 0
        panel_bottom = min(height - 1, 190)

    overlay = error_image.copy()
    cv2.rectangle(
        overlay,
        (0, panel_top),
        (width - 1, panel_bottom),
        (0, 0, 0),
        -1,
    )
    error_image = cv2.addWeighted(
        overlay, 0.78, error_image, 0.22, 0.0
    )
    cv2.putText(
        error_image,
        type(error).__name__,
        (24, panel_top + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    message = str(error)
    characters_per_line = max(32, int(width / 18))
    for index in range(0, len(message), characters_per_line):
        cv2.putText(
            error_image,
            message[index : index + characters_per_line],
            (
                24,
                panel_top + 92 + 28 * (index // characters_per_line),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return error_image


def _run_once(
    config,
    pipeline,
    cam,
    maix_image,
    debug_dir,
):
    input_debug_path = os.path.join(debug_dir, "puzzle_input.png")
    frame_bgr = None
    try:
        # Camera::clear_buff() is a no-op on MaixCAM2. Discard stale frames,
        # then select the sharpest fresh frame.
        cam.skip_frames(2)
        frame_bgr = _capture_best_frame(
            cam, maix_image, config.capture_frames
        )
        if config.save_debug_images:
            os.makedirs(debug_dir, exist_ok=True)
            if cv2.imwrite(input_debug_path, frame_bgr):
                print(f"saved input frame: {input_debug_path}")
        started = py_time.monotonic()
        pipeline.last_detection_debug_bgr = None
        pipeline.last_frame_detection_debug_bgr = None

        def save_detection(detection_bgr):
            if not config.save_debug_images:
                return
            detection_path = os.path.join(
                debug_dir, "puzzle_detected.jpg"
            )
            if cv2.imwrite(detection_path, detection_bgr):
                print(f"saved detection view: {detection_path}")
            frame_detection = (
                pipeline.last_frame_detection_debug_bgr
            )
            if frame_detection is not None:
                frame_detection_path = os.path.join(
                    debug_dir, "puzzle_detected_frame.jpg"
                )
                if cv2.imwrite(
                    frame_detection_path, frame_detection
                ):
                    print(
                        "saved frame detection view: "
                        f"{frame_detection_path}"
                    )

        result = pipeline.process(
            frame_bgr,
            use_texture=config.texture_enabled,
            detection_callback=save_detection,
        )
        elapsed_ms = int(round((py_time.monotonic() - started) * 1000.0))
        result_payload = result.to_dict()
        result_payload["elapsed_ms"] = elapsed_ms
        print(json.dumps(result_payload, ensure_ascii=False, indent=2))

        uart_payload = _make_uart_payload(result, elapsed_ms, config)
        landscape_result = _fit_landscape(
            result.debug_bgr,
            config.camera_width,
            config.camera_height,
        )
        if config.save_debug_images:
            cv2.imwrite(
                os.path.join(debug_dir, "puzzle_debug_raw.jpg"),
                result.rectified_debug_bgr,
            )
            cv2.imwrite(
                os.path.join(debug_dir, "puzzle_debug.jpg"),
                landscape_result,
            )
            cv2.imwrite(
                os.path.join(debug_dir, "puzzle_mask.png"),
                result.foreground_mask,
            )
        return (
            landscape_result,
            f"DONE: {result.status} ({elapsed_ms} ms)",
            uart_payload,
        )
    except Exception as error:
        error_payload = {
            "status": "ERROR",
            "type": type(error).__name__,
            "message": str(error),
        }
        if os.path.exists(input_debug_path):
            error_payload["input_image"] = input_debug_path
        print(json.dumps(error_payload, ensure_ascii=False))
        traceback.print_exc()
        error_image = _make_error_view(
            error, config, pipeline, frame_bgr
        )
        landscape_error = _fit_landscape(
            error_image,
            config.camera_width,
            config.camera_height,
        )
        if config.save_debug_images:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                error_debug_path = os.path.join(
                    debug_dir, "puzzle_error.jpg"
                )
                if cv2.imwrite(error_debug_path, landscape_error):
                    print(f"saved error view: {error_debug_path}")
            except Exception as save_error:
                print(
                    f"failed to save error view: {save_error}",
                    file=sys.stderr,
                )
        return (
            landscape_error,
            f"ERROR: {type(error).__name__}",
            error_payload,
        )


def main() -> int:
    try:
        from maix import app, camera, display, image, touchscreen
    except ImportError:
        print(
            "This entry point runs on MaixCAM2. For PC image replay use "
            "tools/run_image.py.",
            file=sys.stderr,
        )
        return 2

    config = None
    serial = None
    disp = None
    cam = None
    touch = None
    pipeline = None
    debug_dir = "/maixapp/tmp"
    try:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(project_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                "config.json is required for board-side execution"
            )
        config = load_config(config_path)
        limit_text = (
            f"{config.solver_total_time_limit_ms} ms total"
            if config.solver_time_limit_ms == 0
            else (
                f"{config.solver_time_limit_ms} ms/attempt, "
                f"{config.solver_total_time_limit_ms} ms total"
            )
        )
        print(
            "[vision] config "
            f"solver={config.solver_mode}, "
            f"texture={'on' if config.texture_enabled else 'off'}, "
            f"pieces={config.expected_piece_count or 'auto'}, "
            f"limit={limit_text}",
            flush=True,
        )
        pipeline = PuzzleVisionPipeline(config)

        disp = display.Display()
        display_width = disp.width()
        display_height = disp.height()
        cam = camera.Camera(
            width=config.camera_width,
            height=config.camera_height,
            format=image.Format.FMT_BGR888,
            fps=config.camera_fps,
            buff_num=3,
            open=True,
        )
        cam.skip_frames(10)
        serial = _make_uart(config)
        touch = touchscreen.TouchScreen()
        touch.clear()

        canvas_width = max(config.camera_width, config.camera_height)
        canvas_height = min(config.camera_width, config.camera_height)
        buttons = _TouchButtonReader(
            display_width,
            display_height,
            canvas_width,
            canvas_height,
        )
        print(
            "[ui] touch controls ready: START / STOP / RESET; "
            f"display={display_width}x{display_height}, "
            f"canvas={canvas_width}x{canvas_height}",
            flush=True,
        )

        current_view = None
        state = "READY - tap START"
        frozen = False
        ui_dirty = True
        worker_thread = None
        worker_result = {"value": None}
        worker_lock = threading.Lock()
        cancel_event = threading.Event()
        reset_after_worker = False
        pending_action = (
            "start" if config.start_mode == "immediate" else None
        )
        if config.start_mode == "func_key":
            _wait_for_start(
                cam, disp, app, image, config.start_mode
            )
            pending_action = "start"

        while not app.need_exit():
            if pending_action == "start":
                if worker_thread is None:
                    if current_view is None:
                        preview_frame = cam.read()
                        current_view = image.image2cv(
                            preview_frame, ensure_bgr=False, copy=True
                        )
                    frozen = True
                    state = "RUNNING - STOP cancels this result"
                    ui_dirty = True
                    cancel_event = threading.Event()
                    reset_after_worker = False
                    with worker_lock:
                        worker_result["value"] = None

                    def run_worker():
                        outcome = _run_once(
                            config,
                            pipeline,
                            cam,
                            image,
                            debug_dir,
                        )
                        with worker_lock:
                            worker_result["value"] = outcome

                    worker_thread = threading.Thread(
                        target=run_worker,
                        name="puzzle-solver",
                        daemon=True,
                    )
                    worker_thread.start()
                else:
                    state = "BUSY - STOP or wait"
                frozen = True
                ui_dirty = True
                pending_action = None
            elif pending_action == "stop":
                if worker_thread is not None:
                    cancel_event.set()
                    state = "STOPPED - cancelling solver result"
                else:
                    state = "STOPPED - tap START or RESET"
                reset_after_worker = False
                frozen = True
                ui_dirty = True
                pending_action = None
            elif pending_action == "reset":
                if worker_thread is not None:
                    cancel_event.set()
                    reset_after_worker = True
                    frozen = True
                    state = "RESET - waiting for solver to stop"
                else:
                    current_view = None
                    frozen = False
                    state = "READY - tap START"
                ui_dirty = True
                touch.clear()
                pending_action = None

            if (
                worker_thread is not None
                and not worker_thread.is_alive()
            ):
                worker_thread.join()
                with worker_lock:
                    outcome = worker_result["value"]
                    worker_result["value"] = None
                worker_thread = None
                if reset_after_worker:
                    current_view = None
                    frozen = False
                    state = "READY - tap START"
                    reset_after_worker = False
                elif cancel_event.is_set():
                    frozen = True
                    state = "STOPPED - result discarded"
                else:
                    current_view, state, uart_payload = outcome
                    _send_uart(serial, uart_payload)
                    frozen = True
                ui_dirty = True

            if not frozen:
                preview_frame = cam.read()
                current_view = image.image2cv(
                    preview_frame, ensure_bgr=False, copy=True
                )
            if not frozen or ui_dirty:
                ui_frame = _compose_touch_ui(
                    current_view,
                    canvas_width,
                    canvas_height,
                    state,
                )
                _display_bgr(disp, image, ui_frame)
                ui_dirty = False

            action = buttons.poll(touch)
            if action is not None:
                print(f"[ui] {action} touched", flush=True)
                pending_action = action
            if frozen:
                py_time.sleep(0.02)
        if worker_thread is not None:
            cancel_event.set()
        return 0
    except Exception as error:
        error_payload = {
            "status": "ERROR",
            "type": type(error).__name__,
            "message": str(error),
        }
        print(json.dumps(error_payload, ensure_ascii=False))
        traceback.print_exc()
        try:
            _send_uart(serial, error_payload)
        except Exception as uart_error:
            print(f"failed to send UART error frame: {uart_error}", file=sys.stderr)
        return 1
    finally:
        if touch is not None:
            try:
                touch.close()
            except Exception:
                pass
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass
        if serial is not None:
            try:
                serial.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
