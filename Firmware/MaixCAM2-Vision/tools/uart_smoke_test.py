"""Continuously send a non-motion JSON test frame over MaixCAM2 UART2."""

from __future__ import annotations

import json


_PAYLOAD = {
    "status": "UART_TEST",
    "robot_calibrated": False,
    "elapsed_ms": 0,
    "commands": [
        {
            "piece_id": 0,
            "pickup_x_pulse": 800,
            "pickup_y_pulse": 1600,
            "place_x_pulse": 2400,
            "place_y_pulse": 3200,
            "rotation_deg": 15.0,
        }
    ],
}


def main() -> int:
    from maix import app, err, pinmap, time, uart

    err.check_raise(
        pinmap.set_pin_function("B0", "UART2_TX"),
        "set B0 UART2_TX failed",
    )
    err.check_raise(
        pinmap.set_pin_function("B1", "UART2_RX"),
        "set B1 UART2_RX failed",
    )
    serial = uart.UART("/dev/ttyS2", 115200)
    message = (
        json.dumps(_PAYLOAD, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    )
    try:
        while not app.need_exit():
            written = serial.write_str(message)
            if written != len(message):
                raise OSError(
                    f"UART short write: sent {written} of "
                    f"{len(message)} bytes"
                )
            print(message, end="", flush=True)
            time.sleep_ms(1000)
        return 0
    finally:
        serial.close()


if __name__ == "__main__":
    raise SystemExit(main())
