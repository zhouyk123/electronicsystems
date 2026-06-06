"""
Independent test for detected_color().

This wraps turn_right() with a timeout so a bad encoder signal does not make
the test run forever.

Examples:
python3 test_detect_color.py red
python3 test_detect_color.py green --scan-angle 15 --turn-timeout 3
"""

import argparse
import threading
import time

import cv2
import numpy as np
import RPi.GPIO as GPIO

import metrics


def wait_for_frame(timeout):
    start = time.time()
    while time.time() - start < timeout:
        if isinstance(metrics.frame, np.ndarray):
            return True
        time.sleep(0.05)
    return False


def main():
    parser = argparse.ArgumentParser(description="Test detected_color() search behavior.")
    parser.add_argument("color", choices=["red", "yellow", "green"])
    parser.add_argument("--scan-duty", type=float, default=metrics.TURN_DUTY)
    parser.add_argument("--scan-angle", type=float, default=15.0)
    parser.add_argument("--turn-timeout", type=float, default=3.0)
    parser.add_argument("--max-turn-count", type=int, default=metrics.MAX_TURN_COUNT)
    parser.add_argument("--rest-time", type=float, default=metrics.FIND_TURN_REST_TIME)
    parser.add_argument("--wait-key", type=int, default=1)
    parser.add_argument("--camera-wait", type=float, default=5.0)
    args = parser.parse_args()

    original_turn_right = metrics.turn_right

    def safe_turn_right(duty=metrics.TURN_DUTY, angle=None):
        actual_duty = args.scan_duty if duty == metrics.TURN_DUTY else duty
        actual_angle = args.scan_angle if angle is None else angle
        error = []

        def runner():
            try:
                original_turn_right(actual_duty, actual_angle)
            except BaseException as exc:
                error.append(exc)

        worker = threading.Thread(target=runner, daemon=True)
        worker.start()
        worker.join(args.turn_timeout)
        if worker.is_alive():
            print("[turn timeout] forcing stop")
            metrics.force_move_done()
            metrics.brake(0.1)
            worker.join(1.0)
        if error:
            raise error[0]

    try:
        metrics.init()
        metrics.MAX_TURN_COUNT = args.max_turn_count
        metrics.FIND_TURN_REST_TIME = args.rest_time
        metrics.WAIT_EVERY_TURN = args.wait_key
        metrics.turn_right = safe_turn_right

        metrics.getPic = 1
        metrics.picGet.start()
        if not wait_for_frame(args.camera_wait):
            raise RuntimeError("camera did not produce a frame")

        metrics.detected_color(args.color)
    finally:
        try:
            metrics.getPic = 0
            metrics.getSpeed = 0
            metrics.stop(0)
        except Exception:
            pass
        metrics.cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
