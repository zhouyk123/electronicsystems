"""
Independent test for the forward_color() behavior with live output.

The loop mirrors metrics.forward_color(), but adds timeout, live printing, and
an optional safety stop when the target is lost.

Examples:
python3 test_forward_color.py red --timeout 20
python3 test_forward_color.py yellow --duty 18 --stop-area 70000
"""

import argparse
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


def draw_overlay(frame, mask, color, center_x, area):
    overlay = frame.copy()
    _, height = metrics.picSize
    width, _ = metrics.picSize
    cv2.line(overlay, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)
    if area > 0:
        x = int(center_x)
        cv2.line(overlay, (x, 0), (x, height), (0, 0, 255), 2)
        cv2.circle(overlay, (x, height // 2), 6, (0, 0, 255), -1)
        text = f"{color} center_x={center_x:.1f} area={area:.0f}"
    else:
        text = f"{color} center_x=None area=0"
    cv2.putText(overlay, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("forward_color", overlay)
    cv2.imshow("mask", mask)


def main():
    parser = argparse.ArgumentParser(description="Test forward_color behavior.")
    parser.add_argument("color", choices=["red", "yellow", "green"])
    parser.add_argument("--duty", type=float, default=metrics.FORWARD_DUTY)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--stop-area", type=float, default=metrics.AREA_FOR_TURN)
    parser.add_argument("--print-interval", type=float, default=0.2)
    parser.add_argument("--camera-wait", type=float, default=5.0)
    parser.add_argument(
        "--keep-moving-when-lost",
        action="store_true",
        help="Match metrics.forward_color() exactly when the target disappears.",
    )
    args = parser.parse_args()

    try:
        metrics.FORWARD_DUTY = args.duty
        metrics.AREA_FOR_TURN = args.stop_area
        metrics.pidController = metrics.WheelSpeedPID(**metrics.FORWARD_PID, target=0, lb=-8, ub=8)

        metrics.init()
        metrics.getPic = 1
        metrics.picGet.start()
        if not wait_for_frame(args.camera_wait):
            raise RuntimeError("camera did not produce a frame")

        start = time.time()
        last_print = 0.0
        while time.time() - start < args.timeout:
            frame = metrics.frame.astype(np.uint8)
            mask = metrics.getImg_Mask(frame, args.color)
            center_x, area, contours = metrics.get_Cube_center_area(mask)

            if area > 0:
                metrics.forward(center_x)
            elif not args.keep_moving_when_lost:
                metrics.stop(0)

            now = time.time()
            if now - last_print >= args.print_interval:
                delta = metrics.pidController.u
                print(
                    "t={:.2f}s center_x={} area={:.0f} pid_delta={:.2f} lspeed={:.3f} rspeed={:.3f}".format(
                        now - start,
                        f"{center_x:.1f}" if area > 0 else "None",
                        area,
                        delta,
                        metrics.lspeed,
                        metrics.rspeed,
                    )
                )
                last_print = now

            draw_overlay(frame, mask, args.color, center_x, area)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if area > args.stop_area:
                print(f"[done] area {area:.0f} exceeded stop area {args.stop_area:.0f}")
                break

        metrics.stop()
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
