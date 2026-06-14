"""
Independent test for centor_color() behavior with live output.

The loop mirrors metrics.centor_color(), but adds timeout, live printing, and
visual overlays for tuning CENTER_PID, CENTER_TOLERANCE, and CENTER_PID_LIMIT.

Examples:
python3 test_center_color.py red --timeout 10
python3 test_center_color.py yellow --error 8 --timeout 6
python3 test_center_color.py green --pid-p 0.05 --limit 30
"""

import argparse
import time

import cv2
import numpy as np

import metrics


def wait_for_frame(timeout):
    start = time.time()
    while time.time() - start < timeout:
        if isinstance(metrics.frame, np.ndarray):
            return True
        time.sleep(0.05)
    return False


def clamp_duty(value):
    return max(0, min(100, value))


def turn_with_center_delta(delta):
    metrics.set_turn_pid_mode(None)
    metrics.set_motor_mode(metrics.left_pin, metrics.MODE_L)
    metrics.set_motor_mode(metrics.right_pin, metrics.MODE_R)
    left = clamp_duty(0 - delta)
    right = clamp_duty(0 + delta)
    metrics.set_motor_duty(left, right)
    return left, right


def draw_overlay(frame, mask, color, center_x, area, err, target_error):
    overlay = frame.copy()
    _, height = metrics.picSize
    width, _ = metrics.picSize
    middle = width // 2
    cv2.line(overlay, (middle, 0), (middle, height), (255, 0, 0), 1)
    cv2.line(overlay, (middle - int(target_error), 0), (middle - int(target_error), height), (255, 255, 0), 1)
    cv2.line(overlay, (middle + int(target_error), 0), (middle + int(target_error), height), (255, 255, 0), 1)

    if area > 0:
        x = int(center_x)
        cv2.line(overlay, (x, 0), (x, height), (0, 0, 255), 2)
        cv2.circle(overlay, (x, height // 2), 6, (0, 0, 255), -1)
        text = f"{color} center_x={center_x:.1f} err={err:.1f} area={area:.0f}"
    else:
        text = f"{color} center_x=None area=0"

    cv2.putText(overlay, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("center_color", overlay)
    cv2.imshow("mask", mask)


def main():
    parser = argparse.ArgumentParser(description="Test centor_color behavior.")
    parser.add_argument("color", choices=["red", "yellow", "green"])
    parser.add_argument("--error", type=float, default=metrics.CENTER_TOLERANCE)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--follow-area", type=float, default=metrics.LEAST_AREA_FOLLOW)
    parser.add_argument("--limit", type=float, default=metrics.CENTER_PID_LIMIT)
    parser.add_argument("--pid-p", type=float, default=metrics.CENTER_PID["P"])
    parser.add_argument("--pid-i", type=float, default=metrics.CENTER_PID["I"])
    parser.add_argument("--pid-d", type=float, default=metrics.CENTER_PID["D"])
    parser.add_argument("--print-interval", type=float, default=0.2)
    parser.add_argument("--camera-wait", type=float, default=5.0)
    args = parser.parse_args()

    try:
        metrics.CENTER_PID = {"P": args.pid_p, "I": args.pid_i, "D": args.pid_d}
        metrics.CENTER_PID_LIMIT = args.limit
        metrics.LEAST_AREA_FOLLOW = args.follow_area
        metrics.centerPidController = metrics.WheelSpeedPID(
            **metrics.CENTER_PID,
            target=0,
            lb=-metrics.CENTER_PID_LIMIT,
            ub=metrics.CENTER_PID_LIMIT,
        )

        metrics.init()
        metrics.getPic = 1
        metrics.picGet.start()
        if not wait_for_frame(args.camera_wait):
            raise RuntimeError("camera did not produce a frame")

        start = time.time()
        last_print = 0.0
        while time.time() - start < args.timeout:
            loop_start = time.time()
            frame = metrics.frame.astype(np.uint8)
            mask = metrics.getImg_Mask(frame, args.color)
            center_x, area, contours = metrics.get_Cube_center_area(mask)

            err = 0.0
            delta = metrics.centerPidController.u
            left = 0
            right = 0
            if area >= args.follow_area:
                err = center_x - metrics.Center[0]
                if abs(err) < args.error:
                    print(f"[done] abs(error) {abs(err):.1f} < {args.error:.1f}")
                    break
                delta = metrics.centerPidController.update(err)
                left, right = turn_with_center_delta(delta)
            else:
                metrics.stop(0)

            now = time.time()
            if now - last_print >= args.print_interval:
                print(
                    "t={:.2f}s center_x={} err={} area={:.0f} pid_delta={:.2f} duty=({:.1f}, {:.1f}) lspeed={:.3f} rspeed={:.3f}".format(
                        now - start,
                        f"{center_x:.1f}" if area > 0 else "None",
                        f"{err:.1f}" if area >= args.follow_area else "None",
                        area,
                        delta,
                        left,
                        right,
                        metrics.lspeed,
                        metrics.rspeed,
                    )
                )
                last_print = now

            draw_overlay(frame, mask, args.color, center_x, area, err, args.error)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            elapsed = time.time() - loop_start
            time.sleep(max(0, metrics.FORWARD_CONTROL_PERIOD - elapsed))
        else:
            print(f"[timeout] exceeded {args.timeout:.1f}s")

        metrics.stop()
    finally:
        metrics.shutdown()


if __name__ == "__main__":
    main()
