"""
Independent test for the forward_color() behavior with live output.

The loop mirrors metrics.forward_color(), but adds timeout, live printing, and
an optional safety stop when the target is lost.

Examples:
python3 test_forward_color.py red --timeout 20
python3 test_forward_color.py yellow --duty 18 --stop-area 70000
python3 test_forward_color.py green --follow-area 1200
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


def average_hsv_in_detected_area(frame, contours):
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 500]
    if not filtered_contours:
        return None, 0

    detected_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(detected_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pixels = hsv[detected_mask > 0]
    if pixels.size == 0:
        return None, 0

    hue_angles = pixels[:, 0].astype(np.float32) * (2.0 * np.pi / 180.0)
    mean_sin = np.mean(np.sin(hue_angles))
    mean_cos = np.mean(np.cos(hue_angles))
    mean_hue = np.degrees(np.arctan2(mean_sin, mean_cos)) / 2.0
    if mean_hue < 0:
        mean_hue += 180.0

    mean_sv = np.mean(pixels[:, 1:3], axis=0)
    return (mean_hue, mean_sv[0], mean_sv[1]), len(pixels)


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
    parser.add_argument("--follow-area", type=float, default=metrics.LEAST_AREA_FOLLOW)
    parser.add_argument("--print-interval", type=float, default=0.2)
    parser.add_argument("--camera-wait", type=float, default=5.0)
    parser.add_argument(
        "--keep-moving-when-lost",
        action="store_true",
        help="Do not stop when area is below --follow-area.",
    )
    args = parser.parse_args()

    try:
        metrics.FORWARD_DUTY = args.duty
        metrics.AREA_FOR_TURN = args.stop_area
        metrics.LEAST_AREA_FOLLOW = args.follow_area
        metrics.pidController = metrics.WheelSpeedPID(**metrics.FORWARD_PID, target=0, lb=-8, ub=8)

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
            avg_hsv, hsv_pixel_count = average_hsv_in_detected_area(frame, contours)

            if area > args.stop_area:
                print(f"[done] area {area:.0f} exceeded stop area {args.stop_area:.0f}")
                break
            if area >= args.follow_area:
                metrics.forward(center_x)
            elif not args.keep_moving_when_lost:
                metrics.stop(0)

            now = time.time()
            if now - last_print >= args.print_interval:
                delta = metrics.pidController.u
                hsv_text = (
                    "None"
                    if avg_hsv is None
                    else "({:.1f}, {:.1f}, {:.1f}) n={}".format(
                        avg_hsv[0],
                        avg_hsv[1],
                        avg_hsv[2],
                        hsv_pixel_count,
                    )
                )
                print(
                    "t={:.2f}s center_x={} area={:.0f} avg_hsv={} pid_delta={:.2f} lspeed={:.3f} rspeed={:.3f}".format(
                        now - start,
                        f"{center_x:.1f}" if area > 0 else "None",
                        area,
                        hsv_text,
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

            elapsed = time.time() - loop_start
            time.sleep(max(0, metrics.FORWARD_CONTROL_PERIOD - elapsed))

        metrics.stop()
    finally:
        metrics.shutdown()


if __name__ == "__main__":
    main()
