"""
Live visualization for get_Cube_center_area().

Examples:
python3 test_visual_center_area.py red
python3 test_visual_center_area.py yellow --camera 0 --print-interval 0.2

Press q or Esc to quit.
"""

import argparse
import time

import cv2

import metrics


def main():
    parser = argparse.ArgumentParser(description="Visualize color mask center_x and area.")
    parser.add_argument("color", choices=["red", "yellow", "green"])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=metrics.Width)
    parser.add_argument("--height", type=int, default=metrics.Height)
    parser.add_argument("--fps", type=int, default=metrics.fps)
    parser.add_argument("--print-interval", type=float, default=0.2)
    args = parser.parse_args()

    # metrics opens a camera at import time. Release it so this test owns the device.
    metrics.cap.release()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera] failed to read frame")
                time.sleep(0.2)
                continue

            mask = metrics.getImg_Mask(frame, args.color)
            center_x, area, contours = metrics.get_Cube_center_area(mask)

            overlay = frame.copy()
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            cv2.line(overlay, (args.width // 2, 0), (args.width // 2, args.height), (255, 0, 0), 1)

            if area > 0:
                x = int(center_x)
                cv2.line(overlay, (x, 0), (x, args.height), (0, 0, 255), 2)
                cv2.circle(overlay, (x, args.height // 2), 6, (0, 0, 255), -1)
                text = f"color={args.color} center_x={center_x:.1f} area={area:.0f}"
            else:
                text = f"color={args.color} center_x=None area=0"

            cv2.putText(overlay, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("center_area", overlay)
            cv2.imshow("mask", mask)

            now = time.time()
            if now - last_print >= args.print_interval:
                print(text)
                last_print = now

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
