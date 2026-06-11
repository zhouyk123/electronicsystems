"""
Manual test runner for one decomposed action in metrics.py.

By default this is a dry run: it prints the primitive calls without touching
GPIO, motors, camera, or real sleep.

Examples
python3 test_one_decomposed_action.py _move_forward --dist 2 --real
python3 test_one_decomposed_action.py _move_left --dist 2 --real
python3 test_one_decomposed_action.py _move_right --dist 2 --real
python3 test_one_decomposed_action.py bypass_left --color red --dist 2
python3 test_one_decomposed_action.py circle_clockwise --dist 2 --real

Run on the Raspberry Pi for real hardware movement:
python3 test_one_decomposed_action.py _move_left --dist 1 --real
python3 test_one_decomposed_action.py approach_color --color red --real
"""

import argparse
import sys
import time
import types


ACTION_NAMES = [
    "_rest",
    "_move_forward",
    "_move_left",
    "_move_right",
    "_move_backward",
    "approach_color",
    "bypass_left",
    "bypass_right",
    "circle_clockwise",
    "circle_anticlockwise",
]

COLOR_ACTIONS = {"approach_color", "bypass_left", "bypass_right"}
DIST_ACTIONS = {
    "_move_forward",
    "_move_left",
    "_move_right",
    "_move_backward",
    "bypass_left",
    "bypass_right",
    "circle_clockwise",
    "circle_anticlockwise",
}
CAMERA_ACTIONS = {"approach_color", "bypass_left", "bypass_right"}
MOTION_ACTIONS = DIST_ACTIONS | CAMERA_ACTIONS


def parse_args():
    parser = argparse.ArgumentParser(description="Manually run one decomposed action.")
    parser.add_argument("action", choices=ACTION_NAMES)
    parser.add_argument("--color", choices=["red", "yellow", "green"], default="red")
    parser.add_argument("--dist", type=float, default=1.0)
    parser.add_argument("--real", action="store_true", help="Actually run motors/camera on Raspberry Pi.")
    parser.add_argument("--camera-wait", type=float, default=5.0)
    parser.add_argument("--turn-duty", type=float, help="Override metrics.TURN_DUTY for this run.")
    parser.add_argument("--straight-duty", type=float, help="Override metrics.STRAIGHT_DUTY for this run.")
    parser.add_argument("--ratio", type=float, help="Override metrics.RATIO for this run.")
    parser.add_argument("--rest-time", type=float, help="Override metrics.INTERVAL_SLEEP_TIME for this run.")
    return parser.parse_args()


def install_fake_gpio():
    rpi_module = types.ModuleType("RPi")
    gpio_module = types.ModuleType("RPi.GPIO")

    gpio_module.BCM = "BCM"
    gpio_module.OUT = "OUT"
    gpio_module.IN = "IN"
    gpio_module.LOW = 0
    gpio_module.HIGH = 1
    gpio_module.RISING = "RISING"

    def no_op(*args, **kwargs):
        return None

    class FakePWM:
        def __init__(self, *args, **kwargs):
            self.duty = 0

        def start(self, duty):
            self.duty = duty

        def ChangeDutyCycle(self, duty):
            self.duty = duty

        def stop(self):
            self.duty = 0

    gpio_module.setwarnings = no_op
    gpio_module.setmode = no_op
    gpio_module.cleanup = no_op
    gpio_module.setup = no_op
    gpio_module.output = no_op
    gpio_module.add_event_detect = no_op
    gpio_module.remove_event_detect = no_op
    gpio_module.PWM = FakePWM

    rpi_module.GPIO = gpio_module
    sys.modules["RPi"] = rpi_module
    sys.modules["RPi.GPIO"] = gpio_module


def install_fake_camera():
    try:
        import cv2
    except ImportError:
        cv2 = types.ModuleType("cv2")
        cv2.CAP_PROP_FRAME_WIDTH = 3
        cv2.CAP_PROP_FRAME_HEIGHT = 4
        cv2.CAP_PROP_FPS = 5
        sys.modules["cv2"] = cv2

    class FakeCapture:
        def __init__(self, *args, **kwargs):
            self.props = {}

        def set(self, key, value):
            self.props[key] = value
            return True

        def read(self):
            return False, None

        def release(self):
            return None

    cv2.VideoCapture = FakeCapture


def install_fake_matplotlib():
    matplotlib_module = types.ModuleType("matplotlib")
    pyplot_module = types.ModuleType("matplotlib.pyplot")
    matplotlib_module.pyplot = pyplot_module
    sys.modules["matplotlib"] = matplotlib_module
    sys.modules["matplotlib.pyplot"] = pyplot_module


def import_metrics(real):
    if not real:
        install_fake_gpio()
        install_fake_camera()
        install_fake_matplotlib()

    import metrics

    return metrics


def apply_overrides(metrics, args):
    if args.turn_duty is not None:
        metrics.TURN_DUTY = args.turn_duty
    if args.straight_duty is not None:
        metrics.STRAIGHT_DUTY = args.straight_duty
    if args.ratio is not None:
        metrics.RATIO = args.ratio
    if args.rest_time is not None:
        metrics.INTERVAL_SLEEP_TIME = args.rest_time


def patch_dry_run(metrics):
    def trace(name):
        def recorder(*args, **kwargs):
            text = ", ".join([repr(arg) for arg in args])
            if kwargs:
                kw_text = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
                text = f"{text}, {kw_text}" if text else kw_text
            print(f"{name}({text})")

        return recorder

    metrics.go_straight = trace("go_straight")
    metrics.turn_left = trace("turn_left")
    metrics.turn_right = trace("turn_right")
    metrics.detected_color = trace("detected_color")
    metrics.forward_color = trace("forward_color")
    metrics.stop = trace("stop")
    metrics.time.sleep = trace("sleep")


def wait_for_frame(metrics, timeout):
    start = time.time()
    while time.time() - start < timeout:
        if isinstance(metrics.frame, metrics.np.ndarray):
            return True
        time.sleep(0.05)
    return False


def prepare_real_run(metrics, args):
    if not args.real:
        return
    if args.action not in MOTION_ACTIONS:
        return

    metrics.init()
    time.sleep(0.2)

    if args.action in CAMERA_ACTIONS:
        metrics.pidController = metrics.WheelSpeedPID(**metrics.FORWARD_PID, target=0, lb=-8, ub=8)
        metrics.getPic = 1
        if not metrics.picGet.is_alive():
            metrics.picGet.start()
        if not wait_for_frame(metrics, args.camera_wait):
            raise RuntimeError("camera did not produce a frame")


def run_action(metrics, args):
    action = getattr(metrics, args.action)
    if args.action in COLOR_ACTIONS and args.action in DIST_ACTIONS:
        action(args.color, args.dist)
    elif args.action in COLOR_ACTIONS:
        action(args.color)
    elif args.action in DIST_ACTIONS:
        action(args.dist)
    else:
        action()


def main():
    args = parse_args()
    metrics = import_metrics(args.real)
    apply_overrides(metrics, args)

    print(f"[action] {args.action}")
    if args.action in COLOR_ACTIONS:
        print(f"[color] {args.color}")
    if args.action in DIST_ACTIONS:
        print(f"[dist] {args.dist}")

    if args.real:
        print("[mode] real hardware")
    else:
        print("[mode] dry-run")
        patch_dry_run(metrics)

    try:
        prepare_real_run(metrics, args)
        run_action(metrics, args)
        print("[done]")
    except KeyboardInterrupt:
        print("[interrupted]")
    finally:
        if args.real:
            metrics.shutdown()


if __name__ == "__main__":
    main()
