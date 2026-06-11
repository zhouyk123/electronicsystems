"""
Unit tests for metrics.py decomposed motion actions.

These tests do not touch GPIO, motors, the camera, or real sleep. They verify
the order and arguments of the action composition only.

Run:
python3 -m unittest test_decomposed_actions.py
"""

import sys
import types
import unittest
from unittest import mock


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


install_fake_gpio()
install_fake_camera()
install_fake_matplotlib()

import metrics


class DecomposedActionTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        patches = [
            mock.patch.object(metrics.time, "sleep", self.record("sleep")),
            mock.patch.object(metrics, "go_straight", self.record("go_straight")),
            mock.patch.object(metrics, "turn_left", self.record("turn_left")),
            mock.patch.object(metrics, "turn_right", self.record("turn_right")),
            mock.patch.object(metrics, "detected_color", self.record("detected_color")),
            mock.patch.object(metrics, "forward_color", self.record("forward_color")),
            mock.patch.object(metrics, "stop", self.record("stop")),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def record(self, name):
        def recorder(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return recorder

    def assert_calls(self, expected):
        self.assertEqual(self.calls, [(name, args, {}) for name, args in expected])

    def test_rest(self):
        metrics._rest()

        self.assert_calls([
            ("sleep", (metrics.INTERVAL_SLEEP_TIME,)),
        ])

    def test_move_forward(self):
        metrics._move_forward(3)

        self.assert_calls([
            ("go_straight", (metrics.STRAIGHT_DUTY, metrics.RATIO, 3)),
            ("sleep", (metrics.INTERVAL_SLEEP_TIME,)),
        ])

    def test_move_left(self):
        metrics._move_left(3)

        self.assert_calls([
            ("turn_left", (metrics.TURN_DUTY, 90)),
            ("sleep", (metrics.INTERVAL_SLEEP_TIME,)),
            ("go_straight", (metrics.STRAIGHT_DUTY, metrics.RATIO, 3)),
            ("sleep", (metrics.INTERVAL_SLEEP_TIME,)),
            ("turn_right", (metrics.TURN_DUTY, 90)),
            ("sleep", (metrics.INTERVAL_SLEEP_TIME,)),
        ])

    def test_move_right(self):
        metrics._move_right(3)

        self.assert_calls([
            ("turn_right", (metrics.TURN_DUTY, 90)),
            ("sleep", (metrics.INTERVAL_SLEEP_TIME,)),
            ("go_straight", (metrics.STRAIGHT_DUTY, metrics.RATIO, 3)),
            ("sleep", (metrics.INTERVAL_SLEEP_TIME,)),
            ("turn_left", (metrics.TURN_DUTY, 90)),
            ("sleep", (metrics.INTERVAL_SLEEP_TIME,)),
        ])

    def test_approach_color(self):
        metrics.approach_color("red")

        self.assert_calls([
            ("detected_color", ("red",)),
            ("forward_color", ("red",)),
            ("stop", ()),
            ("sleep", (metrics.INTERVAL_SLEEP_TIME,)),
        ])

    def test_move_backward(self):
        metrics._move_backward(3)

        self.assert_calls([
            ("turn_left", (metrics.TURN_DUTY, 180)),
            ("sleep", (metrics.INTERVAL_SLEEP_TIME,)),
            ("go_straight", (metrics.STRAIGHT_DUTY, metrics.RATIO, 3)),
            ("sleep", (metrics.INTERVAL_SLEEP_TIME,)),
            ("turn_left", (metrics.TURN_DUTY, 180)),
            ("sleep", (metrics.INTERVAL_SLEEP_TIME,)),
        ])


class CompositeActionTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        patches = [
            mock.patch.object(metrics, "approach_color", self.record("approach_color")),
            mock.patch.object(metrics, "_move_forward", self.record("_move_forward")),
            mock.patch.object(metrics, "_move_left", self.record("_move_left")),
            mock.patch.object(metrics, "_move_right", self.record("_move_right")),
            mock.patch.object(metrics, "_move_backward", self.record("_move_backward")),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def record(self, name):
        def recorder(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return recorder

    def assert_calls(self, expected):
        self.assertEqual(self.calls, [(name, args, {}) for name, args in expected])

    def test_bypass_left(self):
        metrics.bypass_left("red", 6)

        self.assert_calls([
            ("approach_color", ("red",)),
            ("_move_left", (3.0,)),
            ("_move_forward", (6,)),
            ("_move_right", (3.0,)),
        ])

    def test_bypass_right(self):
        metrics.bypass_right("yellow", 6)

        self.assert_calls([
            ("approach_color", ("yellow",)),
            ("_move_right", (3.0,)),
            ("_move_forward", (6,)),
            ("_move_left", (3.0,)),
        ])

    def test_circle_clockwise(self):
        metrics.circle_clockwise(4)

        self.assert_calls([
            ("_move_left", (2.0,)),
            ("_move_forward", (4,)),
            ("_move_right", (4,)),
            ("_move_backward", (4,)),
            ("_move_left", (4,)),
            ("_move_forward", (4,)),
            ("_move_right", (2.0,)),
        ])

    def test_circle_anticlockwise(self):
        metrics.circle_anticlockwise(4)

        self.assert_calls([
            ("_move_right", (2.0,)),
            ("_move_forward", (4,)),
            ("_move_left", (4,)),
            ("_move_backward", (4,)),
            ("_move_right", (4,)),
            ("_move_forward", (4,)),
            ("_move_left", (2.0,)),
        ])


if __name__ == "__main__":
    unittest.main()
