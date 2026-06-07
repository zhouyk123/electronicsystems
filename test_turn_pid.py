"""
python3 test_turn_pid.py left --duty 20 --time 5
python3 test_turn_pid.py right --duty 40 --time 3
"""

import argparse
import time

import RPi.GPIO as GPIO

import metrics


def main():
    parser = argparse.ArgumentParser(description="Test PID speed control for turning.")
    parser.add_argument("direction", choices=["left", "right"], help="Turn direction to test.")
    parser.add_argument("--duty", type=float, default=metrics.TURN_DUTY, help="Requested turn duty.")
    parser.add_argument("--time", type=float, default=3.0, help="Test duration in seconds.")
    parser.add_argument("--interval", type=float, default=0.2, help="Print interval in seconds.")
    args = parser.parse_args()

    try:
        print("[init] GPIO and speed PID setup")
        metrics.init()
        time.sleep(0.3)

        if args.direction == "left":
            print(f"[test] turn_left duty={args.duty} duration={args.time}s")
            metrics.turn_left(args.duty)
        else:
            print(f"[test] turn_right duty={args.duty} duration={args.time}s")
            metrics.turn_right(args.duty)

        start = time.time()
        while time.time() - start < args.time:
            with metrics.turnPidLock:
                controller = metrics.turnPidController
                pid_u = controller.u if controller else 0
                target_speed = controller.target_speed if controller else 0

            print(
                "t={:.2f}s target={:.3f} lspeed={:.3f} rspeed={:.3f} pid_duty={:.2f}".format(
                    time.time() - start,
                    target_speed,
                    metrics.lspeed,
                    metrics.rspeed,
                    pid_u,
                )
            )
            time.sleep(args.interval)

        metrics.stop()
        time.sleep(0.2)
        print("[done] stopped")
    except KeyboardInterrupt:
        metrics.stop()
        print("\n[stop] interrupted")
    finally:
        metrics.getSpeed = 0
        GPIO.cleanup()


if __name__ == "__main__":
    main()
