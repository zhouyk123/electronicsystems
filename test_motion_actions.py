"""
Independent test for go_straight(), turn_left(), and turn_right().

Examples:
python3 test_motion_actions.py straight --duty 20 --dist 1 --timeout 5
python3 test_motion_actions.py left --duty 20 --angle 90 --timeout 5
python3 test_motion_actions.py right --duty 20 --angle 90 --timeout 5
"""

import argparse
import threading
import time

import metrics


def read_state():
    with metrics.speedLock:
        return {
            "move_left": metrics.move_lcounter,
            "move_right": metrics.move_rcounter,
            "threshold": metrics.threshold,
            "triggered": metrics.triggered,
            "lspeed": metrics.lspeed,
            "rspeed": metrics.rspeed,
        }


def main():
    parser = argparse.ArgumentParser(description="Run one motion action with live counters.")
    parser.add_argument("action", choices=["straight", "left", "right"])
    parser.add_argument("--duty", type=float, default=metrics.TURN_DUTY)
    parser.add_argument("--ratio", type=float, default=metrics.RATIO)
    parser.add_argument("--dist", type=float, default=1.0, help="Revolutions used by go_straight().")
    parser.add_argument("--angle", type=float, default=90.0, help="Degrees used by turn_left/right().")
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--print-interval", type=float, default=0.1)
    args = parser.parse_args()

    error = []

    def run_action():
        try:
            if args.action == "straight":
                metrics.go_straight(args.duty, args.ratio, args.dist)
            elif args.action == "left":
                metrics.turn_left(args.duty, args.angle)
            else:
                metrics.turn_right(args.duty, args.angle)
        except BaseException as exc:
            error.append(exc)

    try:
        metrics.init()
        time.sleep(0.2)

        worker = threading.Thread(target=run_action, daemon=True)
        worker.start()

        start = time.time()
        last_print = 0.0
        while worker.is_alive() and time.time() - start < args.timeout:
            now = time.time()
            if now - last_print >= args.print_interval:
                state = read_state()
                target_pulses = state["threshold"] * metrics.SPEED_PULSE_PER_REV
                print(
                    "t={:.2f}s target_pulses={:.1f} left_count={} right_count={} "
                    "lspeed={:.3f} rspeed={:.3f} triggered={}".format(
                        now - start,
                        target_pulses,
                        state["move_left"],
                        state["move_right"],
                        state["lspeed"],
                        state["rspeed"],
                        state["triggered"],
                    )
                )
                last_print = now
            time.sleep(0.01)

        if worker.is_alive():
            print(f"[timeout] {args.action} did not finish in {args.timeout:.2f}s; stopping.")
            metrics.force_move_done()
            metrics.brake(0.1)
            worker.join(1.0)

        if error:
            raise error[0]

        state = read_state()
        print(
            "[done] left_count={} right_count={} lspeed={:.3f} rspeed={:.3f}".format(
                state["move_left"], state["move_right"], state["lspeed"], state["rspeed"]
            )
        )
    finally:
        metrics.shutdown()


if __name__ == "__main__":
    main()
