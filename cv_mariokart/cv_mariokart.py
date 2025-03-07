from .controller import Controller
from .utils import find_dolphin_dir
from .car import Car
from eyetracking.eyetracking import EyeTracker
from .blinking import BlinkDetector
import time


def map_gaze_to_turning(gaze_x):
    """
    Map normalized gaze x-coordinate to a turning value.

    Args:
        gaze_x: Normalized gaze x-coordinate (0-1)

    Returns:
        A turning value where 0.5 is straight, >0.5 is right, <0.5 is left.
    """
    # Create a small deadzone in the center for stability
    deadzone = 0.02
    if abs(gaze_x - 0.5) < deadzone:
        return 0.5  # Center position (go straight)

    # Apply sensitivity scaling
    sensitivity = 8

    # Calculate turning value with sensitivity adjustment
    turning_value = 0.5 + (gaze_x - 0.5) * sensitivity
    return max(0, min(1, turning_value))


def run_with_eye_tracking(car):
    """
    Control the car with both eye tracking and blink detection.

    Gaze is used for turning, while blink events trigger actions:
      - Blink right when turning left (turning_value < 0.5) → Drift.
      - Blink left when turning right (turning_value > 0.5) → Drift.
      - Blink both → Use item.
    """
    # Initialize both trackers
    tracker = EyeTracker(show_windows=False)
    blink_detector = BlinkDetector(show_windows=False)

    try:
        print("Eye tracking activated. Position yourself with face clearly visible.")
        print("Press 'q' in the windows or ^C to stop.")
        print("- Look left/right to turn")
        print("- Blink right when turning left to drift")
        print("- Blink left when turning right to drift")
        print("- Blink both to use item")

        car.drive_forward()

        while True:
            # Get gaze data for turning
            gaze = tracker.update()
            if gaze is None:
                continue
            gaze_x, gaze_y = gaze
            turning_value = map_gaze_to_turning(gaze_x)
            car.turn(turning_value)

            # Get blink event from the blink detector
            blink_event, _ = blink_detector.update()
            if blink_event == "Both Blink":
                car.use_item()
            elif blink_event == "Right Blink" and turning_value < 0.5:
                car.drift()
            elif blink_event == "Left Blink" and turning_value > 0.5:
                car.drift()
            else:
                car.stop_drift()

            time.sleep(0.03)  # ~30 fps update rate

    finally:
        car.stop_car()
        tracker.release()
        blink_detector.release()
        print("Eye tracking stopped")


def main():
    dolphin_dir = find_dolphin_dir()
    if dolphin_dir is None:
        print("Could not find dolphin config dir.")
        return

    ctrl = None
    try:
        print("Start dolphin now. Press ^C to stop.")
        ctrl_path = dolphin_dir + "/Pipes/pipe"
        ctrl = Controller(ctrl_path)
        car = Car(ctrl)
        with ctrl:
            run_with_eye_tracking(car)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if ctrl is not None:
            print("Resetting controller")
            ctrl.reset()
            print("Controller reset complete")
        print("Stopped")


if __name__ == "__main__":
    main()
