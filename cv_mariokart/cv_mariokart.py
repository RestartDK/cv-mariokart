from .controller import Controller
from .utils import find_dolphin_dir
from .car import Car
from eyetracking.eyetracking import EyeTracker  # New integrated eye tracking class
import time

def map_gaze_to_turning(gaze_x):
    """
    Map normalized gaze x-coordinate to a turning value.

    Args:
        gaze_x: Normalized gaze x-coordinate (0-1)

    Returns:
        A turning value where 0.5 is straight, >0.5 is right, <0.5 is left.
    """
    deadzone = 0.02
    if abs(gaze_x - 0.5) < deadzone:
        return 0.5  # Center position (go straight)
    sensitivity = 8
    turning_value = 0.5 + (gaze_x - 0.5) * sensitivity
    return max(0, min(1, turning_value))

def run_with_eye_tracking(car):
    """
    Control the car with eye tracking and integrated blink detection.

    Gaze is used for turning:
      - When turning left (turning_value < 0.5) and a "Right Blink" is detected, drift.
      - When turning right (turning_value > 0.5) and a "Left Blink" is detected, drift.
      - A sustained both-eye blink (held for >1 second) triggers using an item.
    """
    tracker = EyeTracker(show_windows=True)
    both_blink_start_time = None

    try:
        print("Eye tracking activated. Position yourself with face clearly visible.")
        print("Press 'q' in the windows or ^C to stop.")
        print("- Look left/right to turn")
        print("- Blink right when turning left to drift")
        print("- Blink left when turning right to drift")
        print("- Blink both for >1 second to use an item")

        car.drive_forward()

        while True:
            # Update eye tracking to get normalized gaze and blink event.
            gaze, blink_event = tracker.update()
            if gaze is None:
                continue
            gaze_x, gaze_y = gaze
            turning_value = map_gaze_to_turning(gaze_x)
            car.turn(turning_value)

            current_time = time.time()

            # Check for a sustained both-eye blink (held > 1 second) to use an item.
            if blink_event == "Both Blink":
                if both_blink_start_time is None:
                    both_blink_start_time = current_time
                elif current_time - both_blink_start_time >= 1.0:
                    car.use_item()
                    # Reset so item use is triggered only once per sustained blink.
                    both_blink_start_time = None
            else:
                both_blink_start_time = None

            # For drifting: if right blink while turning left, or left blink while turning right.
            if blink_event == "Right Blink" and turning_value < 0.5:
                car.drift()
            elif blink_event == "Left Blink" and turning_value > 0.5:
                car.drift()
            else:
                car.stop_drift()

            time.sleep(0.03)  # ~30 fps update rate

    finally:
        car.stop_car()
        tracker.release()
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
