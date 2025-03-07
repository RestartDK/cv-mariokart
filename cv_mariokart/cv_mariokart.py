from .controller import Controller
from .utils import find_dolphin_dir
from .car import Car
from eyetracking.eyetracking import EyeTracker
import time


def map_gaze_to_turning(gaze_x):
    """
    Map normalized gaze x-coordinate to a turning value.

    Args:
        gaze_x: Normalized gaze x-coordinate (0-1)

    Returns:
        A turning value where 0.5 is straight, >0.5 is right, <0.5 is left
    """
    # Create a small deadzone in the center for stability
    deadzone = 0.02
    if abs(gaze_x - 0.5) < deadzone:
        return 0.5  # Center position (go straight)

    # Apply sensitivity scaling
    sensitivity = 8

    # Calculate turning value with sensitivity adjustment
    turning_value = 0.5 + (gaze_x - 0.5) * sensitivity

    # Ensure values are within 0-1 range
    turning_value = max(0, min(1, turning_value))

    return turning_value


def run_with_eye_tracking(car):
    """
    Control the car with eye tracking input.

    Args:
        car: An initialized Car instance
    """
    # Initialize eye tracker
    tracker = EyeTracker(
        show_windows=False,
    )

    try:
        print("Eye tracking activated. Position yourself with face clearly visible.")
        print("Press 'q' in the eye tracking window or ^C to stop.")

        # Start driving forward
        car.drive_forward()

        # Main control loop
        while True:
            # Get normalized gaze coordinates (0-1 range)
            gaze_x, gaze_y = tracker.update()

            # Map gaze_x to turning value
            turning_value = map_gaze_to_turning(gaze_x)
            print(turning_value)

            # Apply turning
            car.turn(turning_value)

            # Add a small delay to avoid overwhelming the controller
            time.sleep(0.03)  # ~30 fps update rate

    finally:
        # Stop the car and clean up resources
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

        # Create car instance
        car = Car(ctrl)

        # Use the car with eye tracking
        with ctrl:  # Use context manager to handle opening/closing the pipe
            # Run with eye tracking
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
