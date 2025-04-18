import sys
import os

# Ensure the project root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.eyetracking import EyeTracker

def main():
    # Instantiate the EnhancedEyeTracker with window display enabled.
    tracker = EyeTracker(show_windows=True)
    try:
        while True:
            # Update the tracker to get the latest gaze and blink event.
            gaze, blink_event = tracker.update()
            if gaze is not None:
                print(f"Gaze: {gaze}, Blink Event: {blink_event}")

            # Check if the quit key was pressed.
            if tracker.check_key():
                break
    finally:
        tracker.release()

if __name__ == "__main__":
    main()
