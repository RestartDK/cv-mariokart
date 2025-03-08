import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from vision.eyetracking import EyeTracker


def main():
    eye_tracker = EyeTracker(show_wiqndows=True)

    try:
        print("Press 'q' to quit.")
        while True:
            eye_tracker.update()
            if eye_tracker.check_key():
                break
    finally:
        eye_tracker.release()
        print("Eye tracking stopped.")


if __name__ == "__main__":
    main()
