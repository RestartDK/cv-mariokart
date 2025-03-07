import cv2
import numpy as np
from typing import Tuple


class EyeTracker:
    """A class to handle eye tracking and provide gaze coordinates."""

    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        screen_dimensions: Tuple[int, int] = (800, 500),
        smoothing_factor: float = 0.5,
        buffer_size: int = 3,
        show_windows: bool = False,
    ):
        """
        Initialize the eye tracker.

        Args:
            camera_id: ID of the camera to use
            resolution: Desired camera resolution (width, height)
            screen_dimensions: Virtual screen dimensions (width, height)
            smoothing_factor: Factor for smoothing gaze movements (0-1)
            buffer_size: Size of moving average buffer for smoothing
            show_windows: Whether to show the video windows
        """
        # Camera setup
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # Load cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade1 = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.eye_cascade2 = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )

        # Screen dimensions
        self.screen_width, self.screen_height = screen_dimensions
        self.screen = np.zeros(
            (self.screen_height, self.screen_width, 3), dtype=np.uint8
        )

        # Smoothing parameters
        self.smoothing_factor = smoothing_factor
        self.last_screen_x, self.last_screen_y = (
            self.screen_width // 2,
            self.screen_height // 2,
        )

        # Moving average buffer
        self.buffer_size = buffer_size
        self.x_buffer = [self.screen_width // 2] * buffer_size
        self.y_buffer = [self.screen_height // 2] * buffer_size

        # Display options
        self.show_windows = show_windows
        if show_windows:
            cv2.namedWindow("Eye Tracking")
            cv2.namedWindow("Screen View")

        # Store the latest gaze coordinates
        self.current_gaze = (self.screen_width // 2, self.screen_height // 2)

    def get_normalized_gaze(self) -> Tuple[float, float]:
        """
        Get the current gaze coordinates normalized to range [0,1].

        Returns:
            Tuple of (x, y) where x and y are in range [0,1]
        """
        x, y = self.current_gaze
        return x / self.screen_width, y / self.screen_height

    def update(self) -> Tuple[float, float]:
        """
        Update the eye tracker and return the current normalized gaze position.

        Returns:
            Tuple of (x, y) where x and y are in range [0,1]
        """
        ret, frame = self.cap.read()
        if not ret:
            return self.get_normalized_gaze()

        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 4, minSize=(80, 80))

        # Clear screen visualization if showing windows
        if self.show_windows:
            self.screen.fill(0)
            cv2.rectangle(
                self.screen,
                (0, 0),
                (self.screen_width - 1, self.screen_height - 1),
                (50, 50, 50),
                2,
            )

        # Process face and eyes if any detected
        if len(faces) > 0:
            # Use the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face

            # Draw face rectangle if showing windows
            if self.show_windows:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Region of interest for the face
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = frame[y : y + h, x : x + w] if self.show_windows else None

            # Try to detect eyes with first cascade
            eyes = self.eye_cascade1.detectMultiScale(
                roi_gray, 1.1, 3, minSize=(20, 20)
            )

            # If that didn't work, try the second cascade
            if len(eyes) < 1:
                eyes = self.eye_cascade2.detectMultiScale(
                    roi_gray, 1.05, 3, minSize=(15, 15)
                )

            # Process eyes and update gaze
            self._process_eyes(eyes, roi_gray, roi_color, x, y, frame)

        # Show frames if needed
        if self.show_windows:
            self._show_frames(frame)

        # Return normalized gaze position
        return self.get_normalized_gaze()

    def _process_eyes(self, eyes, roi_gray, roi_color, face_x, face_y, frame):
        """Process detected eyes and update gaze coordinates."""
        pupil_positions = []

        for i, (ex, ey, ew, eh) in enumerate(eyes):
            # Draw eye rectangle if showing windows
            if self.show_windows and roi_color is not None:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Extract eye ROI
            eye_roi = roi_gray[ey : ey + eh, ex : ex + ew]

            if eye_roi.size > 0:
                # Process the eye to find the pupil
                blurred = cv2.GaussianBlur(eye_roi, (7, 7), 0)
                _, threshold = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)

                # Find contours
                contours, _ = cv2.findContours(
                    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Calculate the center of the contour (pupil)
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        pupil_x = int(M["m10"] / M["m00"])
                        pupil_y = int(M["m01"] / M["m00"])

                        # Draw pupil center if showing windows
                        if self.show_windows:
                            absolute_pupil_x = face_x + ex + pupil_x
                            absolute_pupil_y = face_y + ey + pupil_y
                            cv2.circle(
                                frame,
                                (absolute_pupil_x, absolute_pupil_y),
                                3,
                                (0, 0, 255),
                                -1,
                            )

                        # Calculate relative pupil position within the eye
                        rel_x = pupil_x / ew
                        rel_y = pupil_y / eh

                        # Store the pupil position
                        pupil_positions.append((rel_x, rel_y))

                        # Display info if showing windows
                        if self.show_windows:
                            eye_text = (
                                f"Eye {i + 1}: Pupil at ({rel_x:.2f}, {rel_y:.2f})"
                            )
                            cv2.putText(
                                frame,
                                eye_text,
                                (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )

        # Map pupil positions to screen coordinates
        if pupil_positions:
            self._update_gaze_from_pupils(pupil_positions)

    def _update_gaze_from_pupils(self, pupil_positions):
        """Update gaze coordinates based on pupil positions."""
        # Average the pupil positions if more than one eye is detected
        avg_pupil_x = sum(p[0] for p in pupil_positions) / len(pupil_positions)
        avg_pupil_y = sum(p[1] for p in pupil_positions) / len(pupil_positions)

        # Normalize pupil position
        norm_x = (avg_pupil_x - 0.5) * 2.0
        norm_y = (avg_pupil_y - 0.5) * 2.0

        # Map to screen coordinates
        screen_x = int(self.screen_width * (0.5 + norm_x * 0.8))
        screen_y = int(self.screen_height * (0.5 + norm_y * 0.5))

        # Ensure screen coordinates are within bounds
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))

        # Apply first level of smoothing
        screen_x = int(
            self.last_screen_x * self.smoothing_factor
            + screen_x * (1 - self.smoothing_factor)
        )
        screen_y = int(
            self.last_screen_y * self.smoothing_factor
            + screen_y * (1 - self.smoothing_factor)
        )

        # Add to moving average buffer
        self.x_buffer.pop(0)
        self.x_buffer.append(screen_x)
        self.y_buffer.pop(0)
        self.y_buffer.append(screen_y)

        # Apply second level of smoothing with moving average
        screen_x = sum(self.x_buffer) // len(self.x_buffer)
        screen_y = sum(self.y_buffer) // len(self.y_buffer)

        # Update current position
        self.last_screen_x, self.last_screen_y = screen_x, screen_y
        self.current_gaze = (screen_x, screen_y)

        # Update visualization if showing windows
        if self.show_windows:
            cv2.circle(self.screen, (screen_x, screen_y), 15, (0, 0, 255), -1)
            cv2.circle(self.screen, (screen_x, screen_y), 20, (0, 255, 255), 2)
            cv2.putText(
                self.screen,
                f"Gaze: ({screen_x}, {screen_y})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    def _show_frames(self, frame):
        """Show the camera and screen visualization frames."""
        # Add quit instructions
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Show frames
        cv2.imshow("Eye Tracking", frame)
        cv2.imshow("Screen View", self.screen)

    def check_key(self) -> bool:
        """
        Check for key presses and return True if quit key was pressed.

        Returns:
            True if quit key was pressed, False otherwise
        """
        if self.show_windows:
            key = cv2.waitKey(1) & 0xFF
            return key == ord("q")
        return False

    def release(self):
        """Release resources."""
        self.cap.release()
        if self.show_windows:
            cv2.destroyAllWindows()
