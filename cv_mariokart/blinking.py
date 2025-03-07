import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

class BlinkDetector:
    def __init__(self, show_windows=False):
        """
        Initialize the blink detector.
        
        Args:
            show_windows (bool): If True, display a window with annotated video.
        """
        self.show_windows = show_windows
        self.cap = cv2.VideoCapture(0)  # Use built-in camera
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Inverted indices: use the indices originally assigned for the right eye as left and vice versa.
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]    # Originally for the right eye.
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]   # Originally for the left eye.

        # Threshold parameters:
        # Single-eye threshold: an eye is considered closed if its EAR is below this value.
        self.EYE_AR_THRESH_SINGLE = 0.25  
        # Both-eye threshold: for a "Both Blink" event, require a stricter (lower) average EAR.
        self.EYE_AR_THRESH_BOTH = 0.22

    def eye_aspect_ratio(self, eye_points):
        """
        Compute the Eye Aspect Ratio (EAR) for 6 eye landmark points.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C)

    def update(self):
        """
        Capture a frame from the camera, process it to detect blink state, and return the blink event.
        
        Returns:
            blink_event (str): One of "Left Blink", "Right Blink", "Both Blink", or "No Blink".
            frame (np.array): The annotated video frame.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # Mirror view
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        blink_event = "No Blink"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Extract eye landmark coordinates (convert normalized coords to pixels)
            left_eye_points = []
            right_eye_points = []
            for idx in self.LEFT_EYE_IDX:
                lm = face_landmarks.landmark[idx]
                left_eye_points.append((int(lm.x * width), int(lm.y * height)))
            for idx in self.RIGHT_EYE_IDX:
                lm = face_landmarks.landmark[idx]
                right_eye_points.append((int(lm.x * width), int(lm.y * height)))

            # Compute EAR for each eye
            left_ear = self.eye_aspect_ratio(left_eye_points)
            right_ear = self.eye_aspect_ratio(right_eye_points)

            # Determine closed state per eye
            left_closed = left_ear < self.EYE_AR_THRESH_SINGLE
            right_closed = right_ear < self.EYE_AR_THRESH_SINGLE
            both_closed_strong = ((left_ear + right_ear) / 2.0) < self.EYE_AR_THRESH_BOTH

            # Determine blink event:
            if left_closed and right_closed:
                if both_closed_strong:
                    blink_event = "Both Blink"
                else:
                    blink_event = "Left Blink" if left_ear < right_ear else "Right Blink"
            elif left_closed:
                blink_event = "Left Blink"
            elif right_closed:
                blink_event = "Right Blink"
            else:
                blink_event = "No Blink"

            # Draw landmarks for visualization
            for (x, y) in left_eye_points:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_eye_points:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, blink_event, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if self.show_windows:
            cv2.imshow("Blink Detector", frame)
            cv2.waitKey(1)
        return blink_event, frame

    def release(self):
        """Release camera resources."""
        self.cap.release()
        cv2.destroyAllWindows()