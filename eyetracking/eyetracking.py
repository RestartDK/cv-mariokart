import cv2
import numpy as np


def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    # Increase resolution if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Load face cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Try different eye cascades - sometimes one works better than others
    eye_cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eye_cascade2 = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    )

    # Virtual screen dimensions
    screen_width, screen_height = 800, 500
    screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Smoothing parameters - increased for less twitchy movement
    smoothing_factor = 0.5
    last_screen_x, last_screen_y = screen_width // 2, screen_height // 2

    # Moving average for even smoother tracking
    buffer_size = 3
    x_buffer = [screen_width // 2] * buffer_size
    y_buffer = [screen_height // 2] * buffer_size

    print("Starting eye tracking...")
    print("Position yourself with your face clearly visible")
    print("Press 'q' to quit")

    cv2.namedWindow("Eye Tracking")
    cv2.namedWindow("Screen View")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces - reduced minSize for better distance detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 4, minSize=(80, 80))

        # Create a fresh screen visualization
        screen.fill(0)
        cv2.rectangle(
            screen, (0, 0), (screen_width - 1, screen_height - 1), (50, 50, 50), 2
        )

        if len(faces) > 0:
            # Use the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face

            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Region of interest for the face
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = frame[y : y + h, x : x + w]

            # Try to detect eyes with first cascade - reduced minSize for better distance detection
            eyes = eye_cascade1.detectMultiScale(roi_gray, 1.1, 3, minSize=(20, 20))

            # If that didn't work, try the second cascade with even more permissive parameters
            if len(eyes) < 1:
                eyes = eye_cascade2.detectMultiScale(
                    roi_gray, 1.05, 3, minSize=(15, 15)
                )

            # Initialize variables for pupil positions
            pupil_positions = []

            # Process detected eyes
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                # Draw eye rectangle
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # Extract eye ROI
                eye_roi = roi_gray[ey : ey + eh, ex : ex + ew]

                if eye_roi.size > 0:  # Make sure the eye ROI is valid
                    # Process the eye to find the pupil
                    blurred = cv2.GaussianBlur(eye_roi, (7, 7), 0)
                    _, threshold = cv2.threshold(
                        blurred, 40, 255, cv2.THRESH_BINARY_INV
                    )

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

                            # Draw pupil center on the original frame
                            absolute_pupil_x = x + ex + pupil_x
                            absolute_pupil_y = y + ey + pupil_y
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

                            # Display the relative pupil position
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

                    # Show the processed eye in the corner of the frame
                    if threshold.size > 0:
                        # Resize for visualization
                        disp_h, disp_w = 60, 80
                        try:
                            resized_eye = cv2.resize(threshold, (disp_w, disp_h))

                            # Convert to 3-channel for visualization
                            eye_vis = cv2.cvtColor(resized_eye, cv2.COLOR_GRAY2BGR)

                            # Place in corner
                            pos_y, pos_x = 10, 10 + i * (disp_w + 10)

                            # Make sure we don't go out of bounds
                            if (
                                frame.shape[0] > pos_y + disp_h
                                and frame.shape[1] > pos_x + disp_w
                            ):
                                frame[
                                    pos_y : pos_y + disp_h, pos_x : pos_x + disp_w
                                ] = eye_vis
                        except Exception as e:
                            print(f"Error displaying eye: {e}")

            # Map the pupil positions to screen coordinates
            if pupil_positions:
                # Average the pupil positions if more than one eye is detected
                avg_pupil_x = sum(p[0] for p in pupil_positions) / len(pupil_positions)
                avg_pupil_y = sum(p[1] for p in pupil_positions) / len(pupil_positions)

                # Simple mapping to screen coordinates
                # Adjust these parameters to change sensitivity and mapping

                # Normalize pupil position (typically around 0.5 for center)
                norm_x = (avg_pupil_x - 0.5) * 2.0  # Reduced from 2.5 to 2.0
                norm_y = (avg_pupil_y - 0.5) * 2.0

                # Map to screen coordinates with offset correction
                screen_x = int(
                    screen_width * (0.5 + norm_x * 0.8)
                )  # Reduced from 1.0 to 0.8
                screen_y = int(screen_height * (0.5 + norm_y * 0.5))
                # NO inversion needed (removed the line that inverted norm_x)

                # Ensure screen coordinates are within bounds
                screen_x = max(0, min(screen_width - 1, screen_x))
                screen_y = max(0, min(screen_height - 1, screen_y))

                # Apply first level of smoothing
                screen_x = int(
                    last_screen_x * smoothing_factor + screen_x * (1 - smoothing_factor)
                )
                screen_y = int(
                    last_screen_y * smoothing_factor + screen_y * (1 - smoothing_factor)
                )

                # Add to moving average buffer
                x_buffer.pop(0)
                x_buffer.append(screen_x)
                y_buffer.pop(0)
                y_buffer.append(screen_y)

                # Apply second level of smoothing with moving average
                screen_x = sum(x_buffer) // len(x_buffer)
                screen_y = sum(y_buffer) // len(y_buffer)

                # Remember for next frame
                last_screen_x, last_screen_y = screen_x, screen_y

                # Draw the gaze point on the screen
                cv2.circle(screen, (screen_x, screen_y), 15, (0, 0, 255), -1)
                cv2.circle(screen, (screen_x, screen_y), 20, (0, 255, 255), 2)

                # Display screen coordinates
                cv2.putText(
                    screen,
                    f"Gaze: ({screen_x}, {screen_y})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            if len(eyes) == 0:
                cv2.putText(
                    frame,
                    "No eyes detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Show instructions
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
        cv2.imshow("Screen View", screen)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
