# Eye-Controlled Mario Kart

A computer vision project that lets you play Mario Kart using just your eyes! This innovative system replaces traditional controllers with intuitive eye tracking, allowing players to steer by simply looking in the direction they want to go.

## Project Goal

The primary goal of this project is to make Mario Kart accessible to people who cannot use traditional controllers. By leveraging computer vision techniques, the system tracks eye movements and translates them into game controls, creating an intuitive and immersive gaming experience without physical input devices.

## How It Works

### Eye Tracking System

The core of the project is a sophisticated eye tracking system that uses computer vision to:

1. **Detect Your Face**: Using Haar cascades, the system first locates your face in the webcam feed.
2. **Find Your Eyes**: Within the detected face region, it identifies the position of your eyes.
3. **Track Your Pupils**: The system processes the eye regions to locate the center of your pupils.
4. **Map to Steering Controls**: The horizontal position of your gaze is mapped to steering commands - look left to turn left, look right to turn right.
5. **Detect Blinks**: Using MediaPipe's facial landmark detection, the system can also detect when you blink, which can be mapped to additional controls.

### Technical Implementation

- **Dual Computer Vision Techniques**: The system uses traditional OpenCV algorithms (Haar cascades) for efficient gaze tracking and MediaPipe for precise blink detection.
- **Advanced Smoothing**: A dual-stage smoothing algorithm ensures steering is responsive yet stable.
- **Performance Optimized**: The hierarchical detection approach (face → eyes → pupils) creates a computational funnel that drastically improves performance.
- **Real-time Processing**: All processing happens in real-time with minimal latency, providing responsive control.

## Requirements

- Python 3.7+
- Webcam
- Dolphin Emulator (for running Mario Kart)
- Required Python packages (see `requirements.txt`):
  - opencv-python
  - numpy
  - mediapipe
  - scipy
  - [other dependencies]

## Setup and Installation

1. Clone this repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Configure the Dolphin emulator path in the settings
4. Position your webcam at eye level for optimal tracking
5. Run the main application: `python -m cv_mariokart.cv_mariokart`

## Usage

1. Launch the application
2. The eye tracking system will automatically calibrate to your face
3. Look left to steer left, look right to steer right
4. Blink patterns can trigger additional actions (configurable)
5. Press 'q' to quit the application

## Tips for Best Performance

- Ensure good, consistent lighting on your face
- Position the webcam at eye level
- Avoid wearing glasses that cause glare (non-reflective glasses work fine)
- Keep a reasonable distance from the camera (50-80cm is ideal)