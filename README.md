# Rock-Paper-Scissors

Rock-Paper-Scissors is a simple game implemented using MediaPipe and Scikit-learn.
The game uses hand gesture recognition to play against the computer.

## Getting Started

### Prerequisites

- Python 3.x

### Setup

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the hand landmark model and place it in the project directory:

   Go to the [Hand landmarks detection guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models) page and download the `hand_landmarker.task` file.
   Place it in the project directory.
3. Run the main script:
   ```bash
   python main.py
   ```

## Usage

1. The game will start and access your webcam.
2. Teach the model to recognize your hand gestures for rock, paper, and scissors.
3. Play rock-paper-scissors against the computer by showing your hand gestures.