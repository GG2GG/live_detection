# Live Object Detection & Person Classification

A real-time object detection and person classification system using YOLOv8 and ResNet50. This application provides live video analysis with object tracking and detailed person classification.

## Features

- Real-time object detection using YOLOv8
- Person classification with attributes:
  - Age groups (child, teenager, young adult, adult, elderly)
  - Gender (male, female)
  - Clothing (casual, formal, business)
  - Activity (standing, sitting, walking, running)
- Live tracking with unique ID assignment
- Support for both webcam and video file input
- Adjustable confidence thresholds
- Multiple YOLO model options (nano to xlarge)
- Real-time statistics and session history
- Detailed classification summaries

## Setup

1. Clone the repository:
```bash
git clone https://github.com/GG2GG/live_detection.git
cd live_detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLOv8 weights:
```bash
mkdir yolov8_weights
cd yolov8_weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
cd ..
```

## Usage

1. Start the server:
```bash
python backend/main.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Use the interface to:
   - Switch between webcam and video file input
   - Adjust detection confidence threshold
   - Select different YOLO models
   - Pause/play the video feed
   - View real-time statistics and classifications
   - Stop inference and view final summary

## Project Structure

```
live_detection/
├── backend/
│   └── main.py
├── frontend/
│   └── index.html
├── videos/
├── yolov8_weights/
├── requirements.txt
└── README.md
```

## Dependencies

- Python 3.8+
- FastAPI
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Uvicorn

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
