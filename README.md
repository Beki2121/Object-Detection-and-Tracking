# Real-Time Object Detection and Tracking System

This project implements a real-time object detection and tracking system using YOLOv8 (Ultralytics) for detection and Deep SORT for tracking.

## Features
- Real-time object detection using YOLOv8
- Object tracking with Deep SORT (cosine distance-based appearance model)
- Works with webcam or video file input
- Displays bounding boxes, class labels, and unique tracking IDs

## Setup Instructions

### 1. Python Environment
- Ensure you have **Python 3.8+** installed.
- (Recommended) Create a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

### 2. Install Dependencies
- Install all required packages:
  ```bash
  pip install -r requirements.txt
  ```

### 3. Download YOLOv8 Model
- The script will automatically download the YOLOv8 model weights (e.g., `yolov8n.pt`) on first run using the Ultralytics API.
- You can also manually download from [Ultralytics YOLOv8 Releases](https://github.com/ultralytics/ultralytics/releases) if needed.

### 4. Run the Script
- For webcam input:
  ```bash
  python object_detection_tracking.py --source 0
  ```
- For video file input:
  ```bash
  python object_detection_tracking.py --source path/to/video.mp4
  ```

## Notes
- Deep SORT will be integrated with a cosine distance-based appearance model.
- The script is modular and well-commented for easy understanding and extension.

## References
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Deep SORT Paper](https://arxiv.org/abs/1703.07402)
- [Deep SORT Official Repo](https://github.com/nwojke/deep_sort) 

to test:
 saved video:-python object_detection_tracking.py --source "C:/Users/PC/Desktop/Desktop/test.mp4"
 Live:- python object_detection_tracking.py --source 0