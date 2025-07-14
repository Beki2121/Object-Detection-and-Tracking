import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Real-Time Object Detection and Tracking with YOLOv8 and Deep SORT")
    parser.add_argument('--source', type=str, default='0', help='Video source: 0 for webcam or path to video file')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt', help='YOLOv8 model path or name')
    return parser.parse_args()


def draw_tracks(frame, tracks, model):
    """Draw bounding boxes and class labels on the frame (green boxes, no ID)."""
    for track in tracks:
        if not track.is_confirmed():
            continue
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        # Get class id from supplementary info
        det = track.get_det_supplementary()
        if det is not None and len(det) >= 1:
            cls_id = det[0]
            label = f"{model.names[cls_id]}"
        else:
            label = "Object"
        color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


def main():
    args = parse_args()

    # Video source: webcam (0) or video file
    source = 0 if args.source == '0' else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {args.source}")
        return

    # Load YOLOv8 model
    model = YOLO(args.yolo_model)

    # Initialize Deep SORT tracker (cosine distance-based appearance model)
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, embedder="mobilenet", half=True)

    min_conf = 0.6  # Even higher minimum confidence threshold
    frame_count = 0
    detection_results = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame for maximum speed
        frame = cv2.resize(frame, (320, 180))
        frame_count += 1
        # Only run detection every 5th frame
        if frame_count % 5 == 0 or detection_results is None:
            detection_results = model(frame)
        results = detection_results
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        # Filter by highest confidence threshold
        keep = confidences > min_conf
        boxes = boxes[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]
        # Prepare detections and supplementary info for Deep SORT
        detections_for_tracker = []
        others = []
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            detections_for_tracker.append(([x1, y1, w, h], conf, int(cls_id)))
            others.append((int(cls_id), float(conf)))  # Pass class id and confidence
        # Update tracker and get tracks, passing 'others'
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame, others=others)
        # Draw detections and tracking info
        frame = draw_tracks(frame, tracks, model)
        # Resize frame for display (larger window)
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow('YOLOv8 Object Detection & Tracking', display_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 