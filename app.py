import streamlit as st
import cv2
import tempfile
import numpy as np
from object_detection_tracking import YOLO, DeepSort, draw_tracks

st.title("Real-Time Object Detection & Tracking")

option = st.radio("Choose detection mode:", ("Live Webcam", "Upload Video"))

# Detection parameters (match your optimized settings)
FRAME_SIZE = (320, 180)
DISPLAY_SIZE = (960, 540)
MIN_CONF = 0.6
FRAME_SKIP = 5

@st.cache_resource
def load_models():
    model = YOLO('yolov8n.pt')
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, embedder="mobilenet", half=True)
    return model, tracker

model, tracker = load_models()

def process_video(video_source, is_webcam=False):
    cap = cv2.VideoCapture(video_source)
    frame_count = 0
    detection_results = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frame_count += 1
        if frame_count % FRAME_SKIP == 0 or detection_results is None:
            detection_results = model(frame)
        results = detection_results
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        keep = confidences > MIN_CONF
        boxes = boxes[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]
        detections_for_tracker = []
        others = []
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            detections_for_tracker.append(([x1, y1, w, h], conf, int(cls_id)))
            others.append((int(cls_id), float(conf)))
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame, others=others)
        frame = draw_tracks(frame, tracks, model)
        display_frame = cv2.resize(frame, DISPLAY_SIZE)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        yield display_frame
    cap.release()

if option == "Live Webcam":
    if st.button("Start Live Detection"):
        stframe = st.empty()
        for display_frame in process_video(0, is_webcam=True):
            stframe.image(display_frame, channels="RGB")
elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        if st.button("Start Detection on Uploaded Video"):
            stframe = st.empty()
            for display_frame in process_video(tfile.name):
                stframe.image(display_frame, channels="RGB") 