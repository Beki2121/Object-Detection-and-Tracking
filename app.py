import streamlit as st
import cv2
import tempfile
import numpy as np
from object_detection_tracking import YOLO, DeepSort, draw_tracks

# --- Custom CSS for style ---
st.markdown("""
    <style>
    .main {background-color: #f5f6fa;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stProgress > div > div > div > div {background-color: #4CAF50;}
    </style>
""", unsafe_allow_html=True)

# --- Logo and title ---
# Place your logo as 'logo.png' in the project directory, or comment out the next line if not available
# st.image("logo.png", width=120)
st.title("🚀 Real-Time Object Detection & Tracking")
st.markdown("A modern, fast, and easy-to-use tool for video and live object detection.")

# --- Sidebar for settings ---
st.sidebar.header("Settings")
option = st.sidebar.radio("Detection Mode", ("Live Webcam", "Upload Video"))
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.6, 0.05)
model_choice = st.sidebar.selectbox("YOLOv8 Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])

FRAME_SIZE = (320, 180)
DISPLAY_SIZE = (960, 540)
FRAME_SKIP = 5

# --- Load models (cache for speed) ---
@st.cache_resource
def load_models(model_choice):
    model = YOLO(model_choice)
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, embedder="mobilenet", half=True)
    return model, tracker

model, tracker = load_models(model_choice)

def process_video(video_source, model, tracker, min_conf, is_webcam=False):
    cap = cv2.VideoCapture(video_source)
    frame_count = 0
    detection_results = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
    detected_classes = set()
    with st.spinner("Processing video..."):
        while cap.isOpened():
            # Check for stop signal
            if st.session_state.get("stop_detection", False):
                break
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
            keep = confidences > min_conf
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
                detected_classes.add(model.names[cls_id])
            tracks = tracker.update_tracks(detections_for_tracker, frame=frame, others=others)
            frame = draw_tracks(frame, tracks, model)
            display_frame = cv2.resize(frame, DISPLAY_SIZE)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            # Progress bar for video files
            if not is_webcam and total_frames > 0:
                st.progress(min(frame_count / total_frames, 1.0), text=f"Frame {frame_count}/{total_frames}")
            yield display_frame, detected_classes, frame_count
    cap.release()

# --- Main UI ---
st.markdown("---")
if "stop_detection" not in st.session_state:
    st.session_state["stop_detection"] = False

if option == "Live Webcam":
    col1, col2 = st.columns([1,1])
    start = col1.button("Start Live Detection")
    stop = col2.button("Stop Detection")
    if start:
        st.session_state["stop_detection"] = False
    if stop:
        st.session_state["stop_detection"] = True
    if start and not st.session_state["stop_detection"]:
        st.info("Live detection will open in the browser below. Click Stop to end.")
        stframe = st.empty()
        stats = st.empty()
        for display_frame, detected_classes, frame_count in process_video(0, model, tracker, confidence, is_webcam=True):
            stframe.image(display_frame, channels="RGB")
            stats.markdown(f"**Frames Processed:** {frame_count}  ")
            stats.markdown(f"**Classes Detected:** {', '.join(sorted(detected_classes)) if detected_classes else 'None'}")
elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    col1, col2 = st.columns([1,1])
    start = col1.button("Start Detection on Uploaded Video")
    stop = col2.button("Stop Detection")
    if start:
        st.session_state["stop_detection"] = False
    if stop:
        st.session_state["stop_detection"] = True
    if uploaded_file is not None and start and not st.session_state["stop_detection"]:
        st.info("Detection will open in the browser below. Click Stop to end.")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        stframe = st.empty()
        stats = st.empty()
        for display_frame, detected_classes, frame_count in process_video(tfile.name, model, tracker, confidence):
            stframe.image(display_frame, channels="RGB")
            stats.markdown(f"**Frames Processed:** {frame_count}  ")
            stats.markdown(f"**Classes Detected:** {', '.join(sorted(detected_classes)) if detected_classes else 'None'}")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by **Bereket H** | [GitHub](https://github.com/Beki2121)") 