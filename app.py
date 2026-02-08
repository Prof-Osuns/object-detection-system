import streamlit as st
from PIL import Image
import cv2
import numpy as np
from detect import ObjectDetector
import av

# Page config
st.set_page_config(
    page_title="SafetyVision - Object Detection",
    page_icon="📹",
    layout="wide"
)

# Initialize detector
@st.cache_resource
def load_detector():
    return ObjectDetector()

detector = load_detector()

# Title
st.title("📹SafetyVision - Object Detection System")
st.markdown("### AI-Powered Security & Monitoring")

# Sidebar
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

st.sidebar.subheader("Alerts")
alert_objects = st.sidebar.multiselect(
    "Alert me when detected:",
    ["person", "car", "truck", "dog", "cat", "bicycle"]
)

detection_mode = st.sidebar.radio(
    "Detection Mode",
    ["Image Upload", "Video Upload", "Webcam (Live)"]
)

# Main content
if detection_mode == "Image Upload":
    st.subheader("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    if uploaded_file:
        # Load image
        image = Image.open(uploaded_file)

        # Create columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)


        # Detect button
        if st.button("Detect Objects", type="primary"):
            with st.spinner("Analyzing image..."):
                # Run detection
                annotated, detections = detector.detect_objects(
                    image,
                    conf_threshold=conf_threshold
                )

                # Show results
                with col2:
                    st.subheader("Detection Results")
                    st.image(annotated, use_container_width=True)

                import json
                # Download annotated image
                from io import BytesIO
                
                buf = BytesIO()
                Image.fromarray(annotated).save(buf, format="PNG")
                st.download_button(
                    "Download Result",
                    buf.getvalue(),
                    "detection_result.png",
                    "image/png"
                )
                # Download detection data
                st.download_button(
                    "Download Data (JSON)",
                    json.dumps(detections, indent=2),
                    "detections.json",
                    "application/json"
                    )


                # Object counts
                st.markdown("---")
                st.subheader("Detection Summary")

                if detections:
                    counts = detector.count_objects(detections)

                    # Display counts
                    cols = st.columns(len(counts))
                    for i, (obj, count) in enumerate(counts.items()):
                        with cols[i]:
                            st.metric(obj.title(), count)

                    import pandas as pd
                    import plotly.express as px

                    st.subheader("Detection Analytics")

                    df = pd.DataFrame(detections)

                    fig = px.histogram(
                        df,
                        x='class',
                        title="Object Distribution"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Detailed list
                    st.subheader("Detailed Detections")
                    for i, det in enumerate(detections, 1):
                        st.write(
                            f"{i}. **{det['class'].title()}** -"
                            f"Confidence: {det['confidence']:.1%}"
                        )
                else:
                    st.info("No objects detected. Try lowering the confidence threshold.")

                if detections and alert_objects:
                    detected_alert_objects = [d['class'] for d in detections if d['class'] in alert_objects]
                    if detected_alert_objects:
                        st.warning(f"ALERT: Detect {', '.join(set(detected_alert_objects))}")

elif detection_mode == "Video Upload":
    st.subheader("Video Detection")

    uploaded_video = st.file_uploader(
        "Upload video...",
        type=['mp4', 'avi', 'mov']
    )
    if uploaded_video:
        # Save uploaded video
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        # Show original
        st.video("temp_video.mp4")

        if st.button("Process Video", type="primary"):
            with st.spinner("Processing video... This may take a few minutes"):
                output_path, detections = detector.detect_video(
                    "temp_video.mp4",
                    conf_threshold=conf_threshold
                )
                st.success(f"Processed! Found {len(detections)} objects")

                # Show processed video
                st.subheader("Results")
                st.video(output_path)

                # Stats
                counts = detector.count_objects(detections)
                st.subheader("Detection Summary")
                for obj, count in counts.items():
                    st.metric(obj.title(), count)


else: # Webcam
    st.subheader("Live Webcam Detection")

    run_webcam = st.checkbox("Start Webcam")

    if run_webcam:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.detector = ObjectDetector()

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")

                # Detect
                results = self.detector.model(img, conf=conf_threshold)
                annotated = results[0].plot()

                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            
        webrtc_streamer(key="detection", video_processor_factory=VideoProcessor)



# Footer
st.markdown("---")
st.markdown("### Use Cases")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Security Monitoring**")
    st.write("Detect unauthorized persons or vehicles in restricted areas")

with col2:
    st.markdown("**Retail Analytics**")
    st.write("Count customers, track product placement")

with col3:
    st.markdown("**Traffic Management**")
    st.write("Vehicle counting and classification")

st.markdown("---")
st.markdown("**Tech:** YOLOv8, OpenCV, Streamlit | **Model:** Ultralytics YOLO")