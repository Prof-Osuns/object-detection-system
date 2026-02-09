import streamlit as st
from PIL import Image
import cv2
import numpy as np
import json
import os
from io import BytesIO


# Set environment variable for Ultralytics
os. environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'

# Page config
st.set_page_config(
    page_title="SafetyVision - Object Detection",
    page_icon="📹",
    layout="wide"
)

from detect import ObjectDetector

# Initialize detector
@st.cache_resource
def load_detector():
    return ObjectDetector()

try:
    detector = load_detector()
except Exception as e:
    st.error(f"Error loading detector: {str(e)}")
    st.stop()

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
                try:
                    # Run detection
                    annotated, detections = detector.detect_objects(
                        image,
                        conf_threshold=conf_threshold
                    )

                    # Show results
                    with col2:
                        st.subheader("Detection Results")
                        st.image(annotated, use_container_width=True)

                    # Download buttons
                    col_d1, col_d2 = st.columns(2)

                    with col_d1:
                        buf = BytesIO()
                        Image.fromarray(annotated).save(buf, format="PNG")
                        st.download_button(
                            "Download Result",
                            buf.getvalue(),
                            "detection_result.png",
                            "image/png"
                        )

                    with col_d2:
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
                        count_items = list(counts.items())
                        num_cols = min(len(count_items), 4)
                        cols = st.columns(num_cols)

                        for i, (obj, count) in enumerate(count_items):
                            with cols[i % num_cols]:
                                st.metric(obj.title(), count)

                        # Detailed list
                        st.markdown("---")
                        st.subheader("Detailed Detections")
                        for i, det in enumerate(detections, 1):
                            st.write(
                                f"{i}. **{det['class'].title()}** -"
                                f"Confidence: {det['confidence']:.1%}"
                            )

                        # Alert system
                        if alert_objects:
                            detected_alert_objects = [d['class'] for d in detections if d['class'] in  alert_objects]
                            if detected_alert_objects:
                                st.warning(f"ALERT: Detected {','.join(set(detected_alert_objects))}")


                    else:
                        st.info("No objects detected. Try lowering the confidence threshold.")

                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
                    

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
                try:
                    output_path, detections = detector.detect_video(
                        "temp_video.mp4",
                        conf_threshold=conf_threshold
                    )
                    st.success(f"Processed! Found {len(detections)} objects")

                    # Show processed video
                    st.subheader("Results")
                    st.video(output_path)

                    # Stats
                    if detections:
                        counts = detector.count_objects(detections)
                        st.subheader("Detection Summary")

                        count_items = list(counts.items())
                        num_cols = min(len(count_items), 4)
                        cols = st.columns(num_cols)

                        for i, (obj, count) in enumerate(count_items):
                            with cols[i % num_cols]:
                                st.metric(obj.title(), count)

                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")

else: # Webcam (Live)
    st.subheader("📹Live Webcam Detection")
    st.info("Webcam feature coming soon for web deployment!")

    st.markdown("""
        **Available when running locally:**
        - Clone the [Github repo](https://github.com/Prof-Osuns/object-detection-system)
        - Run `streamlit run app.py`
        - Enoy real-time object detection!
    """)

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