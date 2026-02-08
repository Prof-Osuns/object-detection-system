import streamlit as st

st.title("Test App")
st.write("If you see this, Streamlit is working")

# Test imports one by one
st.write("Testing imports...")

try:
    import PIL
    st.success("PIL imported")

except Exception as e:
    st.error(f"PIL failed: {e}")

try:
    import cv2
    st.success("OpenCV imported")
except Exception as e:
    st.error(f"OpenCV failed: {e}")

try:
    import numpy as np
    st.success("NumPy imported")
except Exception as e:
    st.error(f"NumPy failed: {e}")

try:
    from ultralytics import YOLO
    st.success("Ultralytics imported")
except Exception as e:
    st.error(f"Ultralytics failed: {e}")

try:
    model = YOLO('yolov8n.pt')
    st.success("YOLO model loaded")
except Exception as e:
    st.error(f"YOLO model failed: {e}")