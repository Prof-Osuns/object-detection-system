from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

class ObjectDetector:
    def __init__(self):
        # Load pre-trained YOLOv8 model
        self.model = YOLO('yolov8n.pt') # 'n' = nano (fastest)

    def detect_objects(self, image, conf_threshold=0.5):
        """
        Detect objects in an image

        Args:
            image: PIL Image or numpy array
            conf_threshold: Confidence threshold (0-1)

        Returns:
               annotated_image, detections_list
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Run detection
        results = self.model(image, conf=conf_threshold)

        # Get annotated image
        annotated = results[0].plot()

        # Extract detection info
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)

        return annotated, detections
    
    def count_objects(self, detections):
        """Count objects by class"""
        counts = {}
        for det in detections:
            cls = det['class']
            counts[cls] = counts.get(cls, 0) + 1
        return counts

def detect_video(self, video_path, conf_threashold=0.5, save_path='output.mp4'):
    """Process video file"""
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    frame_count = 0
    total_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        results = self.model(frame, conf=conf_threashold)
        annotated = results[0].plot()

        # Save frame
        out.write(annotated)

        # Track detections
        for result in results:
            for box in result.boxes:
                total_detections.append({
                    'frame': frame_count,
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf)
                })
        
        frame_count += 1

    cap.release()
    out.release()
    
    return save_path, total_detections