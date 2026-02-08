# SafetyVision - AI Object Detection System

Real-time object detection system powered by YOLOv8 for security monitoring and analytics.

## Live Demo
[Try it here](link)

## Business Applications

**Securitiy & Monitoring:**
- Detect unauthorized persons in restricted areas
- Vehicle tracking and counting
- Perimeter breach detection

**Retail Analytics:**
- Customer counting and flow analysis
- Product placement monitoring
- Queue management

**Traffic Management**
- Vehicle classification and counting
- Parking lot monitoring
- Traffic flow analysis

## Features
- **Image Detection:** Upload photos for instant object detection
- **Video Processing:** Analyze security footage frame-by-frame
- **Real-time Webcam:** Live detection from camera feed
- **Custom Alerts:** Get notified when specific objects are detected
- **Analytics Dashboard:** Visualize detection statistics
- **Export Results:** Download annotated images and detection data

## Model Performance

- **Model:** YOLOv8 Nano (fastest, 80 object classes)
- **Speed:** ~100 FPS on GPU, ~30 FPS on CPU
- **Accuracy:** 90%+ on common obejects (people, vehicles)
- **Classes:** Detects 80 objects including people, vehicles, animals, furniture

## Technical Stack

- **Model:** Ultralytics YOLOv8
- **CV Library:** Open CV
- **Framework:** Streamlit
- **Core:** Python, NumPy, PIL

## Run Locally

```bash
git clone https://github.com/Prof-Osuns/object-detection-system.git
cd object-detection-system
pip install -r requirements.txt
streamlit run app.py
```

## What I Learned

- Computer vision fundamentals with YOLO
- Real-time object detection pipeline
- Video processing and frame analysis
- Deploying CV models as web applications
- Balancing speed vs accuracy in production

## Future Enhancements

- Custom model training for specific use cases
- Multi-camera support
- Cloud storage integration
- Mobile app version
- Real-time alerting via SMS/email

## Author
Ayomikun Osunseyi - ML Engineer
[Github](link) | [LinkedIn](link)