from detect import ObjectDetector
from PIL import Image
import requests
from io import BytesIO

detector = ObjectDetector()

# Download a Sample image
url = "https://ultralytics.com/images/bus.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img.save('test.jpg')

print("Test image downloaded!")

test_img = Image.open('test.jpg')
annotated, dets = detector.detect_objects(test_img)

print(f"Found {len(dets)} objects")
for det in dets:
    print(f"- {det['class']}: {det['confidence']:.2f}")