from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 🔹 1. Load YOLOv8 Model (best.pt)
model = YOLO(r"/home/tanvir-rahman/Programming/ML project/bestV8.pt")  # Use raw string for file path

# 🔹 2. Load Image/Video for Prediction
source = r"/home/tanvir-rahman/Programming/ML project/images/WhatsApp Image 2025-02-04 at 1.49.19 AM.jpeg" # Use raw string for file path

# 🔹 3. Run Prediction
results = model(source, show=True)  # show=True to display results

# 🔹 4. If using OpenCV, display output
for result in results:
    img = result.plot()  # Get annotated image
    cv2.imshow("YOLOv8 Prediction", img)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows(1)  # Close OpenCV window after key press