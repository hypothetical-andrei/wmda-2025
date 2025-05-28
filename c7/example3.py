# [Show bounding boxes detecting different animals in a safari image]
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load a pre-trained YOLO model from Torch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the safari image
image_path = 'safari.jpg'  # Replace with your actual image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

# Perform object detection
results = model(image_rgb)

# Extract detection results
detections = results.pandas().xyxy[0]  # Get detections as Pandas DataFrame

# Define a list of animal categories (subset from COCO dataset)
animal_classes = ['elephant', 'zebra', 'giraffe', 'lion', 'deer', 'bear', 'horse']

# Draw bounding boxes for detected animals
for index, row in detections.iterrows():
    class_name = row['name']
    if class_name in animal_classes:
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']

        # Draw rectangle around detected animal
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Convert BGR to RGB for displaying in Matplotlib
image_rgb_output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image with bounding boxes
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb_output)
plt.axis("off")
plt.title("Animal Detection in Safari Image")
plt.savefig("output.jpg") z
