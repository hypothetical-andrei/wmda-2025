# Exercise 3: Object Detection with YOLOv5
from ultralytics import YOLO

# TODO

results = model("https://ultralytics.com/images/zidane.jpg")
results[0].show()  # Show image with boxes
