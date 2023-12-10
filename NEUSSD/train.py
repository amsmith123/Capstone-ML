# Imports
from ultralytics import YOLO
import os

# Get directory path for the file
pd = os.path.abspath(os.path.dirname(__file__))

# Load a COCO-pretrained YOLOv8l (large) model
model = YOLO('yolov8l-seg.pt')

# Train the model using the dataset for 100 epochs
results = model.train(data=f'{pd}/Data/data.yaml', epochs=100)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format='onnx')