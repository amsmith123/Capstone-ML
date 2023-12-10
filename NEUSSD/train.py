# Imports
from ultralytics import YOLO
import os

# Get directory path for the file
pd = os.path.abspath(os.path.dirname(__file__))

# Load a COCO-pretrained YOLOv8s model
model = YOLO('yolov8s-seg.pt')

# Train the model using the dataset for 20 epochs
results = model.train(data=f'{pd}/Data/data.yaml', epochs=20)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format, save as model.onnx
success = model.export(format='onnx', path=f'{pd}/model.onnx')