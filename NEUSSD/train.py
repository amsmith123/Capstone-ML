from ultralytics import YOLO
import os

pd = os.path.abspath(os.path.dirname(__file__))

# Load a COCO-pretrained YOLOv8n model
model = YOLO('yolov8n-seg.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data=f'{pd}/Data/data.yaml', epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format='onnx', path=f'{pd}/model.onnx')