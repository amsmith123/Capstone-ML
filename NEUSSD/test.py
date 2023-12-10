import torch
from ultralytics import YOLO

device = "0" if torch.cuda.is_available() else "cpu"

if device == 0:
    torch.cuda.set_device(1)

print("device: ", device)
model = YOLO("yolov8s-seg.pt")
print("Before: ", model.device.type)
results = model("crazing.jpg")
print("After: ", model.device.type)