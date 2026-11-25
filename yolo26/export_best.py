import json
from ultralytics import YOLO

model = YOLO("pen_pencil.pt")
model.export(format="onnx", imgsz=640, batch=1, dynamic=False,
             half=False, opset=17, simplify=False,
             end2end=True, nms=False, device="cpu")

# class names in the model's own order: ["pencil", "pen"]
with open("pen_pencil.json", "w") as f:
    json.dump([model.names[i] for i in range(len(model.names))], f)