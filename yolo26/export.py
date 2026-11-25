import json
from ultralytics import YOLO

#for name in ["yolo26n.pt", "yolo26s.pt"]:
#    model = YOLO(name)
#    model.info()
#    model.export(format="onnx", imgsz=640, batch=1, dynamic=False,
#                 half=False, opset=17, simplify=False,
#                 end2end=True, nms=False, device="cpu")

# class names for drawing labels (index -> name)
#names = YOLO("yolo26n.pt").names
#with open("coco80.json", "w") as f:
#    json.dump([names[i] for i in range(len(names))], f)

YOLO('yolo26n.pt').info()
YOLO('yolo26s.pt').info()