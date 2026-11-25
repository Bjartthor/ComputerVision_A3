"""Article example: run after export_yolo26.py, from this directory."""
import json
import os

# Check before importing OpenCV: this variable overrides the named engine.
if "OPENCV_FORCE_DNN_ENGINE" in os.environ:
    raise RuntimeError("Unset OPENCV_FORCE_DNN_ENGINE before running this script.")
import cv2
import numpy as np

# Current 5.x calls this ENGINE_OPENCV; the tested 5.0.0 wheel uses ENGINE_NEW.
if hasattr(cv2.dnn, "ENGINE_OPENCV"):
    DNN_ENGINE = cv2.dnn.ENGINE_OPENCV
elif hasattr(cv2.dnn, "ENGINE_NEW"):
    DNN_ENGINE = cv2.dnn.ENGINE_NEW
else:
    raise RuntimeError("This example requires OpenCV 5's built-in DNN engine.")

image = cv2.imread("bus.jpg")
if image is None:
    raise FileNotFoundError("Run export_yolo26.py to download bus.jpg first.")
with open("coco80.json") as stream:
    classes = json.load(stream)

# Preserve aspect ratio, then pad to the exported model's fixed input size.
height, width = image.shape[:2]
size = 640
gain = min(size / width, size / height)
new_width, new_height = round(width * gain), round(height * gain)
resized = cv2.resize(image, (new_width, new_height))
left, top = (size - new_width) // 2, (size - new_height) // 2
padded = cv2.copyMakeBorder(
    resized, top, size - new_height - top, left, size - new_width - left,
    cv2.BORDER_CONSTANT, value=(114, 114, 114),
)

# The model expects RGB float32 pixels in [0, 1], in NCHW order.
blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, swapRB=True, crop=False)
net = cv2.dnn.readNetFromONNX("yolo26n.onnx", DNN_ENGINE)
# The selected built-in OpenCV engine currently runs on CPU.
net.setInput(blob)
output = net.forward()
if output.shape != (1, 300, 6):
    raise ValueError(f"Use the accompanying end-to-end export; got {output.shape}")

# Each row is x1, y1, x2, y2, confidence, class_id. No NMS is needed.
detections = output[0][output[0, :, 4] >= 0.35].copy()
detections[:, [0, 2]] = (detections[:, [0, 2]] - left) / gain
detections[:, [1, 3]] = (detections[:, [1, 3]] - top) / gain
detections[:, [0, 2]] = detections[:, [0, 2]].clip(0, width)
detections[:, [1, 3]] = detections[:, [1, 3]].clip(0, height)

for x1, y1, x2, y2, score, class_id in detections:
    if x2 <= x1 or y2 <= y1:
        continue
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    label = f"{classes[int(class_id)]} {score:.2f}"
    cv2.rectangle(image, p1, p2, (40, 220, 40), 2)
    cv2.putText(image, label, (p1[0], max(p1[1] - 8, 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 220, 40), 2, cv2.LINE_AA)
    print(label)

if not cv2.imwrite("yolo26_tutorial_result.jpg", image):
    raise OSError("Could not write output image.")
