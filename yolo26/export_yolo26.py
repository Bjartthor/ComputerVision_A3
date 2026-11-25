"""Export the exact fixed-size YOLO26n graph used by detect_yolo26.py.

Export dependencies: ultralytics==8.4.27, torch==2.10.0, onnx==1.20.1.
The inference script itself only needs OpenCV 5 and NumPy.
"""

import hashlib
import json
import os
from pathlib import Path
import urllib.request

os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import onnx
import torch
import ultralytics
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
IMAGE_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.27/ultralytics/assets/bus.jpg"
EXPORT_ARGS = dict(
    format="onnx", imgsz=640, batch=1, dynamic=False, half=False,
    opset=17, simplify=False, end2end=True, nms=False, device="cpu",
)


def download(url, destination):
    if not destination.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            destination.write_bytes(response.read())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if ultralytics.__version__ != "8.4.27":
        raise RuntimeError("Use ultralytics==8.4.27 for this version-pinned export recipe.")
    weights = HERE / "yolo26n.pt"
    download(WEIGHTS_URL, weights)
    download(IMAGE_URL, HERE / "bus.jpg")
    model = YOLO(str(weights))
    onnx_path = Path(model.export(**EXPORT_ARGS))
    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph)
    classes = [model.names[index] for index in range(len(model.names))]
    (HERE / "coco80.json").write_text(json.dumps(classes, indent=2) + "\n")
    (HERE / "export_manifest.json").write_text(json.dumps({
        "weights_url": WEIGHTS_URL,
        "weights_sha256": sha256(weights),
        "image_url": IMAGE_URL,
        "image_sha256": sha256(HERE / "bus.jpg"),
        "onnx_sha256": sha256(onnx_path),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "export_args": EXPORT_ARGS,
        "class_mapping": "coco80.json; contiguous class IDs 0..79 from model.names",
        "input_shape": [1, 3, 640, 640],
        "output_shape": [1, 300, 6],
        "output_columns": ["x1", "y1", "x2", "y2", "confidence", "class_id"],
    }, indent=2) + "\n")
    print(f"Exported {onnx_path}")


if __name__ == "__main__":
    main()
