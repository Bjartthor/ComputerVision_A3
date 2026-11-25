"""Run the accompanying NMS-free YOLO26n ONNX export using OpenCV 5 DNN."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

# Check before importing OpenCV: this variable overrides the named engine.
if "OPENCV_FORCE_DNN_ENGINE" in os.environ:
    raise RuntimeError("Unset OPENCV_FORCE_DNN_ENGINE before running this script.")
import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
# Current 5.x calls this ENGINE_OPENCV; the tested 5.0.0 wheel uses ENGINE_NEW.
# engine_name is this script's logging label, not an OpenCV API constant.
if hasattr(cv2.dnn, "ENGINE_OPENCV"):
    DNN_ENGINE, engine_name = cv2.dnn.ENGINE_OPENCV, "ENGINE_OPENCV"
elif hasattr(cv2.dnn, "ENGINE_NEW"):
    DNN_ENGINE, engine_name = cv2.dnn.ENGINE_NEW, "ENGINE_NEW"
else:
    raise RuntimeError("This example requires OpenCV 5's built-in DNN engine.")


def letterbox(image, size=640):
    """Resize with preserved aspect ratio; pad to size x size as during export."""
    height, width = image.shape[:2]
    gain = min(size / height, size / width)
    new_width, new_height = round(width * gain), round(height * gain)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    padded = cv2.copyMakeBorder(
        resized, top, size - new_height - top, left, size - new_width - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, gain, left, top


def detect(model, image, confidence=0.35):
    padded, gain, left, top = letterbox(image)
    blob = cv2.dnn.blobFromImage(padded, scalefactor=1 / 255.0, swapRB=True, crop=False)
    # Select OpenCV's built-in engine explicitly across the API rename.
    net = cv2.dnn.readNetFromONNX(str(model), DNN_ENGINE)
    # This engine currently runs on CPU; classic backend/target setters do not apply.
    net.setInput(blob)
    started = perf_counter()
    output = net.forward()
    elapsed_ms = (perf_counter() - started) * 1000
    if output.shape != (1, 300, 6):
        raise ValueError(f"Expected the accompanying (1, 300, 6) end-to-end export; got {output.shape}")
    detections = output[0][output[0, :, 4] >= confidence].copy()
    # Coordinates refer to the padded 640 x 640 input. Undo padding before scale.
    detections[:, [0, 2]] = (detections[:, [0, 2]] - left) / gain
    detections[:, [1, 3]] = (detections[:, [1, 3]] - top) / gain
    height, width = image.shape[:2]
    detections[:, [0, 2]] = detections[:, [0, 2]].clip(0, width)
    detections[:, [1, 3]] = detections[:, [1, 3]].clip(0, height)
    valid = (detections[:, 2] > detections[:, 0]) & (detections[:, 3] > detections[:, 1])
    # No NMS: this graph already contains YOLO26's one-to-one prediction path.
    return detections[valid], output, elapsed_ms


def draw(image, detections, classes):
    result = image.copy()
    for x1, y1, x2, y2, score, class_id in detections:
        class_id = int(class_id)
        label = f"{classes[class_id]} {score:.2f}"
        color = (70, 210, 70) if class_id == 0 else (255, 170, 30)
        p1 = (round(float(x1)), round(float(y1)))
        p2 = (round(float(x2)), round(float(y2)))
        cv2.rectangle(result, p1, p2, color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        tx = min(max(p1[0], 0), max(0, result.shape[1] - tw - 6))
        ty = max(p1[1], th + baseline + 6)
        cv2.rectangle(result, (tx, ty - th - baseline - 6), (tx + tw + 6, ty), color, -1)
        cv2.putText(result, label, (tx + 3, ty - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (15, 15, 15), 2, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=HERE / "yolo26n.onnx")
    parser.add_argument("--image", type=Path, default=HERE / "bus.jpg")
    parser.add_argument("--classes", type=Path, default=HERE / "coco80.json")
    parser.add_argument("--output", type=Path, default=HERE / "yolo26_result.jpg")
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    classes = json.loads(args.classes.read_text())
    detections, output, elapsed_ms = detect(args.model, image, args.confidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), draw(image, detections, classes)):
        raise OSError(f"Cannot save image: {args.output}")
    records = [{
        "class_id": int(row[5]), "class_name": classes[int(row[5])],
        "confidence": float(row[4]), "xyxy": row[:4].tolist(),
    } for row in detections]
    report = {
        "opencv": cv2.__version__, "engine": engine_name, "target": "CPU",
        "image_shape": list(image.shape), "output_shape": list(output.shape),
        "confidence_threshold": args.confidence, "detections": records,
        "first_forward_ms_not_a_benchmark": elapsed_ms,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
