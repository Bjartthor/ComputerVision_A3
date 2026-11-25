"""Development validation: needs the pinned export dependencies, not just runtime."""
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Reuse the detector's named engine selection and its pre-import override check.
from detect_yolo26 import engine_name, HERE, detect, letterbox

import cv2
import numpy as np
import onnx
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

image = cv2.imread(str(HERE / "bus.jpg"))
assert image is not None
padded, gain, left, top = letterbox(image)
reference_preprocessor = LetterBox(new_shape=(640, 640), auto=False)
np.testing.assert_array_equal(padded, reference_preprocessor(image=image))
# An odd-sized landscape input exercises asymmetric vertical padding.
landscape = cv2.resize(image, (961, 721))
np.testing.assert_array_equal(letterbox(landscape)[0], reference_preprocessor(image=landscape))

graph = onnx.load(HERE / "yolo26n.onnx")
onnx.checker.check_model(graph)
input_shape = [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
assert input_shape == [1, 3, 640, 640]
classes = json.loads((HERE / "coco80.json").read_text())
model = YOLO(str(HERE / "yolo26n.pt"))
assert classes == [model.names[i] for i in range(80)]
blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, swapRB=True, crop=False)
model.model.eval()
model.model.end2end = True
with torch.inference_mode():
    reference = model.model(torch.from_numpy(blob))[0].numpy()

detections, actual, elapsed_ms = detect(HERE / "yolo26n.onnx", image)
reference_kept = reference[0][reference[0, :, 4] >= 0.35]
actual_kept = actual[0][actual[0, :, 4] >= 0.35]
np.testing.assert_array_equal(actual_kept[:, 5], reference_kept[:, 5])
np.testing.assert_allclose(actual_kept[:, :4], reference_kept[:, :4], atol=0.001, rtol=0)
np.testing.assert_allclose(actual_kept[:, 4], reference_kept[:, 4], atol=0.00001, rtol=0)
assert len(detections) == 5
assert sorted(detections[:, 5].astype(int).tolist()) == [0, 0, 0, 0, 5]

report = {
    "validated_utc": datetime.now(timezone.utc).isoformat(),
    "status": "pass",
    "python": sys.version,
    "platform": platform.platform(),
    "architecture": platform.machine(),
    "opencv": cv2.__version__,
    "opencv_module": cv2.__file__,
    "opencv_build_information": "opencv_build_information.txt",
    "numpy": np.__version__,
    "ultralytics": ultralytics.__version__,
    "torch": torch.__version__,
    "onnx": onnx.__version__,
    "engine": f"cv2.dnn.{engine_name}; CPU, no classic fallback",
    "input_shape": list(blob.shape),
    "output_shape": list(actual.shape),
    "onnx_checker": "pass",
    "preprocessing_matches_ultralytics_letterbox": {
        "bus_1080x810": "exact pixel equality",
        "odd_landscape_721x961": "exact pixel equality",
    },
    "source_framework_reference": "PyTorch source weights; identical input blob; one-to-one head",
    "retained_detection_count": len(detections),
    "retained_class_ids": actual_kept[:, 5].astype(int).tolist(),
    "max_box_absolute_error_input_pixels": float(np.abs(actual_kept[:, :4] - reference_kept[:, :4]).max()),
    "max_confidence_absolute_error": float(np.abs(actual_kept[:, 4] - reference_kept[:, 4]).max()),
    "reference_tolerances": {"box_absolute_input_pixels": 0.001, "confidence_absolute": 0.00001},
    "scope": "Single real image execution and source-framework agreement, not dataset accuracy or speed benchmarking.",
    "executed_commands": [f"{sys.executable} validate_yolo26.py"],
}
(HERE / "opencv_build_information.txt").write_text(cv2.getBuildInformation())
(HERE / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
