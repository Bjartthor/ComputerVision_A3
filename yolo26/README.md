# YOLO26 object detection with OpenCV 5

This example uses the YOLO26 nano model trained on COCO. Export happens once using the pinned Ultralytics package. Subsequent Python inference needs only OpenCV 5 and NumPy; C++ inference needs only the OpenCV 5 C++ libraries. Neither inference program uses Ultralytics, PyTorch, or ONNX Runtime.

## C++ inference

Build using the shared `CMakeLists.txt` and instructions in the download's root README. From the extracted download root (the directory containing `CMakeLists.txt`), after exporting the model:

```sh
./build/yolo26_dnn
./build/yolo26_dnn --model=yolo26/yolo26n.onnx --image=yolo26/bus.jpg \
  --output=yolo26/yolo26_cpp_result.jpg --details=yolo26/yolo26_cpp_result.json
```

The C++17 program letterboxes to 640×640, swaps BGR to RGB, scales pixels by 1/255, and reads the end-to-end `[1, 300, 6]` output without adding NMS. It rejects an `OPENCV_FORCE_DNN_ENGINE` environment override. `--help` lists its arguments; use equals signs for named values. On Windows with Visual Studio, use `build\Release\yolo26_dnn.exe`.

`cpp-validation.json` records actual execution against libraries built from official OpenCV 5.0.0 source and a numerical comparison with the Python example on the bus photograph: the same bus and four people, box coordinates within 0.001 original-image pixels, and confidence scores within 0.00001. This is one-image execution validation, not a dataset accuracy benchmark.

## Reproduce

Use Python 3.11 or newer, with separate export and runtime environments. Run these commands from this directory. For export:

```bash
python3 -m venv .venv-export
source .venv-export/bin/activate
python -m pip install -r requirements-export.txt
python -m pip check
python export_yolo26.py
deactivate
```

Then create an inference-only environment:

```bash
python3 -m venv .venv-runtime
source .venv-runtime/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python detect_yolo26.py
python tutorial_yolo26.py
```

On Windows, activate each environment with its `Scripts/activate` script. Ultralytics declares `opencv-python` as a dependency, so the export environment uses that package. The separate inference environment contains only `opencv-python-headless` and NumPy. Do not install several packages that provide `cv2` together. The article's runtime code uses no GUI functions.

`export_yolo26.py` downloads the official weights and sample image, writes `yolo26n.onnx`, and saves the model's class labels to `coco80.json`. `detect_yolo26.py` has CLI arguments for other images and paths; `tutorial_yolo26.py` is the compact complete code for the article.

The export is FP32, ONNX opset 17, batch one, with a fixed `1 × 3 × 640 × 640` input. For **Ultralytics 8.4.27**, the explicit options are `end2end=True, nms=False`. This chooses the one-to-one head and produces a `1 × 300 × 6` output: `[x1, y1, x2, y2, confidence, class_id]`. Confidence filtering is still required; NMS is not. The inference code checks the output shape so an incompatible export fails clearly.

The input is letterboxed, converted from BGR to RGB, scaled to `[0, 1]`, and arranged as NCHW. The code removes the exact integer padding and divides by the scale to return boxes to the original image. Class IDs index the exported contiguous 80-class label list, not the sparse COCO annotation category IDs.

The current OpenCV `5.x` API calls the built-in CPU engine `cv2.dnn.ENGINE_OPENCV`; the pinned OpenCV 5.0.0 wheel calls it `cv2.dnn.ENGINE_NEW`. Both scripts prefer `ENGINE_OPENCV` when available and otherwise use `ENGINE_NEW`, passing the named constant to `readNetFromONNX`. The latest API was checked in [OpenCV's 5.x source at commit 8d126c4](https://github.com/opencv/opencv/blob/8d126c4a672936fde5a21d9133c352b18d4299a5/modules/dnn/include/opencv2/dnn/dnn.hpp); actual inference was tested with the pinned 5.0.0 wheel, not a build of that latest source.

Do not set a numeric engine override: enum values changed, and `OPENCV_FORCE_DNN_ENGINE=2` selects ONNX Runtime in the inspected latest API. The scripts reject any `OPENCV_FORCE_DNN_ENGINE` setting before importing OpenCV. If needed, run `unset OPENCV_FORCE_DNN_ENGINE` in your shell (PowerShell: `Remove-Item Env:OPENCV_FORCE_DNN_ENGINE`) and start the script again. Run these examples as standalone processes. The classic `setPreferableTarget` call does not select hardware for this engine and produces a warning in the tested build, so the example omits it.

## Verified result

To rerun the source-framework comparison, use the export environment with `requirements-export.txt` installed and the exported model, source weights, and sample image present. From this folder, run `python validate_yolo26.py` with `OPENCV_FORCE_DNN_ENGINE` unset. The validator imports the detector's named engine selection; it does not set an engine override. The local `engine_name` variable is a logging label, not an OpenCV API constant. `review-validation.json` records the corrected helper's successful rerun and override-rejection check.

On the bundled upstream `bus.jpg`, both complete inference scripts produced one bus and four people at a 0.35 confidence threshold. `yolo26_result.jpg` is the actual OpenCV output. The source weights run in PyTorch produced the same retained detections; detailed numeric agreement, shapes, exact versions, and executed commands are in `validation.json`. `export_manifest.json` records weight/model hashes and all export arguments; `opencv_build_information.txt` records the complete OpenCV build.

The export recipe was also rerun unchanged in a fresh isolated Python 3.14.3 environment, installed from `requirements-export.txt`. `pip check`, ONNX export, and the exporter’s ONNX model check all passed. `validation.json` records that independent run, and `fresh-export-freeze.txt` records all resolved packages. The inference scripts also passed in a separate minimal runtime containing only the runtime requirements.

After the engine-selection correction, both scripts were rerun in that minimal OpenCV 5.0.0 runtime and again retained five detections. The additional record is `../engine-compatibility-validation.json`; earlier validation records are preserved.

This validates this export and one real image, not COCO-wide accuracy or an inference speed claim. The first-call time in the JSON output is diagnostic and includes initial graph execution; it is not a benchmark.

## Sources and version caveat

- [YOLO26 model documentation](https://docs.ultralytics.com/models/yolo26/)
- [Pinned exporter source, Ultralytics v8.4.27](https://github.com/ultralytics/ultralytics/blob/v8.4.27/ultralytics/engine/exporter.py)
- [Pinned detection-head source, Ultralytics v8.4.27](https://github.com/ultralytics/ultralytics/blob/v8.4.27/ultralytics/nn/modules/head.py)
- [Pinned upstream bus image](https://github.com/ultralytics/ultralytics/blob/v8.4.27/ultralytics/assets/bus.jpg)
- [Official YOLO26 nano weights](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt)
- [OpenCV 5 overview](https://github.com/opencv/opencv/wiki/OpenCV-5)

The live Ultralytics documentation has since changed its export selection interface to `nms=False`; these instructions deliberately pin the earlier verified exporter that accepts `end2end=True`. Do not combine options or output assumptions from different versions. The exported model metadata identifies the model's license as AGPL-3.0; see the [Ultralytics licensing page](https://www.ultralytics.com/license) for its terms.
