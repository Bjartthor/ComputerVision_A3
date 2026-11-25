import cv2
import time
import numpy as np
import json


MODELS = [
    ("yolo26/yolo26n.onnx",     "yolo26/coco80.json"),
    ("yolo26/yolo26s.onnx",     "yolo26/coco80.json"),
    ("yolo26/pen_pencil.onnx",  "yolo26/pen_pencil.json"),
]
model_idx = 0

def load_model(i):
    path, names_file = MODELS[i]
    with open(names_file) as stream:
        names = json.load(stream)
    return path, cv2.dnn.readNetFromONNX(path), names

MODEL, net, classes = load_model(model_idx)
net = cv2.dnn.readNetFromONNX(MODEL)




cap = cv2.VideoCapture(2)

N = 61
dtime_values = np.zeros(N)
frame_count = 0
filled = False

def display_fps(frame, loc):
    global cur_time, frame_count, filled
    prev_time = cur_time
    cur_time = time.perf_counter()
    d_time = cur_time-prev_time
    dtime_values[frame_count] = d_time
    frame_count += 1
    if frame_count > N-1:
        filled = True
        frame_count = 0
    if filled:
        avg_dtime = dtime_values.mean()
    else: 
        avg_dtime = dtime_values[1:frame_count].mean() #skips the first frame
    avg_fps = 1/avg_dtime
    cv2.putText(frame, f"Avg fps: {avg_fps:.1f}", loc, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1)
    return 0

# computing letterbox once:
ret, image = cap.read()
height, width = image.shape[:2]
size = 640
gain = min(size / width, size / height)
new_width, new_height = round(width * gain), round(height * gain)


def letterbox(image):
    resized = cv2.resize(image, (new_width, new_height))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    padded = cv2.copyMakeBorder(
        resized, top, size - new_height - top, left, size - new_width - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, left, top


cur_time = time.perf_counter()
while(True):
    ret, frame = cap.read()
    if not ret:
        print("Camera stopped delivering frames")
        break
    
    padded, left, top = letterbox(frame)
    blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, swapRB=True, crop=False)

    net.setInput(blob)
    output = net.forward()

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
        cv2.rectangle(frame, p1, p2, (40, 220, 40), 2)
        cv2.putText(frame, label, (p1[0], max(p1[1] - 8, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 220, 40), 2, cv2.LINE_AA)

    display_fps(frame, (10, 30))
    cv2.imshow('frame',frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        tag = MODEL.split("/")[-1].replace(".onnx", "")
        name = f"frames/frame_{int(time.time())}_{tag}.jpg"
        ok = cv2.imwrite(name, frame)
        print("Saved" if ok else "FAILED to save", name)
    if key == ord('m'):
        model_idx = (model_idx + 1) % len(MODELS)
        MODEL, net, classes = load_model(model_idx)
        print(f"Switched to {MODEL}")
cap.release()
cv2.destroyAllWindows()