import time
import sys
import cv2
import csv
import urllib.request
import os
import pathlib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from argparse import ArgumentParser

# Landmarker model
model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
model_path = "face_landmarker.task"

# Face detector model (BlazeFace short-range)
det_model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
det_model_path = "blaze_face_short_range.tflite"

def ensure_model():
    if not os.path.exists(model_path):
        print(f"Downloading model from {model_url} to {model_path}")
        urllib.request.urlretrieve(model_url, model_path)
    else:
        print(f"Model already exists at {model_path}")

def ensure_face_model():
    if not os.path.exists(det_model_path):
        print(f"Downloading model from {det_model_url} to {det_model_path}")
        urllib.request.urlretrieve(det_model_url, det_model_path)
    else:
        print(f"Model already exists at {det_model_path}")

def extract_landmark_image(image_path, use_gpu=False, lm_det_conf=0.5):
    ensure_model()
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU,
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=10,
        min_face_detection_confidence=lm_det_conf,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    with FaceLandmarker.create_from_options(options) as face_landmarker:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = face_landmarker.detect(mp_image)

    h, w = img_bgr.shape[:2]
    faces = result.face_landmarks or []
    print("=== Detection Summary ===")
    print(f"Image: {os.path.basename(image_path)} ({w}x{h})")
    print(f"Number of faces: {len(faces)}")
    for i, lm_list in enumerate(faces):
        xs = np.array([lm.x for lm in lm_list], dtype=np.float32)
        ys = np.array([lm.y for lm in lm_list], dtype=np.float32)
        valid = (
            np.isfinite(xs) & np.isfinite(ys) &
            (xs >= 0.0) & (xs <= 1.0) &
            (ys >= 0.0) & (ys <= 1.0)
        )
        vc = int(valid.sum())
        if vc > 0:
            xpix = (xs[valid] * w).astype(int)
            ypix = (ys[valid] * h).astype(int)
            x1, y1 = int(xpix.min()), int(ypix.min())
            x2, y2 = int(xpix.max()), int(ypix.max())
            print(f" - Face #{i}: valid_landmarks={vc}, bbox_px=({x1},{y1},{x2},{y2})")
        else:
            print(f" - Face #{i}: valid_landmarks=0, bbox_px=None")

    return result

def extract_landmark_dir(dir_path, use_gpu=False, lm_det_conf=0.5):
    ensure_model()
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU,
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=10,
        min_face_detection_confidence=lm_det_conf,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    results = {}
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    dirp = pathlib.Path(dir_path)
    with FaceLandmarker.create_from_options(options) as landmarker:
        for f in sorted(dirp.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                img_bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                results[str(f)] = landmarker.detect(mp_image)
    return results

def detect_faces_image(image_path, use_gpu=False, min_conf=0.5):
    ensure_face_model()
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(
            model_asset_path=det_model_path,
            delegate=BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU
        ),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=min_conf
    )

    with FaceDetector.create_from_options(options) as detector:
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(image_path)
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = detector.detect(mp_image)

    dets = result.detections or []
    print("=== Face Detection (bbox only) ===")
    print(f"Image: {os.path.basename(image_path)} ({w}x{h})")
    print(f"Number of faces: {len(dets)}")
    for i, det in enumerate(dets):
        bb = det.bounding_box
        x1, y1 = int(bb.origin_x), int(bb.origin_y)
        x2, y2 = int(bb.origin_x + bb.width), int(bb.origin_y + bb.height)
        score = det.categories[0].score if det.categories else 0.0
        print(f" - Face #{i}: score={score:.3f}, bbox_px=({x1},{y1},{x2},{y2})")
    return result

def detect_faces_dir(dir_path, use_gpu=False, min_conf=0.5):
    ensure_face_model()
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(
            model_asset_path=det_model_path,
            delegate=BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU
        ),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=min_conf
    )

    results = {}
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    dirp = pathlib.Path(dir_path)
    with FaceDetector.create_from_options(options) as detector:
        for f in sorted(dirp.iterdir()):
            if not (f.is_file() and f.suffix.lower() in exts):
                continue
            img_bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            res = detector.detect(mp_image)
            results[str(f)] = res
            
            draw_face_detections(str(f), res, out_path=None)
    return results

def draw_face_detections(image_path, detections, out_path=None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    vis = img.copy()
    dets = detections.detections or []
    for det in dets:
        bb = det.bounding_box
        x1, y1 = int(bb.origin_x), int(bb.origin_y)
        x2, y2 = int(bb.origin_x + bb.width), int(bb.origin_y + bb.height)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2, cv2.LINE_AA)
        sc = det.categories[0].score if det.categories else 0.0
        txt = f"{sc:.2f}"
        cv2.putText(vis, txt, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    if out_path is None:
        base, _ = os.path.splitext(os.path.basename(image_path))
        out_dir = 'MediaPipe_Annotation'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}_facedet.jpg")
    cv2.imwrite(out_path, vis)

def draw_landmarks_xy_only(image_path, result, out_path=None, draw_mesh=False, csv_path=None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w = img.shape[:2]
    r = max(1, min(w, h) // 900)
    FACEMESH_TESSELATION = []
    vis = img.copy()
    has_face = bool(getattr(result, "face_landmarks", None))
    total_valid = 0
    if has_face:
        for lm_list in result.face_landmarks:
            xs = np.array([lm.x for lm in lm_list], dtype=np.float32)
            ys = np.array([lm.y for lm in lm_list], dtype=np.float32)
            valid = (
                np.isfinite(xs) & np.isfinite(ys) &
                (xs >= 0.0) & (xs <= 1.0) &
                (ys >= 0.0) & (ys <= 1.0)
            )
            total_valid += int(valid.sum())
            pts = np.stack([(xs * w).astype(int), (ys * h).astype(int)], axis=1)
            idx = np.where(valid)[0]
            for i in idx:
                x, y = int(pts[i, 0]), int(pts[i, 1])
                cv2.circle(vis, (x, y), r, (0, 255, 0), 1, cv2.LINE_AA)
            if draw_mesh and FACEMESH_TESSELATION:
                vset = set(idx.tolist())
                for a, b in FACEMESH_TESSELATION:
                    if (a in vset) and (b in vset):
                        pa = (int(pts[a, 0]), (pts[a, 1]))
                        pb = (int(pts[b, 0]), (pts[b, 1]))
                        cv2.line(vis, pa, pb, (0, 200, 255), 1, cv2.LINE_AA)

    if out_path is None:
        base, _ = os.path.splitext(os.path.basename(image_path))
        out_dir = 'MediaPipe_Annotation'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}_annotated.jpg")
    cv2.imwrite(out_path, vis)

    if csv_path is not None:
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        new_file = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["image_name", "is_there_face", "numbers_of_feature"])
            writer.writerow([
                os.path.basename(image_path),
                int(has_face),
                int(total_valid)
            ])

if __name__ == "__main__":
    parser = ArgumentParser(description='Face landmark runner')
    parser.add_argument('--is_img', action='store_true', help='is single image or directory')
    parser.add_argument('--path', type=str, required=True, help='path for single image or directory')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU delegate if you want')
    parser.add_argument('--only_face', action='store_true', help='Detect face bbox only (no landmarks)')
    parser.add_argument('--det_conf', type=float, default=0.5, help='min detection confidence for FaceDetector')
    parser.add_argument('--lm_det_conf', type=float, default=0.5, help='min face detection confidence in FaceLandmarker')
    args = parser.parse_args(sys.argv[1:])

    if args.is_img:
        image_path = args.path
        if args.only_face:
            det = detect_faces_image(image_path, use_gpu=args.use_gpu, min_conf=args.det_conf)
            draw_face_detections(image_path, det, out_path=None)
        else:
            result = extract_landmark_image(image_path, use_gpu=args.use_gpu, lm_det_conf=args.lm_det_conf)
            draw_landmarks_xy_only(image_path, result, out_path=None, draw_mesh=False, csv_path='landmark.csv')
    else:
        t0 = time.perf_counter()
        dir_path = args.path
        if args.only_face:
            _ = detect_faces_dir(dir_path, use_gpu=args.use_gpu, min_conf=args.det_conf)
        else:
            result_dict = extract_landmark_dir(dir_path, use_gpu=args.use_gpu, lm_det_conf=args.lm_det_conf)
            for img_path, res in result_dict.items():
                draw_landmarks_xy_only(img_path, res, out_path=None, draw_mesh=False, csv_path='landmark.csv')
        elapsed = time.perf_counter() - t0
        print(f"[INFO] Total elapsed: {elapsed:.3f}s")
