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

model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
model_path = "face_landmarker.task"

def ensure_model():
    if not os.path.exists(model_path):
        print(f"Downloading model from {model_url} to {model_path}")
        urllib.request.urlretrieve(model_url, model_path)
    else:
        print(f"Model already exists at {model_path}")

def extract_landmark_image(image_path, use_gpu=False):
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
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    with FaceLandmarker.create_from_options(options) as face_landmarker:
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = face_landmarker.detect(mp_image)

    return result


def extract_landmark_dir(dir_path, use_gpu=False):
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
        num_faces=1,
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

def draw_landmarks_xy_only(image_path, result, out_path=None, draw_mesh=False, csv_path=None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
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
                        pa = (int(pts[a, 0]), int(pts[a, 1]))
                        pb = (int(pts[b, 0]), int(pts[b, 1]))
                        cv2.line(vis, pa, pb, (0, 200, 255), 1, cv2.LINE_AA)

    if out_path is None:
        base, _ = os.path.splitext(os.path.basename(image_path))
        out_dir = 'Annotation'
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
                int(has_face),        # 1 if at least one face detected, else 0
                int(total_valid)      # number of valid landmarks across faces (0~478 typically)
            ])

if __name__ == "__main__":
    parser = ArgumentParser(description='Face landmark runner')
    parser.add_argument('--is_img', action='store_true', help='is single image or directory')
    parser.add_argument('--path', type=str, required=True, help='path for single image or directory')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU delegate if tou want')
    args = parser.parse_args(sys.argv[1:])

    if args.is_img:
        # for single image landmark.
        image_path = args.path
        result = extract_landmark_image(image_path, use_gpu=args.use_gpu)
        
        draw_landmarks_xy_only(image_path, result, out_path=None, draw_mesh=False, csv_path='landmark.csv')

    else:
        # for directory image landmark.
        t0 = time.perf_counter()
        dir_path = args.path
        result_dict = extract_landmark_dir(dir_path, use_gpu=args.use_gpu)

        elapsed = time.perf_counter() - t0
        print(f"[INFO] Total elapsed: {time.perf_counter() - t0:.3f}s")

        for img_path, res in result_dict.items():
            p = pathlib.Path(img_path)
            base = p.stem
            draw_landmarks_xy_only(img_path, res, out_path=None, draw_mesh=False, csv_path='landmark.csv')
