import sys
import cv2
import urllib.request
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
model_path = "face_landmarker.task"

def ensure_model():
    if not os.path.exists(model_path):
        print(f"Downloading model from {model_url} to {model_path}")
        urllib.request.urlretrieve(model_url, model_path)
    else:
        print(f"Model already exists at {model_path}")

def extract_landmark(image_path):
    ensure_model()

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
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

def draw_landmarks_xy_only(image_path, result, out_path=None, draw_mesh=False):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    r = max(1, min(w, h) // 900)

    FACEMESH_TESSELATION = []

    vis = img.copy()
    if result.face_landmarks:
        for lm_list in result.face_landmarks:
            xs = np.array([lm.x for lm in lm_list], dtype=np.float32)
            ys = np.array([lm.y for lm in lm_list], dtype=np.float32)
            valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0.0) & (xs <= 1.0) & (ys >= 0.0) & (ys <= 1.0)
            pts = np.stack([ (xs * w).astype(int), (ys * h).astype(int) ], axis=1)

            idx = np.where(valid)[0]
            for i in idx:
                x, y = int(pts[i,0]), int(pts[i,1])
                cv2.circle(vis, (x, y), r, (0, 255, 0), 1, cv2.LINE_AA)

            if draw_mesh and FACEMESH_TESSELATION:
                vset = set(idx.tolist())
                for a, b in FACEMESH_TESSELATION:
                    if (a in vset) and (b in vset):
                        pa, pb = (int(pts[a,0]), int(pts[a,1])), (int(pts[b,0]), int(pts[b,1]))
                        cv2.line(vis, pa, pb, (0, 200, 255), 1, cv2.LINE_AA)

    if out_path is None:
        dir_name = os.path.dirname(image_path)
        base, _ = os.path.splitext(os.path.basename(image_path))
        out_path = os.path.join(dir_name, f"{base}_annotated.jpg")
    
    cv2.imwrite(out_path, vis)

if __name__ == "__main__":
    image_path = 'Image/lenna.png'
    result = extract_landmark(image_path)
    draw_landmarks_xy_only(image_path, result, out_path=None, draw_mesh=False)