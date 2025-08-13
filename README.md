# face_landmarker

A minimal Python wrapper around MediaPipe’s Face Landmarker.

- Official guide: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python  
- This repository provides a simple reference implementation.

---

## 1) Requirements

```bash
python -m pip install mediapipe opencv-python
```

## 2) Recommended project layout

Place your images in `Image/`. Annotated visualization outputs will be saved to `Annotation/`.
The model file `face_landmarker.task` is auto-downloaded to the project root on first run.

```
project_root/
├─ Image/                     # put your input images here
│  ├─ img1.jpg
│  ├─ img2.png
│  └─ ...
├─ Annotation/                # annotated outputs will be saved here (auto-created)
├─ face_landmark.py           # main script
├─ face_landmarker.task       # model file (auto-downloaded)
└─ README.md
```

## 3)  Model download

You don’t need to download the `.task` file manually.
face_landmark.py already includes code that downloads the model automatically.

If you want to use the extractor project directly:

```
from face_landmarker import extract_landmark
```

## 4) Running the Script
Single image:
```
python face_landmark.py --is_img --path Image/image_name.png
```
Directory of images:
```
python face_landmkar.py --path Image
```
Optional GPU toggle (when you want to utilize GPU)
```
python face_landmark.py --is_img --path Image/image_name.png --use_gpu
```

## 5) What gets detected (detection behavior)
- The solution includes a built-in face detector. If no face is detected in an image. no landmarks are returned for that image.
- If a face is detected, the landmarker typically returns 478 landmarks per face. (Or lower)
- If a mant face are detected, the landmarker returns landmarks for each faces.

#### Output structure

`extract_landmark` returns a `FaceLandmarkerResult`:
```
FaceLandmarkerResult
├── face_landmarks : List[List[NormalizedLandmark]]
│   ├── [face_idx]             # the number of detected face
│   │   ├── [lm_idx]           # index of landmars (typically 478 points)
│   │   │   ├── x : float      # [0.0..1.0] normalized width coordinates
│   │   │   ├── y : float      # [0.0..1.0] normalized height coordinates
│   │   │   ├── z : float      # Depth
│   │   │   ├── visibility : float  # Always 0.0
│   │   │   └── presence  : float   # Always 0.0
│   │   └── ... 
│   └── ... (Repeats for the number of detected face )
│
├── face_blendshapes : []
└── facial_transformation_matrixes : []
```
Accessing the j-th landmark of the i-th face:
```
lm = result.face_landmarks[i][j]
x, y, z = lm.x, lm,y, lm.z
```
Given an image size (W, H), convert to pixel coordinates:
```
px = int(lm.x * W)
py = int(lm.y * H)
pz = lm.z * W
```
> Notes
> - `x` and `y` are normalized to `[0, 1]` relative to the input image dimensions.
> - `z` is a relative depth value (not pixels). Use or scale it as needed for your application.

If you operate `draw_landmarks_xy_only` functions. (Operated in default),
then you can get annotated images in `/Annotation` and get `csv file` summarizing the output of landmark detecting results.
