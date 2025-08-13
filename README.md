# face_landmarker

A minimal Python wrapper around MediaPipe’s Face Landmarker.

- Official guide: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python  
- This repository provides a simple reference implementation.

---

## 1) Requirements

```bash
python -m pip install mediapipe opencv-python
```

## 2)  Model download

You don’t need to download the '.task' file manually.
face_landmark.py already includes code that downloads the model automatically.

If you want to use the extractor directly:

```
from face_landmarker import extract_landmark
```

## 3) Output structure

'extract_landmark' returns a 'FaceLandmarkerResult':
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
> - x and y are normalized to the input image size.
> - z is a relative depth value (not pixels). Use or scale it as needed for your application.
