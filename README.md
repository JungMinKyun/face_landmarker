# face_landmarker

### 0. You can follow whole codes easily by referring below website.
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
(This repository is simple implementation)

### 1. Needed dependencies
```
python -m pip install mediapipe opencv-python
```

### 2. Download face_landmark.task from here or url.
In face_landmark.py, there's an download codes already. \\
Without download from this repository, code is available.

```
from face_landmarker import extract_landmark
```

### 3. Structures of Outputs.
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
For the "result", output of the ==extract_landmark== \\
j-th landmark of i-th face is represented below,
```
lm = result.face_landmarks[i][j]
lm.x, lm,y, lm.z
```
And, given an image of size (W, H):
```
px = int(lm.x * W)
py = int(lm.y * H)
pz = lm.z * W
```
