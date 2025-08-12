# Model Weights Directory

This directory contains the pre-trained model files required for the DSLEEP fatigue detection system.

## Required Files

### 1. shape_predictor_68_face_landmarks.dat (96MB)
- **Purpose**: dlib 68-point facial landmark detection model
- **Download**: [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- **Instructions**:
  1. Download the .bz2 file
  2. Extract: `bunzip2 shape_predictor_68_face_landmarks.dat.bz2`
  3. Place in this `weights/` directory

### 2. best.pt (5MB)
- **Purpose**: Custom trained YOLO V11 model for facial state detection
- **Classes**: closed_eye, open_eye, closed_mouth, open_mouth
- **Note**: This is the primary detection model

### 3. yolov5.pt (19MB) 
- **Purpose**: Backup YOLOv5 model
- **Download**: Available from Ultralytics or train your own
- **Usage**: Fallback option if best.pt is not available

## Model Information

| File | Size | Format | Framework | Purpose |
|------|------|---------|-----------|---------|
| shape_predictor_68_face_landmarks.dat | 96MB | .dat | dlib | Facial landmarks |
| best.pt | 5MB | .pt | PyTorch/YOLO | Facial state detection |
| yolov5.pt | 19MB | .pt | PyTorch/YOLO | Backup detection |

## Installation Steps

1. Create this directory if it doesn't exist:
   ```bash
   mkdir -p weights
   ```

2. Download the dlib model:
   ```bash
   cd weights
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```

3. Place your trained YOLO models (best.pt, yolov5.pt) in this directory

## Verification

After downloading, verify the files:
```bash
ls -lh weights/
# Should show:
# shape_predictor_68_face_landmarks.dat (≈96MB)
# best.pt (≈5MB)  
# yolov5.pt (≈19MB)
```

## Notes

- These model files are excluded from Git tracking due to size limitations
- The system will automatically detect and load available models
- GPU acceleration is recommended for YOLO models
- dlib model works on both CPU and GPU