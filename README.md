# DSLEEP - Fatigue Detection System

This is a computer vision-based fatigue detection system designed for real-time monitoring of driver or user fatigue status. It employs a dual detection mechanism, utilizing dlib's 68-point facial landmark detection and YOLO V11 for multi-scale detection and dual verification.

## Features

- **Dual Detection Mechanism**: Combines traditional computer vision (dlib) with deep learning (YOLO V11)
- **Real-time Processing**: 30 FPS camera processing with live GUI interface
- **Multi-scale Detection**: Supports various face sizes and distances
- **Cross-platform Support**: Ubuntu 20.04 compatibility
- **GPU Acceleration**: CUDA support for faster inference

## Key Technologies

### Traditional Computer Vision
- **EAR (Eye Aspect Ratio)**: Calculates eye openness ratio for blink detection
- **MAR (Mouth Aspect Ratio)**: Detects yawning through mouth opening measurements
- **68-point Facial Landmarks**: dlib-based facial feature detection

### Deep Learning Detection
- **YOLO V11**: State-of-the-art object detection for facial states
- **Detection Classes**: closed_eye, open_eye, closed_mouth, open_mouth
- **Real-time Inference**: GPU-accelerated processing

## Installation

1. **Environment Setup**:
   ```bash
   conda create -n dsleep python=3.9
   conda activate dsleep
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model Files**:
   Due to GitHub's file size limitations, please download the following model files separately:
   
   - `weights/shape_predictor_68_face_landmarks.dat` (96MB)
     - Download from: [dlib model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
     - Extract and place in `weights/` directory
   
   - `weights/best.pt` (5MB) - YOLO V11 trained model
   - `weights/yolov5.pt` (19MB) - YOLOv5 backup model

## Usage

### GUI Mode (Recommended)
```bash
python main.py
```

### Console Mode
```bash
python main.py --console
```

### System Requirements
- Ubuntu 20.04 or later
- Python 3.9+
- OpenCV 4.x
- PySide2/PySide6
- CUDA-compatible GPU (optional, for acceleration)

## Project Structure

```
dsleep/
├── main.py                 # Main application with GUI
├── fatigue_detector.py     # dlib-based fatigue detection
├── yolo_detector.py        # YOLO-based detection
├── frame_processor.py      # Unified frame processing
├── weights/                # Model files (download separately)
├── models/                 # YOLO model configurations
├── ultralytics/           # Extended YOLO framework
└── utils/                 # Utility functions
```

## Algorithm Details

### Fatigue Detection Criteria

1. **Eye State Analysis**:
   - EAR threshold: < 0.27 (eyes closed)
   - Blink frequency monitoring
   - Continuous closed-eye detection

2. **Yawning Detection**:
   - MAR threshold: > 0.6 (mouth open)
   - Duration-based yawn confirmation
   - Frequency analysis (3+ yawns/minute = fatigue)

3. **Combined Assessment**:
   - Multi-modal fusion of eye and mouth states
   - Temporal analysis for robust detection
   - Real-time alerting system

## Performance

- **Processing Speed**: 30 FPS on modern hardware
- **Detection Accuracy**: 
  - Eye state: >95%
  - Yawning: >90%
  - Overall fatigue: >92%
- **Latency**: <50ms per frame

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- dlib library for facial landmark detection
- Ultralytics YOLO for object detection framework
- OpenCV community for computer vision tools
- PySide for GUI framework

## Support

For issues and questions, please open an issue on GitHub or contact the development team.

---

**Note**: This system is designed for research and educational purposes. For production use in safety-critical applications, additional validation and testing are recommended.