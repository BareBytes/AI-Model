# AI-Model
Documentation for AI models

## YOLOv10
YOLOv10, built on the Ultralytics Python package, a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10 achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales.

### Cv2 Installation
*To import cv2, first install OpenCV library. The following command will install the OpenCV Library*

>pip install opencv-python

### Ultralytics Installation
*To install the ultralytics Package from PyPI. The following command will install the ultralytics*

>pip install ultralytics

*To install the ultralytics package from GitHub. The following command will install ultralytics from GitHub*

>pip install git+https://github.com/ultralytics/ultralytics.git@main

### Installing Yolo10 Model
*To install YOLO10 for real time end-to-end object detection. It aims to improve both the performance and efficiency of YOLOs and optimizing model architecture comprehensively*

>git clone https://github.com/THU-MIG/yolov10.git


__We are using webcam for the real time capturing of the guard. But if we want to switch it to a video based capturing then__
*uncomment below two codes from the login file. (Line No: 39,40 and 100, 101)*

>video_path = './Security_Guard_Footage.mp4'
>cap = cv2.VideoCapture(video_path)

__And__
*comment down below code from the login file.(Line No: 38 and 99)*

>cap = cv2.VideoCapture(0)
