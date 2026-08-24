# PPE Compliance Detection

A computer vision application that detects personal protective equipment and potential safety violations in construction-site images.

## Overview

This project uses a fine-tuned YOLO11n object-detection model to identify workers, safety equipment and missing PPE. Users can upload an image through a Streamlit interface, view the model's bounding-box predictions and adjust the minimum confidence threshold.

I built the project to practise transfer learning, object detection, dataset analysis and model evaluation using a practical workplace-safety problem.

## Features

- Detects ten construction-site object classes
- Identifies hard hats, masks and safety vests
- Flags missing PPE
- Displays bounding boxes and confidence scores
- Summarises detections by class
- Provides an adjustable confidence threshold
- Supports JPG, JPEG and PNG uploads
