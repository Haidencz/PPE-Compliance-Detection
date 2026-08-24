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

## Dataset

The model was trained using version 27 of the [Construction Site Safety dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27), published under the CC BY 4.0 licence.

The downloaded dataset contained:

| Split | Images |
|---|---:|
| Training | 2,603 |
| Validation | 114 |
| Test | 82 |

The ten classes are:

- Hardhat
- Mask
- NO-Hardhat
- NO-Mask
- NO-Safety Vest
- Person
- Safety Cone
- Safety Vest
- machinery
- vehicle

Before training, I checked the number of image and label files, measured the class distribution and visualised the supplied bounding-box annotations. The training split included augmented images using techniques such as mosaics, grayscale conversion and cutout.

## Model Training

I fine-tuned a pretrained YOLO11n model using an NVIDIA Tesla T4 GPU in Google Colab.

| Setting | Value |
|---|---:|
| Epochs | 50 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Random seed | 42 |
| Parameters | 2,584,102 |

The training notebook is available in `notebooks/PPE_Model_Training.ipynb`.

## Results

| Dataset | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Validation | 0.877 | 0.697 | 0.762 | 0.459 |
| Test | 0.884 | 0.640 | 0.708 | 0.421 |

The HO test results show that the model's predictions are usually precise, although its lower recall means that it still misses some objects such as a maskless woman in the image above (though it did pick out the other two maskless men). Performance was strongest for hard hats, people and safety vests, while safety cones were more difficult to localise accurately.
