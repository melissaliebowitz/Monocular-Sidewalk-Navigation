# Visual Navigation for the Legally Blind Using Monocular Depth Estimation

## Overview

This project presents a vision-based navigation system designed to assist visually impaired pedestrians using a single RGB camera. The system combines semantic segmentation and monocular depth estimation to identify sidewalks, detect obstacles, and provide simple navigation instructions.

The project was developed as part of an Introduction to Navigation course and demonstrates how modern computer vision models can be used to support safe pedestrian navigation in outdoor urban environments during clear daylight conditions.

---

## Project Pipeline

The navigation pipeline consists of three main stages:

### 1. Sidewalk Detection

A custom semantic segmentation model based on SegFormer was trained to distinguish between:

* Sidewalk
* Road

Training data was created by manually labeling frames extracted from outdoor videos using SAM2. The resulting segmentation model identifies the walkable sidewalk region in each frame.

### 2. Depth Estimation

The project uses **Depth Anything V2** to generate a depth map from a single camera image.

Depth estimation provides relative distance information for objects and obstacles located within the scene without requiring stereo cameras or LiDAR sensors.

### 3. Navigation Decision

After identifying the sidewalk region, the system:

1. Extracts the sidewalk boundaries.
2. Divides the walkable area into three regions:

   * Left
   * Center
   * Right
3. Computes obstacle scores using the depth map.
4. Determines whether an obstacle blocks the current path.
5. Generates navigation instructions such as:

   * Continue forward
   * Move left
   * Move right
   * Stop

---

## Technologies Used

* Python
* OpenCV
* PyTorch
* Transformers (Hugging Face)
* SegFormer
* SAM2
* Depth Anything V2
* NumPy

---

## Dataset Creation

Custom training data was collected by recording videos of sidewalks and roads in outdoor urban environments.

Frames were extracted from the videos and annotated using SAM2.

The dataset contains pixel-level labels for:

* Sidewalk
* Road

These annotations were used to fine-tune a SegFormer semantic segmentation model.

---

## Project Structure

```text
project/
│
├── data/
│   ├── videos/
│   ├── images/
│   └── labels/
│
├── checkpoints/
│   └── depth_anything_v2_vits.pth
│
├── segformer_sidewalk_road_model/
│
├── train_segformer_sidewalk.py
├── sidewalk_navigation.py
│
└── README.md
```

---

## Running the Navigation System

1. Download the Depth Anything V2 checkpoint.
2. Place the checkpoint inside the `checkpoints/` directory.
3. Load the trained SegFormer model.
4. Run:

```bash
python sidewalk_navigation.py
```

The application displays:

* Camera frame with navigation overlay
* Segmentation results
* Depth map visualization

---

## Example Output

The system highlights the detected sidewalk region, estimates obstacle distances using Depth Anything V2, and provides navigation guidance based on the safest available direction.

Example instructions:

```text
Path clear - continue forward
Obstacle ahead - move left
Obstacle ahead - move right
No sidewalk detected - STOP
```

---

## Limitations

* Designed for clear daylight conditions.
* Performance depends on the quality of sidewalk segmentation.
* Depth Anything V2 provides relative depth rather than absolute metric distance.
* Tested primarily on sidewalks and roads represented in the training dataset.

---

## Future Work

* Audio feedback for users.
* Real-time deployment on mobile devices.
* Additional classes such as crosswalks, bicycles, pedestrians, and vehicles.
* Improved obstacle detection using object detection models.
* Support for nighttime and adverse weather conditions.

---

## Authors

Melissa Liebowitz

Ariel University – Computer Science Department
