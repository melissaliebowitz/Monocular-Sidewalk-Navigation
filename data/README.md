# Data

This directory contains the data used to train and evaluate the sidewalk segmentation model.

## Dataset Overview

The dataset was created specifically for this project using videos recorded in outdoor urban environments during daylight conditions.

Frames were extracted from the videos and manually annotated using **SAM2** to generate pixel-level segmentation masks.

The dataset contains two semantic classes:

| Class ID | Class Name |
| -------- | ---------- |
| 1        | Sidewalk   |
| 2        | Road       |

These annotations were used to fine-tune a SegFormer semantic segmentation model for sidewalk detection.

---

## Directory Structure

```text
data/
├── videos/    # Original recorded videos
├── images/    # Extracted video frames
└── labels/    # Segmentation masks generated using SAM2
```

---

## Label Format

Each image in the `images/` directory has a corresponding segmentation mask in the `labels/` directory.

Example:

```text
images/
├── frame001.jpg
├── frame002.jpg
└── ...

labels/
├── frame001.png
├── frame002.png
└── ...
```

The segmentation masks use the following pixel values:

| Pixel Value | Class      |
| ----------- | ---------- |
| 0           | Background |
| 1           | Sidewalk   |
| 2           | Road       |

---

## Availability

The dataset is not included in this repository due to storage limitations.

To train the segmentation model, place the images and labels in the directory structure shown above.

---

## Data Collection Conditions

* Outdoor urban environments
* Daylight conditions
* Single RGB camera
* Sidewalk and road scenes
* Recorded specifically for sidewalk navigation experiments

The project assumes clear weather and good lighting conditions during operation.

