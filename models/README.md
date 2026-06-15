# Models

This directory contains the pretrained and fine-tuned models required to run the project.

## Required Models

### Depth Anything V2

Download the **Depth Anything V2 Small** checkpoint:

```text
depth_anything_v2_vits.pth
```

Place the file in:

```text
checkpoints/
└── depth_anything_v2_vits.pth
```

This model is used for monocular depth estimation and obstacle detection.

---

### Fine-Tuned SegFormer Model

The project uses a fine-tuned SegFormer semantic segmentation model trained to classify:

* Sidewalk
* Road

You can download from this link : https://drive.google.com/drive/folders/1JXkXLbQ3Ayr-_kCykUgtyqiy5y6qtVgQ?usp=drive_link 

Place the exported model folder in:

```text
segformer_sidewalk_road_model/
```

Expected structure:

```text
segformer_sidewalk_road_model/
├── config.json
├── preprocessor_config.json
├── model.safetensors
└── ...
```

This model is used to identify the sidewalk region in each frame before obstacle detection is performed.

---

## Download Links

The model files are hosted externally due to GitHub size limitations.

- Fine-Tuned SegFormer: [Google Drive] https://drive.google.com/drive/folders/1JXkXLbQ3Ayr-_kCykUgtyqiy5y6qtVgQ?usp=drive_link
- Depth Anything V2 Small: [Official Download] https://github.com/DepthAnything/Depth-Anything-V2



### Recommended Directory Structure

```text
project/
├── checkpoints/
│   └── depth_anything_v2_vits.pth
│
├── segformer_sidewalk_road_model/
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── model.safetensors
│   └── ...
│
├── sidewalk_navigation.py
└── README.md
```

---

## Running the Project

After downloading the required models:

```bash
python sidewalk_navigation.py
```

The application will automatically load:

* Depth Anything V2 for depth estimation.
* SegFormer for sidewalk and road segmentation.

No additional configuration is required.
