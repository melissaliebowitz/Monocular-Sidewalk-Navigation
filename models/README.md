Models

This directory contains the pretrained and fine-tuned models required to run the project.

Required Models
Depth Anything V2

Download the Depth Anything V2 Small checkpoint:

depth_anything_v2_vits.pth

Place the file in:

checkpoints/
└── depth_anything_v2_vits.pth

This model is used for monocular depth estimation and obstacle detection.

Fine-Tuned SegFormer Model

The project uses a fine-tuned SegFormer semantic segmentation model trained to classify:

Sidewalk
Road

Place the exported model folder in:

segformer_sidewalk_road_model/

Expected structure:

segformer_sidewalk_road_model/
├── config.json
├── preprocessor_config.json
├── model.safetensors
└── ...

This model is used to identify the sidewalk region in each frame before obstacle detection is performed.

Downloading Models

Model files are not included in this repository due to GitHub file size limitations.

Download the required models and place them in the directories shown above before running the project.

Recommended Directory Structure
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
Running the Project

After downloading the required models:

python sidewalk_navigation.py

The application will automatically load:

Depth Anything V2 for depth estimation.
SegFormer for sidewalk and road segmentation.

No additional configuration is required.
