# Models

This directory contains the pretrained and fine-tuned models required to run the navigation system.

## Required Models
### 1. Depth Anything V2

Download the Depth Anything V2 Small checkpoint:

depth_anything_v2_vits.pth

Place it in:
 '''text 
checkpoints/
└── depth_anything_v2_vits.pth
''' 

This model is used to generate a depth map from a single RGB image.

### 2. Fine-Tuned SegFormer Model

The project uses a SegFormer semantic segmentation model trained to distinguish between:

Sidewalk
Road

The model was trained using frames extracted from custom sidewalk videos and labeled with SAM2.

Place the exported model folder in:

segformer_sidewalk_road_model/

### Expected structure:

segformer_sidewalk_road_model/
├── config.json
├── preprocessor_config.json
├── model.safetensors
└── ...
## Download Links

Due to GitHub file size limitations, model files are not included in this repository.

Download the required models and place them in the locations described above before running the project.

## Usage

After downloading the models, run:

python sidewalk_navigation.py

The navigation system will automatically load:

Depth Anything V2 for depth estimation.
SegFormer for sidewalk and road segmentation.

No additional configuration is required.
