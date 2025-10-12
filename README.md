# Perception for Planning

A collection of useful perception utility functions that are useful for model-based planning.

NOTE: this repo is currently under active development; specific functions/APIs and design patterns are likely to change!


## Key Features

- Object detection with Gemini Vision API
- Instance segmentation with SAM-2
- 3D object mesh generation via convex hull
- Automatic task planning from natural language

## Quick Start

```bash
# Install
git clone https://github.com/NishanthJKumar/perception_for_planning.git
cd perception_for_planning
pip install -e .

# Set API keys
export GOOGLE_API_KEY="your_gemini_api_key"  # Required
export REPLICATE_API_TOKEN="your_replicate_token"  # For SAM-2 API
```

## Usage Examples

```bash
# Using local SAM model
python example_gemini.py --sam-mode local --sam-checkpoint /path/to/sam_vit_h.pth

# Using remote SAM server
python example_gemini.py --sam-mode remote --sam-server http://localhost:8000

# Using Replicate API for SAM-2 (easiest)
python example_gemini.py --sam-mode replicate

# Specify a custom task for planning
python example_gemini.py --task "Place the red cube on top of the blue box"

# Disable API call caching
python example_gemini.py --no-cache
```

## Output

The pipeline generates:
- Detection visualizations (`output/detection_results.png`)
- Segmentation masks (`output/segmentation_results.png`)
- 3D object meshes (`output/meshes/`)
- Task planning predicates from natural language
