# Perception for Planning

Object detection and spatial reasoning using Google's Gemini Robotics-ER 1.5 model.

## Features

- Object detection using Gemini Vision API
- Instance segmentation using SAM2 (optional)
- 3D table segmentation using RANSAC
- 3D object mesh creation using convex hull
- Task-oriented goal predicate generation

## Installation

```bash
git clone https://github.com/NishanthJKumar/perception_for_planning.git
cd perception_for_planning
pip install -e .
```

### Dependencies

- Required:
  - numpy
  - open3d
  - pillow
  - google-generativeai
  - trimesh (for 3D mesh handling)
  
- Optional:
  - SAM2 (for segmentation)
  - curobo (for robotics planning)

## Setup

1. Get a Google API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Set your API key:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```

## Usage

Place an image named `gemini-test-img.png` in the project directory and run:

```bash
python demo_visualization.py
```

This will detect objects, classify them as movable/immovable, and generate visualizations with unique object IDs.
