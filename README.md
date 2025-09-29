# Perception for Planning

Object detection and spatial reasoning using Google's Gemini Robotics-ER 1.5 model.

## Installation

```bash
git clone https://github.com/NishanthJKumar/perception_for_planning.git
cd perception_for_planning
pip install -e .
```

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
