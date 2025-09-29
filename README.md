# Perception for Planning

A variety of useful perception utilities that tend to be useful as part of robot planning pipelines, featuring **Gemini Robotics-ER 1.5** for advanced object detection and bounding box generation.

## Overview

This repository contains implementations for object detection and spatial reasoning using Google's Gemini Robotics-ER 1.5 model. The model is specifically designed for robotics applications and provides:

- **Object Detection**: Identify and locate objects in images
- **Bounding Box Generation**: Get precise 2D bounding boxes for detected objects
- **Point Detection**: Get normalized coordinates for specific objects
- **Category-based Detection**: Detect objects by category (e.g., "fruit", "tools")
- **Spatial Reasoning**: Understand object relationships and scene context

## Features

- 🤖 **Gemini Robotics-ER 1.5 Integration**: Uses Google's latest robotics-focused vision model
- 📦 **Bounding Box Detection**: Generate precise 2D bounding boxes for objects
- 🎯 **Point Detection**: Get normalized coordinates for object locations
- 🏷️ **Smart Labeling**: Automatic object identification and labeling
- 🖼️ **Image Processing**: Built-in image resizing and optimization
- 📊 **Multiple Detection Modes**: Support for general detection and specific object queries
- 🎨 **Interactive Visualization**: Overlay bounding boxes and points on images with matplotlib
- 💾 **Save Results**: Export visualizations as high-quality PNG images
- 🌈 **Color-coded Objects**: Each detected object gets a unique color for easy identification

## Installation

### Option 1: Using pip with requirements.txt
```bash
# Clone the repository
git clone https://github.com/NishanthJKumar/perception_for_planning.git
cd perception_for_planning

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using setup.py
```bash
# Install in development mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Option 3: Install specific dependency groups
```bash
# Install with visualization capabilities
pip install -e ".[visualization]"

# Install development dependencies
pip install -e ".[dev]"
```

## Setup

1. **Get a Google API Key**:
   - Visit [Google AI Studio](https://aistudio.google.com/apikey)
   - Create a new API key for the Gemini API

2. **Set up your API key**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit the .env file and add your API key
   # OR set it as an environment variable
   export GOOGLE_API_KEY="your_api_key_here"
   ```

## Usage

### Basic Usage

Place an image named `gemini-test-img.png` in the project directory and run:

```bash
python query_gemini.py
```

### Quick Visualization Demo

For a quick demonstration of the visualization features:

```bash
python demo_visualization.py
```

### Command Line Usage (after installation)

```bash
# If installed with setup.py
gemini-detect
```

### Example Output

The script will run three detection examples with interactive visualizations:

1. **Bounding Box Detection**: Detects all objects and returns bounding boxes with overlay
2. **Specific Object Detection**: Looks for specific objects (person, chair, table, etc.) with points
3. **Category Detection**: Finds objects by category (e.g., fruit) with points

Example console output:
```
Bounding Boxes Results:
==================================================
 1. red apple
    Bounding Box: [ymin=150, xmin=200, ymax=250, xmax=300]
 2. wooden chair
    Bounding Box: [ymin=100, xmin=150, ymax=400, xmax=350]

Points Results:
==================================================
 1. person
    Point: [y=200, x=150]
 2. laptop
    Point: [y=300, x=400]

Total objects detected: 8

Generating bounding box visualization...
Visualization saved to: gemini-test-img_bounding_boxes.png

Generating point detection visualization...
Visualization saved to: gemini-test-img_points.png
```

## Visualization Features

The script automatically generates interactive visualizations for all detection results:

- **Bounding Boxes**: Colored rectangles around detected objects
- **Points**: Colored circles for point-based detections  
- **Labels**: Numbered labels with object names
- **Auto-save**: Results saved as PNG files with descriptive names
- **Interactive Display**: Matplotlib windows for zooming and inspection

### Generated Files

After running `python query_gemini.py`, you'll get:
- `gemini-test-img_bounding_boxes.png` - Image with bounding box overlays
- `gemini-test-img_points.png` - Image with point detection overlays  
- `gemini-test-img_fruit.png` - Image with fruit detection overlays

## Code Examples

### Basic Object Detection

```python
from query_gemini import setup_client, load_and_prepare_image, detect_objects_with_bounding_boxes

# Setup
client = setup_client()
image = load_and_prepare_image("your_image.jpg")

# Detect objects with bounding boxes
response = detect_objects_with_bounding_boxes(client, image)
print(response)
```

### Specific Object Detection

```python
from query_gemini import detect_specific_objects

# Look for specific objects
objects_to_find = ["bottle", "cup", "phone"]
response = detect_specific_objects(client, image, objects_to_find)
print(response)
```

## Requirements

- Python 3.8+
- Google API key for Gemini API
- Image file for testing

## Dependencies

### Core Dependencies
- `google-genai>=0.6.0` - Google GenAI SDK for Gemini API
- `Pillow>=9.0.0` - Image processing
- `requests>=2.25.0` - HTTP requests
- `numpy>=1.21.0` - Numerical computing

### Optional Dependencies
- `matplotlib>=3.5.0` - Visualization
- `opencv-python>=4.5.0` - Advanced image processing
- `seaborn>=0.11.0` - Statistical visualization

## Model Information

This project uses **Gemini Robotics-ER 1.5**, Google's robotics-focused vision-language model:

- **Model ID**: `gemini-robotics-er-1.5-preview`
- **Capabilities**: Object detection, bounding boxes, spatial reasoning
- **Input**: Images (JPEG, PNG), text prompts
- **Output**: JSON with normalized coordinates (0-1000 range)
- **Coordinates Format**: 
  - Points: `[y, x]`
  - Bounding boxes: `[ymin, xmin, ymax, xmax]`

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your `GOOGLE_API_KEY` is set correctly
2. **Import Errors**: Run `pip install -r requirements.txt` to install dependencies
3. **Image Not Found**: Ensure `gemini-test-img.png` exists in the project directory
4. **Model Access**: The model is in preview - ensure you have access to the Gemini API

### Getting Help

- Check the [Gemini API Documentation](https://ai.google.dev/gemini-api/docs/robotics-overview)
- Review the [Robotics Cookbook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/gemini-robotics-er.ipynb)
- Open an issue on this repository

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google DeepMind for the Gemini Robotics-ER 1.5 model
- Google AI for the GenAI SDK
- The robotics and AI community for continued innovation
