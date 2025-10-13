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
# Preferably, create a new python environment (e.g. uv, conda) before
# installing this!
pip install -e .

# Set API keys
export GOOGLE_API_KEY="your_gemini_api_key"  # Required
```

## Setting up Replicate API (Recommended)

Using Replicate API for SAM-2 segmentation is the easiest approach:

1. Create a free account at [replicate.com](https://replicate.com/)
2. Generate an API token from your account settings at https://replicate.com/account/api-tokens
3. Set the environment variable: 
   ```bash
   export REPLICATE_API_TOKEN="r8_..."
   ```
4. Run the example with `--sam-mode replicate`

Note: The free tier includes some credits; after that, there's a small per-API-call cost.

## Setting up SAM2 Server (For Remote Segmentation)

If you want to run SAM2 segmentation on a separate machine (e.g., a GPU server), you can set up the SAM2 server:

### 1. Install SAM2 Server Dependencies

```bash
pip install -e ".[sam-server]"
```

### 2. Download SAM2 Model Weights

Download one of the official SAM2.1 model checkpoints from Meta:

```bash
# SAM2.1 Huge (best quality, requires most VRAM)
wget https://dl.fbaipublicfiles.com/segment_anything/sam2.1_hiera_large.pt

# SAM2.1 Large
wget https://dl.fbaipublicfiles.com/segment_anything/sam2.1_hiera_large.pt

# SAM2.1 Base Plus
wget https://dl.fbaipublicfiles.com/segment_anything/sam2.1_hiera_base_plus.pt

# SAM2.1 Small
wget https://dl.fbaipublicfiles.com/segment_anything/sam2.1_hiera_small.pt

# SAM2.1 Tiny (fastest, lowest memory)
wget https://dl.fbaipublicfiles.com/segment_anything/sam2.1_hiera_tiny.pt
```

For more checkpoint options, see the [official SAM2 repository](https://github.com/facebookresearch/segment-anything-2).

### 3. Start the SAM2 Server

```bash
# Basic usage (replace with your chosen model)
python sam_server.py --checkpoint sam2.1_hiera_large.pt --config sam2.1_hiera_l

# Custom port
python sam_server.py --checkpoint sam2.1_hiera_large.pt --config sam2.1_hiera_l --port 8000

# Bind to specific host
python sam_server.py --checkpoint sam2.1_hiera_large.pt --config sam2.1_hiera_l --host 0.0.0.0 --port 8000
```

**Config names for different models:**
- `sam2.1_hiera_l` - Large model
- `sam2.1_hiera_b+` - Base Plus model
- `sam2.1_hiera_s` - Small model
- `sam2.1_hiera_t` - Tiny model

### 4. Use the Remote Server

Once the server is running, you can use it in your application:

```bash
python examples/example_gemini.py --sam-mode remote --sam-server http://localhost:8000
```

Or if running on a different machine:

```bash
python examples/example_gemini.py --sam-mode remote --sam-server http://your-gpu-server:8000
```

## Usage Examples

```bash
# Using local SAM model
python examples/example_gemini.py --sam-mode local --sam-checkpoint /path/to/sam_vit_h.pth

# Using remote SAM server
python examples/example_gemini.py --sam-mode remote --sam-server http://localhost:8000

# Using Replicate API for SAM-2 (easiest)
python examples/example_gemini.py --sam-mode replicate

# Specify a custom task for planning
python examples/example_gemini.py --task "Place the red cube on top of the blue box"

# Disable API call caching
python examples/example_gemini.py --no-cache
```

## Output

The pipeline generates:
- Detection visualizations (`output/detection_results.png`)
- Segmentation masks (`output/segmentation_results.png`)
- 3D object meshes (`output/meshes/`)
- Task planning predicates from natural language
