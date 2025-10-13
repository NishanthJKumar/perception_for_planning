#!/usr/bin/env python3
"""
SAM Server for remote segmentation.

This script sets up a FastAPI server that exposes SAM2 (Segment Anything Model v2) functionality
as an API endpoint. This allows segmentation to run on a separate machine (potentially with
better GPU resources) while the main application runs elsewhere.
"""

import argparse
import base64
import io
import logging
import numpy as np
import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from typing import List
from pydantic import BaseModel
# Import SAM2 from the correct location
from segment_anything import build_sam
from segment_anything.predictor import SamPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_log = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="SAM Segmentation Server", description="Remote segmentation using SAM")

# Global SAM model
sam_predictor = None

def load_model_checkpoint(checkpoint_path):
    """
    Load and preprocess model checkpoint to handle different formats.
    
    Some checkpoints have the weights inside a 'model' key, others have them directly
    in the state dict. This function handles both cases.
    """
    _log.info(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # If the checkpoint has a 'model' key, extract the nested state dict
    if "model" in state_dict:
        _log.info("Found 'model' key in checkpoint, extracting nested state dict")
        state_dict = state_dict["model"]
    
    return state_dict


class SegmentationRequest(BaseModel):
    """Request model for segmentation API."""
    image_base64: str
    boxes: List[List[float]]  # List of [x0, y0, x1, y1] boxes in pixel coordinates


class SegmentationResponse(BaseModel):
    """Response model for segmentation API."""
    masks: List[List[str]]  # Base64-encoded binary masks
    scores: List[float]


@app.on_event("startup")
async def startup_event():
    """Load SAM2 model on server startup."""
    global sam_predictor
    try:
        _log.info("Loading SAM2 predictor...")
        
        model_path = os.environ.get("SAM_CHECKPOINT", "")
        model_cfg = os.environ.get("SAM_CONFIG", "")
        
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"SAM2 checkpoint not found: {model_path}")
        if not model_cfg or not os.path.exists(model_cfg):
            raise ValueError(f"SAM2 config not found: {model_cfg}")
            
        _log.info(f"Using SAM2 model from {model_path}")
        # Preprocess the checkpoint first
        state_dict = load_model_checkpoint(model_path)
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        
        # Build and load the SAM2 model
        sam_model = build_sam(model_cfg, checkpoint=None, device=device)
        # Load state dict after model is created
        sam_model.load_state_dict(state_dict)
        sam_predictor = SamPredictor(sam_model)
        _log.info("SAM2 model loaded successfully")

    except Exception as e:
        _log.error(f"Failed to load SAM2 model: {e}")
        _log.error(f"Error details: {str(e)}")
        import traceback
        _log.error(traceback.format_exc())
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if sam_predictor is None:
        raise HTTPException(status_code=503, detail="SAM2 model not loaded")
    return {"status": "healthy", "model": "SAM2 loaded"}


@app.post("/segment")
async def segment(request: SegmentationRequest) -> SegmentationResponse:
    """Segment objects in an image based on bounding boxes."""
    if sam_predictor is None:
        raise HTTPException(status_code=503, detail="SAM2 model not loaded")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Process with SAM2
        _log.info(f"Processing image of size {image.size} with {len(request.boxes)} boxes")
        
        # Set image and predict masks using SAM2
        sam_predictor.set_image(image)
        masks, scores, _ = sam_predictor.predict(
            point_coords=None, 
            point_labels=None, 
            box=np.array(request.boxes), 
            multimask_output=False
        )
        
        # Encode masks as base64 for response
        encoded_masks = []
        for mask_batch in masks:
            batch_encoded = []
            for mask in mask_batch:
                # Compress mask using run-length encoding
                mask_bytes = io.BytesIO()
                np.save(mask_bytes, mask)
                mask_bytes.seek(0)
                batch_encoded.append(base64.b64encode(mask_bytes.read()).decode())
            encoded_masks.append(batch_encoded)
        
        return SegmentationResponse(masks=encoded_masks, scores=scores.tolist())
        
    except Exception as e:
        _log.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


def main():
    """Run the SAM2 segmentation server."""
    parser = argparse.ArgumentParser(description="SAM2 Segmentation Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to SAM2 config")
    args = parser.parse_args()
    
    # Set environment variables for model loading
    os.environ["SAM_CHECKPOINT"] = args.checkpoint
    os.environ["SAM_CONFIG"] = args.config
    
    # Start server
    _log.info(f"Starting SAM2 server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()