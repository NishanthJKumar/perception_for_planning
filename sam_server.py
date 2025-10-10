#!/usr/bin/env python3
"""
SAM Server for remote segmentation.

This script sets up a FastAPI server that exposes SAM (Segment Anything Model) functionality
as an API endpoint. This allows segmentation to run on a separate machine (potentially with
better GPU resources) while the main application runs elsewhere.

Usage:
    # Start the server
    python sam_server.py --port 8000 --checkpoint /path/to/sam_model.pth
"""

import argparse
import base64
import io
import logging
import numpy as np
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
from typing import Dict, List, Optional, Union
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_log = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="SAM Segmentation Server", description="Remote segmentation using SAM")

# Global SAM model
sam_predictor = None


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
    """Load SAM model on server startup."""
    global sam_predictor
    try:
        _log.info("Loading SAM predictor...")
        
        # Try to import SAM2 first (newer version)
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            model_path = os.environ.get("SAM_CHECKPOINT", "")
            model_cfg = os.environ.get("SAM_CONFIG", "")
            
            if not model_path or not os.path.exists(model_path):
                raise ValueError(f"SAM2 checkpoint not found: {model_path}")
            if not model_cfg or not os.path.exists(model_cfg):
                raise ValueError(f"SAM2 config not found: {model_cfg}")
                
            _log.info(f"Using SAM2 model from {model_path}")
            sam_model = build_sam2(model_cfg, model_path, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
            sam_predictor = SAM2ImagePredictor(sam_model)
            _log.info("SAM2 model loaded successfully")
            
        # Fall back to original SAM if SAM2 not available
        except ImportError:
            _log.info("SAM2 not available, trying original SAM...")
            from segment_anything import sam_model_registry, SamPredictor
            
            model_path = os.environ.get("SAM_CHECKPOINT", "")
            model_type = os.environ.get("SAM_MODEL_TYPE", "vit_h")
            
            if not model_path or not os.path.exists(model_path):
                raise ValueError(f"SAM checkpoint not found: {model_path}")
                
            _log.info(f"Using original SAM model from {model_path}")
            sam_model = sam_model_registry[model_type](checkpoint=model_path)
            device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
            sam_model.to(device=device)
            sam_predictor = SamPredictor(sam_model)
            _log.info("Original SAM model loaded successfully")
            
    except Exception as e:
        _log.error(f"Failed to load SAM model: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if sam_predictor is None:
        raise HTTPException(status_code=503, detail="SAM model not loaded")
    return {"status": "healthy", "model": "loaded"}


@app.post("/segment")
async def segment(request: SegmentationRequest) -> SegmentationResponse:
    """Segment objects in an image based on bounding boxes."""
    if sam_predictor is None:
        raise HTTPException(status_code=503, detail="SAM model not loaded")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Process with SAM
        _log.info(f"Processing image of size {image.size} with {len(request.boxes)} boxes")
        
        # Check if we're using SAM2 or original SAM
        if hasattr(sam_predictor, "set_image"):  # SAM2 interface
            sam_predictor.set_image(image)
            masks, scores, _ = sam_predictor.predict(
                point_coords=None, 
                point_labels=None, 
                box=np.array(request.boxes), 
                multimask_output=False
            )
        else:  # Original SAM interface
            sam_predictor.set_image(np.array(image))
            masks, scores, _ = sam_predictor.predict_torch(
                point_coords=None, 
                point_labels=None, 
                boxes=np.array(request.boxes), 
                multimask_output=False
            )
            # Convert from torch to numpy if needed
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
                scores = scores.cpu().numpy()
        
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
    """Run the SAM segmentation server."""
    parser = argparse.ArgumentParser(description="SAM Segmentation Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM checkpoint")
    parser.add_argument("--config", type=str, help="Path to SAM2 config (if using SAM2)")
    parser.add_argument("--model-type", type=str, default="vit_h", 
                       help="Model type for original SAM (ignored for SAM2)")
    args = parser.parse_args()
    
    # Set environment variables for model loading
    os.environ["SAM_CHECKPOINT"] = args.checkpoint
    if args.config:
        os.environ["SAM_CONFIG"] = args.config
    os.environ["SAM_MODEL_TYPE"] = args.model_type
    
    # Start server
    _log.info(f"Starting SAM server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()