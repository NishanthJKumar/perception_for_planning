#!/usr/bin/env python3
"""
SAM Client module that supports both local and remote segmentation.

This module provides a unified interface for using SAM segmentation, either by
running the model locally or by connecting to a remote SAM server.
"""

import base64
import io
import logging
import os
import numpy as np
import requests
import torch.cuda
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from functools import cache

# Import Replicate client for API-based segmentation
try:
    from sam_replicate_client import ReplicateClient
except ImportError:
    ReplicateClient = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_log = logging.getLogger(__name__)


class SAMClient:
    """SAM client that can operate in local, remote, or replicate API mode."""
    
    def __init__(self, mode: str = "local", server_url: str = None, api_token: str = None, **kwargs):
        """Initialize the SAM client.
        
        Args:
            mode: One of "local", "remote", or "replicate"
            server_url: URL of the SAM server (required for remote mode)
            api_token: Replicate API token (required for replicate mode)
            **kwargs: Additional arguments for local mode
        """
        self.mode = mode
        self.predictor = None
        
        if mode == "remote":
            if server_url is None:
                raise ValueError("server_url is required for remote mode")
            self.server_url = server_url
            _log.info(f"Initialized remote SAM client using server at {server_url}")
            # Test the connection
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                response.raise_for_status()
                _log.info("Successfully connected to SAM server")
            except Exception as e:
                _log.warning(f"Failed to connect to SAM server: {e}")
        
        elif mode == "replicate":
            if ReplicateClient is None:
                raise ImportError("ReplicateClient not found. Make sure replicate_client.py is in your path.")
            _log.info("Initializing Replicate API client for SAM-2")
            self.predictor = ReplicateClient(api_token)
            _log.info("Successfully initialized Replicate API client")
                
        elif mode == "local":
            self._init_local_predictor(**kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'local', 'remote', or 'replicate'")
    
    def _init_local_predictor(self, checkpoint=None, model_type="vit_h", device=None):
        """Initialize a local SAM predictor."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        try:
            # Try SAM2 first
            try:
                _log.info("Attempting to initialize SAM2...")
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                
                if checkpoint is None:
                    raise ValueError("checkpoint is required for SAM2")
                    
                config = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
                _log.info(f"Loading SAM2 with checkpoint={checkpoint}, config={config}, device={device}")
                sam_model = build_sam2(config, checkpoint, device=device)
                self.predictor = SAM2ImagePredictor(sam_model)
                self.sam_version = "sam2"
                _log.info("Successfully loaded SAM2")
                
            # Fall back to original SAM
            except ImportError:
                _log.info("SAM2 not available, falling back to original SAM...")
                from segment_anything import sam_model_registry, SamPredictor
                
                if checkpoint is None:
                    raise ValueError("checkpoint is required for SAM")
                
                _log.info(f"Loading SAM with checkpoint={checkpoint}, model_type={model_type}, device={device}")
                sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
                sam_model.to(device=device)
                self.predictor = SamPredictor(sam_model)
                self.sam_version = "sam"
                _log.info("Successfully loaded original SAM")
                
        except Exception as e:
            _log.error(f"Failed to initialize local SAM predictor: {e}")
            raise
    
    def segment(self, image: Image.Image, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment objects in an image based on bounding boxes.
        
        Args:
            image: PIL Image
            boxes: Array of shape (N, 4) with boxes in format [x0, y0, x1, y1]
            
        Returns:
            Tuple of (masks, scores)
            - masks: Array of shape (M, 1, H, W) where M may be greater than or equal to N
            - scores: Array of shape (M, 1)
            
        Note: 
            The function may return more masks than the number of input boxes.
            In this case, the caller is responsible for matching masks to boxes.
        """
        if self.mode == "remote":
            return self._segment_remote(image, boxes)
        elif self.mode == "replicate":
            return self.predictor.segment(image, boxes)
        else:
            return self._segment_local(image, boxes)
    
    def _segment_local(self, image: Image.Image, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run segmentation locally using the loaded predictor."""
        if self.predictor is None:
            raise RuntimeError("Local predictor is not initialized")
        
        # Check if we're using SAM or SAM2
        if self.sam_version == "sam2":
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                point_coords=None, 
                point_labels=None, 
                box=boxes, 
                multimask_output=False
            )
        else:  # Original SAM
            # Convert PIL to numpy
            image_np = np.array(image)
            self.predictor.set_image(image_np)
            masks, scores, _ = self.predictor.predict_torch(
                point_coords=None, 
                point_labels=None, 
                boxes=boxes, 
                multimask_output=False
            )
            # Convert torch tensors to numpy if needed
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
                scores = scores.cpu().numpy()
                
        return masks, scores
    
    def _segment_remote(self, image: Image.Image, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run segmentation by calling the remote SAM server."""
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode()
        
        # Prepare request payload
        payload = {
            "image_base64": base64_image,
            "boxes": boxes.tolist()
        }
        
        # Send request to server
        try:
            response = requests.post(
                f"{self.server_url}/segment",
                json=payload,
                timeout=30  # Longer timeout for segmentation
            )
            response.raise_for_status()
            result = response.json()
            
            # Decode masks from base64
            masks = []
            for mask_batch in result["masks"]:
                batch = []
                for mask_b64 in mask_batch:
                    mask_bytes = base64.b64decode(mask_b64)
                    mask = np.load(io.BytesIO(mask_bytes))
                    batch.append(mask)
                masks.append(batch)
                
            masks = np.array(masks)
            scores = np.array(result["scores"])
            
            return masks, scores
            
        except Exception as e:
            _log.error(f"Remote segmentation failed: {e}")
            raise RuntimeError(f"Remote segmentation failed: {str(e)}")


@cache
def get_sam_client(mode="local", server_url=None, checkpoint=None, model_type="vit_h", api_token=None):
    """Get a cached SAM client instance.
    
    Args:
        mode: One of "local", "remote", or "replicate"
        server_url: URL of the SAM server (required for remote mode)
        checkpoint: Path to SAM checkpoint (required for local mode)
        model_type: SAM model type if mode is "local"
        api_token: Replicate API token (required for replicate mode)
        
    Returns:
        A configured SAM client
    """
    return SAMClient(
        mode=mode, 
        server_url=server_url, 
        checkpoint=checkpoint,
        model_type=model_type,
        api_token=api_token
    )