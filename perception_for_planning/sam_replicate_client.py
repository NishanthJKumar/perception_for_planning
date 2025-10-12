#!/usr/bin/env python3
"""
Replicate API client for SAM-2 segmentation.

This module provides functions to interact with the Replicate API for SAM-2 segmentation.
"""

import base64
import io
import logging
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import requests
import replicate
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_log = logging.getLogger(__name__)

class ReplicateClient:
    """Client for interacting with Replicate's SAM API."""
    
    def __init__(self, api_token: Optional[str] = None):
        """Initialize the Replicate client.
        
        Args:
            api_token: Replicate API token. If None, will try to get from environment.
        """
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "REPLICATE_API_TOKEN is required. Get your API token from https://replicate.com/account/api-tokens"
            )
        
        # Set the API token in the environment if it was passed explicitly
        if api_token:
            os.environ["REPLICATE_API_TOKEN"] = api_token
            
        _log.info("Initialized Replicate API client")
    
    def segment(self, image: Image.Image, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment objects in an image based on bounding boxes using Replicate's SAM API.
        
        Args:
            image: PIL Image
            boxes: Array of shape (N, 4) with boxes in format [x0, y0, x1, y1]
            
        Returns:
            Tuple of (masks, scores)
            - masks: Array of shape (N, 1, H, W)
            - scores: Array of shape (N, 1)
        """
        # Prepare inputs for the API call
        _log.info(f"Processing {len(boxes)} boxes with Replicate API in a single call")
        
        # Save image to a buffer
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)  # Reset buffer position to the beginning
        
        # Format all bounding boxes into a single prompt
        box_prompts = []
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            box_prompts.append(f"box: {x0},{y0},{x1},{y1}")
        
        # Combine all box prompts with newlines
        prompt = "\n".join(box_prompts)
        
        # Prepare input for replicate with image as file-like object
        input_data = {
            "image": buffer,
            "prompt": prompt
        }
        
        # Initialize results arrays
        results = []
        scores = []
        
        try:
            # Run the model inference for all boxes at once
            _log.info("Sending request to Replicate API")
            output = replicate.run(
                "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
                input=input_data
            )
            
            # Process the output
            _log.info(f"Received response from Replicate API: {type(output)}")
            
            # The SAM-2 model returns a dict with 'combined_mask' and 'individual_masks'
            individual_masks = []
            combined_mask = None
            
            if isinstance(output, dict):
                if "individual_masks" in output:
                    individual_masks = output["individual_masks"]
                    _log.info(f"Found {len(individual_masks)} individual masks")
                if "combined_mask" in output:
                    combined_mask = output["combined_mask"]
                    _log.info("Found combined mask")
            
            # Use all individual masks when available, let the segmentation function handle the matching
            mask_urls = []
            if individual_masks:
                # Use all individual masks returned by the API
                _log.info(f"Using all {len(individual_masks)} individual masks")
                mask_urls = individual_masks
            elif combined_mask:
                # Only have combined mask, use it as a fallback
                _log.info("Using combined mask as a fallback")
                mask_urls = [combined_mask]
            else:
                # No masks at all
                _log.warning("No masks returned from API")
                mask_urls = [None] * len(boxes)
            
            _log.info(f"Processing {len(mask_urls)} masks (for {len(boxes)} boxes)")
            
            # Process all mask URLs
            for i, mask_obj in enumerate(mask_urls):
                if not mask_obj:
                    _log.warning(f"No mask at index {i}")
                    # Create an empty mask
                    empty_mask = np.zeros((image.height, image.width), dtype=bool)
                    results.append(empty_mask)
                    scores.append(0.0)
                    continue
                
                try:
                    # Get the URL or content from the FileOutput object
                    if hasattr(mask_obj, "url"):
                        # If it has a URL, download it
                        mask_url = mask_obj.url
                        mask_response = requests.get(mask_url)
                        mask_response.raise_for_status()
                        mask_data = mask_response.content
                    else:
                        # Try to get string representation which might be the URL
                        mask_url = str(mask_obj)
                        mask_response = requests.get(mask_url)
                        mask_response.raise_for_status()
                        mask_data = mask_response.content
                    
                    # Load mask image
                    mask_image = Image.open(io.BytesIO(mask_data))
                    mask_array = np.array(mask_image)
                except Exception as e:
                    _log.error(f"Error processing mask for box {i}: {e}")
                    # Create an empty mask on error
                    empty_mask = np.zeros((image.height, image.width), dtype=bool)
                    results.append(empty_mask)
                    scores.append(0.0)
                    continue
                
                # Convert to binary mask (assuming white is the mask)
                if len(mask_array.shape) == 3:  # Color image
                    mask_binary = mask_array.mean(axis=2) > 128
                else:  # Grayscale
                    mask_binary = mask_array > 128
                
                results.append(mask_binary)
                scores.append(1.0)  # Replicate doesn't return confidence scores, assume 1.0
                
        except Exception as e:
            _log.error(f"Error calling Replicate API: {e}")
            # Create empty masks on error for all boxes
            for _ in range(len(boxes)):
                empty_mask = np.zeros((image.height, image.width), dtype=bool)
                results.append(empty_mask)
                scores.append(0.0)
        
        # Format results to match SAM output
        # SAM returns: masks shape (N, 1, H, W), scores shape (N, 1)
        masks = np.array(results)[:, np.newaxis, :, :]
        scores = np.array(scores)[:, np.newaxis]
        
        _log.info(f"Returning {masks.shape[0]} masks from Replicate API")
        
        # If we have no boxes but have masks, ensure we return all masks
        # If we have boxes but more masks, return all masks and let the segmentation function handle matching
        return masks, scores