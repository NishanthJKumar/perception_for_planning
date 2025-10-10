#!/usr/bin/env python3
"""
Example script for perception pipeline using Gemini Vision API for object detection and segmentation.

This script demonstrates the following workflow:
1. Load RGB image, depth image, and pointcloud
2. Detect objects using Gemini Vision API
3. Visualize detection results
4. Segment the table using RANSAC
5. Segment objects using masks and create convex hull meshes
6. Generate goal predicates from natural language instructions

Use the --no-cache option to disable caching of Gemini API calls.
"""

import os
import logging
import argparse
import numpy as np
from PIL import Image
import open3d as o3d
from pathlib import Path

from gemini import setup_client, detect_bboxes, translate_task
from visualization import visualize_detections, visualize_masks
from segmentation import segment_table_with_ransac, segment_objects, segment_pointcloud_by_masks, save_meshes

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_log = logging.getLogger(__name__)

# Define input and output paths
DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RGB_PATH = DATA_DIR / "example-rgb-img.png"
DEPTH_PATH = DATA_DIR / "example-depth-img.png"
POINTCLOUD_PATH = DATA_DIR / "example-pointcloud.ply"
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Output paths
BBOX_VIZ_PATH = OUTPUT_DIR / "detection_results.png"
MASK_VIZ_PATH = OUTPUT_DIR / "segmentation_results.png"
MESH_DIR = OUTPUT_DIR / "meshes"

def main(use_cache=True, task_instruction="Put the red cube on the blue cube"):
    """Run the perception pipeline.
    
    Args:
        use_cache: Whether to use caching for Gemini API calls
        task_instruction: Natural language task instruction
    """
    # 1. Load RGB image, depth image, and pointcloud
    _log.info(f"Loading data from {DATA_DIR}")
    rgb_img = Image.open(RGB_PATH)
    depth_img = Image.open(DEPTH_PATH)
    pointcloud = o3d.io.read_point_cloud(str(POINTCLOUD_PATH))

    # Convert to numpy arrays
    rgb_np = np.array(rgb_img)
    depth_np = np.array(depth_img)
    xyz = np.asarray(pointcloud.points)
    valid_mask = ~np.isnan(xyz).any(axis=1) if xyz.size > 0 else np.zeros(0, dtype=bool)
    
    _log.info(f"Loaded RGB image: {rgb_np.shape}, depth image: {depth_np.shape}, pointcloud: {xyz.shape}")

    # 2. Run object detection using Gemini Vision API
    cache_status = "enabled" if use_cache else "disabled"
    _log.info(f"Running object detection with Gemini Vision API (cache {cache_status})")
    client = setup_client()
    detection_results = detect_bboxes(rgb_img, client, use_cache=use_cache)
    _log.info(f"Detected {len(detection_results)} objects")

    # 3. Visualize detection results
    _log.info(f"Visualizing detection results to {BBOX_VIZ_PATH}")
    fig, ax = visualize_detections(rgb_img, detection_results, str(BBOX_VIZ_PATH), show_plot=False)
    
    # 4. Segment the table using RANSAC
    _log.info("Segmenting table using RANSAC")
    
    # Get the flattened pointcloud and colors
    xyz_flat = np.asarray(pointcloud.points)
    rgb_flat = np.asarray(pointcloud.colors)
    
    # Filter out invalid points
    valid_mask = ~np.isnan(xyz_flat).any(axis=1)
    xyz_valid = xyz_flat[valid_mask]
    rgb_valid = rgb_flat[valid_mask]
    
    # Instead of reshaping, we'll pass the flattened pointcloud directly
    # We need to update the segment_table_with_ransac function to work with flattened pointcloud
    table_box = segment_table_with_ransac(xyz_valid, rgb_valid, None)
    _log.info(f"Table segmented: {table_box.extents}")
    
    # 5. Segment objects using masks and create convex hull meshes
    _log.info("Segmenting objects using bounding boxes")
    masks = segment_objects(rgb_img, detection_results)
    
    # Visualize the segmentation masks
    segmentation_viz = visualize_masks(rgb_img, masks, detection_results)
    Image.fromarray(segmentation_viz).save(MASK_VIZ_PATH)
    _log.info(f"Saved segmentation visualization to {MASK_VIZ_PATH}")
    
    # Create convex hull meshes for each segmented object using flat pointcloud
    # We'll need to create an approximate mapping between image pixels and pointcloud points
    # We will project the pointcloud points to the image plane to find correspondences
    
    # Get RGB image dimensions and create an empty object_meshes dictionary
    H, W = rgb_np.shape[:2]
    object_meshes = {}
    
    # Create convex hull meshes for each segmented object
    # Pass the flattened point cloud and set project_points to True
    object_meshes = segment_pointcloud_by_masks(
        xyz_valid, rgb_valid, masks, detection_results, project_points=True)
    _log.info(f"Created {len(object_meshes)} object meshes")
    
    # Save all meshes to files
    # Create the directory if it doesn't exist
    os.makedirs(MESH_DIR, exist_ok=True)
    
    # Save individual meshes
    for name, mesh in object_meshes.items():
        safe_name = name.replace(" ", "_").replace("/", "_")
        mesh_path = MESH_DIR / f"{safe_name}.obj"
        mesh.export(str(mesh_path))
    
    # Save the table mesh
    table_mesh_path = MESH_DIR / "table.obj"
    table_box.export(str(table_mesh_path))
    
    _log.info(f"Saved all meshes to {MESH_DIR}")
    
    # 6. Generate goal predicates from a natural language instruction
    cache_status = "enabled" if use_cache else "disabled"
    _log.info(f"Generating goal predicates for task: '{task_instruction}' (cache {cache_status})")
    predicates = translate_task(task_instruction, detection_results, str(BBOX_VIZ_PATH), use_cache=use_cache)
    _log.info(f"Generated {len(predicates)} goal predicates:")
    for p in predicates:
        _log.info(f"  - {p['predicate']}({', '.join(p['args'])})")
    
    # Done!
    _log.info("Perception pipeline completed successfully!")
    return detection_results, masks, table_box, object_meshes, predicates

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Perception pipeline for object detection and segmentation")
    
    # Task instruction
    parser.add_argument("--task", type=str, default="Put the red cube on the blue cube",
                       help="Natural language task instruction")
    
    # Caching options
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching of Gemini API calls")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(use_cache=not args.no_cache, task_instruction=args.task)
