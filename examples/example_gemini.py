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

You can choose from three modes to run SAM segmentation:
    - Local mode: python example_gemini.py --sam-mode local --sam-checkpoint /path/to/sam_checkpoint.pth
    - Remote mode: python example_gemini.py --sam-mode remote --sam-server http://localhost:8000
    - Replicate API: python example_gemini.py --sam-mode replicate --replicate-token your_api_token

Use the --no-cache option to disable caching of Gemini API calls.
"""

import os
import logging
import argparse
import numpy as np
from PIL import Image
import open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree  # For point cloud projection

from perception_for_planning.gemini import setup_client, detect_bboxes, translate_task
from perception_for_planning.visualization import visualize_detections, visualize_masks
from perception_for_planning.segmentation import segment_table_with_ransac, segment_objects, segment_pointcloud_by_masks, save_meshes

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

def main(
    use_cache=True, 
    task_instruction="Put the red cube on the blue cube",
    sam_mode="local",
    sam_server_url=None,
    sam_checkpoint=None,
    replicate_api_token=None
):
    """Run the perception pipeline.
    
    Args:
        use_cache: Whether to use caching for Gemini API calls
        task_instruction: Natural language task instruction
        sam_mode: Mode for SAM segmentation: "local", "remote", or "replicate"
        sam_server_url: URL of SAM server (for remote mode)
        sam_checkpoint: Path to SAM checkpoint (for local mode)
        replicate_api_token: Replicate API token (for replicate mode)
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
    _log.info(f"Segmenting objects using {sam_mode} SAM mode")
    masks = segment_objects(
        rgb_img, 
        detection_results, 
        sam_mode=sam_mode,
        sam_server_url=sam_server_url,
        sam_checkpoint=sam_checkpoint,
        replicate_api_token=replicate_api_token
    )
    
    _log.info(f"Got {masks.shape[0]} masks for {len(detection_results)} detection results")
    
    # Visualize the segmentation masks (the visualize_masks function now handles mask-bbox matching)
    segmentation_viz = visualize_masks(rgb_img, masks, detection_results)
    Image.fromarray(segmentation_viz).save(MASK_VIZ_PATH)
    _log.info(f"Saved segmentation visualization to {MASK_VIZ_PATH}")
    
    # Create convex hull meshes for each segmented object using flat pointcloud
    # We'll need to create an approximate mapping between image pixels and pointcloud points
    # We will project the pointcloud points to the image plane to find correspondences
    
    # Get RGB image dimensions and create an empty object_meshes dictionary
    H, W = rgb_np.shape[:2]
    object_meshes = {}
    
    # Create a structured point cloud from the flattened one
    # We need to project the flattened point cloud back to the image space
    _log.info("Projecting flattened point cloud to image space")
    
    # Get image dimensions
    H, W = rgb_np.shape[:2]
    
    # Create an empty structured point cloud and RGB data
    structured_xyz = np.full((H, W, 3), np.nan)
    structured_rgb = np.full((H, W, 3), np.nan)
    
    # To project the points, we need the camera intrinsics
    # Since we don't have the exact intrinsics, we'll create a simple projection
    # based on the depth image and pointcloud
    
    # First, normalize the depth image to approximate Z values
    if depth_np.size > 0:
        depth_norm = depth_np.astype(float)
        # Normalize to approximate depth range (assuming values between 0-255)
        depth_norm = depth_norm / 255.0 * 2.0  # Scale to approximately 0-2m range
        
        # For each point in the pointcloud, find the closest pixel in the image
        # This is a simplified approach - a real implementation would use proper camera projection
        
        # Create 2D grid of pixel coordinates
        y_grid, x_grid = np.mgrid[0:H, 0:W]
        pixel_coords = np.column_stack([y_grid.flatten(), x_grid.flatten()])
        
        # Create a KD-tree for efficient nearest neighbor search
        pixel_tree = cKDTree(pixel_coords)
        
        # Project points to image space (simplified approach)
        # Assuming points are roughly in the camera frame and normalized
        points_z = xyz_valid[:, 2]  # Z-coordinates
        
        # Filter points with reasonable Z values (in front of camera)
        valid_z_mask = (points_z < 2.0) & (points_z > -2.0)
        xyz_filtered = xyz_valid[valid_z_mask]
        rgb_filtered = rgb_valid[valid_z_mask]
        
        # Normalize XY by Z to get approximate image coordinates (simplified projection)
        points_x = xyz_filtered[:, 0] / (xyz_filtered[:, 2] + 1e-6)  # Add epsilon to avoid division by zero
        points_y = xyz_filtered[:, 1] / (xyz_filtered[:, 2] + 1e-6)
        
        # Scale and shift to image coordinates
        # These scaling factors would ideally come from camera intrinsics
        scale_x = W / 2.0
        scale_y = H / 2.0
        
        img_x = np.clip((points_x * scale_x) + W/2, 0, W-1).astype(int)
        img_y = np.clip((points_y * scale_y) + H/2, 0, H-1).astype(int)
        
        # Assign points to the structured arrays
        for i in range(len(img_y)):
            y, x = img_y[i], img_x[i]
            if 0 <= y < H and 0 <= x < W:
                # Only overwrite if the new point is closer (smaller Z)
                if np.isnan(structured_xyz[y, x, 2]) or xyz_filtered[i, 2] < structured_xyz[y, x, 2]:
                    structured_xyz[y, x] = xyz_filtered[i]
                    structured_rgb[y, x] = rgb_filtered[i]
    
    # Check if we have enough valid points in our structured point cloud
    valid_points = np.sum(~np.isnan(structured_xyz).any(axis=2))
    _log.info(f"Structured point cloud has {valid_points} valid points out of {H*W} pixels")
    
    if valid_points < 100:
        _log.warning("Not enough valid points in structured point cloud, segmentation might fail")
        
    # Create convex hull meshes for each segmented object using the structured point cloud
    object_meshes = segment_pointcloud_by_masks(
        structured_xyz, structured_rgb, masks, detection_results)
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
    
    # SAM configuration
    parser.add_argument("--sam-mode", type=str, choices=["local", "remote", "replicate"], default="local",
                       help="Mode for SAM segmentation: local, remote, or replicate")
    parser.add_argument("--sam-server", type=str, 
                       help="URL of SAM server (required for remote mode)")
    parser.add_argument("--sam-checkpoint", type=str,
                       help="Path to SAM checkpoint (required for local mode)")
    parser.add_argument("--replicate-token", type=str,
                       help="Replicate API token (required for replicate mode, or use REPLICATE_API_TOKEN env var)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        use_cache=not args.no_cache, 
        task_instruction=args.task,
        sam_mode=args.sam_mode,
        sam_server_url=args.sam_server,
        sam_checkpoint=args.sam_checkpoint,
        replicate_api_token=args.replicate_token
    )
