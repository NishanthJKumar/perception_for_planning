import logging
from functools import cache
from typing import Dict

import cv2
import numpy as np
import open3d as o3d
import trimesh

# Set up logging
_log = logging.getLogger(__name__)


def process_frame(frame: dict, camera_key: str):
    bgra = frame["image"][camera_key]
    bgr = bgra[:, :, :3] / 255.0
    rgb = bgr[:, :, ::-1]
    depth = frame["depth"][camera_key]

    # Convert point cloud to meters
    points = frame["pointcloud"][camera_key][..., :3] / 1000.0
    return rgb, depth, points


def aabb_to_cuboid(aabb: np.ndarray, name: str) -> trimesh.primitives.Box:
    """Convert AABB to trimesh Box.
    
    Args:
        aabb: Axis-aligned bounding box as np.ndarray of shape (2, 3) 
              where aabb[0] is min point and aabb[1] is max point
        name: Name to associate with the box
        
    Returns:
        A trimesh.primitives.Box representing the AABB
    """
    # Calculate box dimensions
    extents = aabb[1] - aabb[0]  # [width, depth, height]
    center = aabb.mean(0)        # Center point
    
    # Create a box centered at origin with the right dimensions
    box = trimesh.primitives.Box(extents=extents)
    
    # Move box to the correct position
    box.apply_translation(center)
    
    # Store name as metadata
    box.metadata = {'name': name}
    
    # Set a default color (light gray)
    box.visual.face_colors = [200, 200, 200, 255]
    
    return box


def segment_table_with_ransac(xyz_world: np.ndarray, rgb: np.ndarray, valid_mask: np.ndarray = None) -> trimesh.primitives.Box:
    """Segment table using RANSAC plane fitting.

    Args:
        xyz_world: Point cloud in world frame. Can be either:
                  - (N, 3) flattened point cloud
                  - (H, W, 3) structured point cloud
        rgb: RGB colors corresponding to the point cloud. Can be either:
                  - (N, 3) flattened colors
                  - (H, W, 3) structured colors
        valid_mask: Optional mask of valid points. If None, all points are considered valid.
                   Should match the shape of xyz_world (excluding the last dimension).

    Returns:
        table_box: trimesh.primitives.Box representing the table
    """
    # Check if pointcloud is already flattened (N, 3) or structured (H, W, 3)
    if len(xyz_world.shape) == 2:
        # Already flattened
        if valid_mask is not None:
            xyz_valid = xyz_world[valid_mask]
            rgb_valid = rgb[valid_mask]
        else:
            # If no mask is provided, use all points but filter NaNs
            valid_mask = ~np.isnan(xyz_world).any(axis=1)
            xyz_valid = xyz_world[valid_mask]
            rgb_valid = rgb[valid_mask]
    else:
        # Structured pointcloud - get valid points
        if valid_mask is None:
            valid_mask = ~np.isnan(xyz_world).any(axis=2)
        xyz_valid = xyz_world[valid_mask]
        rgb_valid = rgb[valid_mask]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_valid)
    pcd.colors = o3d.utility.Vector3dVector(rgb_valid)

    # Filter points above z = -0.1 and downsample
    points = np.asarray(pcd.points)
    mask = points[:, 2] > -0.1
    pcd = pcd.select_by_index(np.where(mask)[0])
    pcd = pcd.voxel_down_sample(voxel_size=0.005)

    # RANSAC plane segmentation
    plane_model, table_idxs = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    table_pcd = pcd.select_by_index(table_idxs)

    # Remove statistical outliers to eliminate distant points that happen to lie on the plane
    table_pcd, inlier_idxs = table_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Get table AABB using percentile-based bounds to handle remaining outliers
    table_pts = np.asarray(table_pcd.points)
    # Use 2nd and 98th percentiles for XY to avoid extreme outliers while keeping most of table
    xy_min = np.percentile(table_pts[:, :2], 2, axis=0)
    xy_max = np.percentile(table_pts[:, :2], 98, axis=0)
    # Use actual min/max for Z since height is well-defined by RANSAC
    z_min = table_pts[:, 2].min()
    z_max = table_pts[:, 2].max()

    table_aabb = np.stack([
        np.append(xy_min, z_min),
        np.append(xy_max, z_max)
    ])
    surface_z = table_pts[:, 2].mean()

    # Create table box
    table_box = aabb_to_cuboid(table_aabb, "table")
    
    # Adjust height position so the top of the box aligns with detected surface
    # We need to adjust the transform directly since trimesh.Box works differently
    extents = table_box.extents
    table_center = table_box.center_mass
    # Offset the box down so its top surface aligns with the detected plane
    height_offset = surface_z - table_center[2] - extents[2]/2 - 0.02  # small offset
    table_box.apply_translation([0, 0, height_offset])
    
    # Set color from point cloud
    table_color = (np.asarray(table_pcd.colors).mean(0) * 255).astype(np.uint8)
    table_color_rgba = np.append(table_color, 255)
    table_box.visual.face_colors = table_color_rgba
    
    _log.info(f"Table surface at z = {surface_z:.3f}, dims = {table_box.extents}")
    return table_box



def project_points_to_table(points: np.ndarray, colors: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Project a set of 3D points down to the table level (minimum z value) and augment the original points with these projections.
    Args:
        points: np.ndarray of shape (N, 3). The original 3D points in world coordinates.
        colors: Optional[np.ndarray] of shape (N, 3) or (N, 4). The RGB(A) colors associated with each point.
    Returns:
        augmented_points: np.ndarray of shape (2N, 3). The original points stacked with their projections onto the table plane (z = min_z).
        augmented_colors: np.ndarray of shape (2N, 3) or (2N, 4), or None. The original colors stacked with themselves, or None if no colors were provided.
    The function is useful for visualizing or processing both the original points and their projections onto the table surface, for example to aid in segmentation or visualization tasks.
    """
    # Find minimum z value (table level)
    min_z = points[:, 2].min()

    projected_points = points.copy()
    projected_points[:, 2] = min_z
    augmented_points = np.vstack([points, projected_points])
    if colors is not None:
        augmented_colors = np.vstack([colors, colors])
    else:
        augmented_colors = None

    return augmented_points, augmented_colors


def segment_pointcloud_by_masks(
    xyz_world: np.ndarray, rgb: np.ndarray, masks: np.ndarray, bboxes: list[dict],
    max_z: float, return_pcd: bool = False, erode_pixels: int = 0
) -> dict[str, trimesh.Trimesh] | tuple[dict[str, trimesh.Trimesh], dict]:
    """Segment pointcloud using object masks.

    Args:
        xyz_world: Point cloud in world frame with shape (H, W, 3) matching the masks
        rgb: RGB colors corresponding to the point cloud with shape (H, W, 3)
        masks: (num_objects, 1, H, W) segmentation masks from SAM
        bboxes: List of bbox dictionaries with 'label' and 'box_2d' keys
        max_z: Maximum z value for filtering points
        return_pcd: Whether to return point clouds in addition to meshes
        erode_pixels: Number of pixels to erode the mask by to handle depth edge noise. Default is 0 (no erosion).

    Returns:
        Dictionary mapping object labels to trimesh.Trimesh objects
    """
    object_meshes = {}
    object_pcds = {}
    masks_2d = masks.squeeze(1).astype(bool)  # (num_objects, H, W)
    
    # Check that we have a structured pointcloud
    if len(xyz_world.shape) != 3 or xyz_world.shape[2] != 3:
        raise ValueError(f"Expected structured pointcloud with shape (H, W, 3), got {xyz_world.shape}. " 
                         f"Flattened pointclouds are not supported as they cannot be directly masked.")
    
    # Handle case where we have more masks than bounding boxes
    if masks.shape[0] > len(bboxes):
        _log.warning(f"More masks ({masks.shape[0]}) than bounding boxes ({len(bboxes)}). "
                    f"Selecting best masks for each bbox based on IoU.")
        
        # Extract 2D bounding boxes from the detection results
        bbox_coords = []
        for bbox in bboxes:
            # Extract coordinates from the box_2d field
            if "box_2d" in bbox and len(bbox["box_2d"]) == 4:
                # Assuming box_2d is [ymin, xmin, ymax, xmax] normalized to 0-1000
                ymin, xmin, ymax, xmax = bbox["box_2d"]
                # Convert to image coordinates based on mask dimensions
                h, w = masks.shape[2:]
                x_min = int((xmin / 1000.0) * w)
                y_min = int((ymin / 1000.0) * h)
                x_max = int((xmax / 1000.0) * w)
                y_max = int((ymax / 1000.0) * h)
                bbox_coords.append([x_min, y_min, x_max, y_max])
            else:
                # If bbox format is invalid, use a dummy bbox
                _log.warning(f"Invalid bbox format for {bbox.get('label', 'unknown')}")
                bbox_coords.append([0, 0, 10, 10])  # Small dummy box to ensure lowest IoU
        
        # Calculate IoU between each mask and each bbox to find best matches
        best_mask_indices = []
        
        for bbox_idx, bbox_coord in enumerate(bbox_coords):
            x_min, y_min, x_max, y_max = bbox_coord
            # Create a binary mask for the bbox
            bbox_mask = np.zeros((masks.shape[2], masks.shape[3]), dtype=bool)
            bbox_mask[y_min:y_max, x_min:x_max] = True
            
            # Calculate IoU for each mask with this bbox
            best_iou = -1
            best_idx = -1
            
            for mask_idx, mask in enumerate(masks_2d):
                # Skip if this mask has already been assigned
                if mask_idx in best_mask_indices:
                    continue
                
                # Calculate intersection and union
                intersection = np.logical_and(mask, bbox_mask).sum()
                union = np.logical_or(mask, bbox_mask).sum()
                iou = intersection / union if union > 0 else 0
                
                # Update if better IoU is found
                if iou > best_iou:
                    best_iou = iou
                    best_idx = mask_idx
            
            if best_idx >= 0:
                best_mask_indices.append(best_idx)
            else:
                # If no good match is found (all masks already assigned),
                # use any unassigned mask
                remaining_indices = [i for i in range(len(masks_2d)) if i not in best_mask_indices]
                if remaining_indices:
                    best_mask_indices.append(remaining_indices[0])
                else:
                    # No masks left - this should not happen with more masks than boxes
                    _log.warning(f"No mask available for bbox {bbox_idx}")
                    best_mask_indices.append(0)  # Use first mask as fallback
        
        _log.info(f"Selected mask indices {best_mask_indices} for {len(bboxes)} bounding boxes")
        
        # Create a new masks array with just the selected masks
        selected_masks = np.zeros((len(best_mask_indices), 1, masks.shape[2], masks.shape[3]), dtype=masks.dtype)
        for i, idx in enumerate(best_mask_indices):
            selected_masks[i, 0] = masks[idx, 0]
        masks = selected_masks
        masks_2d = masks.squeeze(1).astype(bool)  # Update masks_2d with selected masks
    
    # Process each mask and create a mesh for each object
    for mask_2d, bbox in zip(masks_2d, bboxes):
        label = bbox["label"]

        # Erode the mask to handle depth edge noise
        if erode_pixels > 0:
            kernel = np.ones((erode_pixels * 2 + 1, erode_pixels * 2 + 1), np.uint8)
            mask_2d = cv2.erode(mask_2d.astype(np.uint8), kernel, iterations=1).astype(bool)

        # Get points for this object using the mask
        xyz_obj = xyz_world[mask_2d]
        rgb_obj = rgb[mask_2d]

        # Filter out invalid points
        valid = ~np.isnan(xyz_obj).any(axis=1)
        xyz_obj = xyz_obj[valid]
        rgb_obj = rgb_obj[valid]

        if len(xyz_obj) < 10:
            _log.warning(f"Skipping {label}: too few points ({len(xyz_obj)})")
            continue

        z_mask = xyz_obj[..., 2] > max_z
        if z_mask.sum() == 0:
            _log.warning(f"{label} had no points after masking out points with z > {max_z}")
            continue
        xyz_proj, rgb_proj = project_points_to_table(xyz_obj[z_mask], rgb_obj[z_mask])

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_proj)
        pcd.colors = o3d.utility.Vector3dVector(rgb_proj)

        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        object_pcds[label] = pcd

        # Compute convex hull
        try:
            hull, _ = pcd.compute_convex_hull()
            vertices = np.asarray(hull.vertices)
            centroid = vertices.mean(0)
            
            # Get hull faces as triangles
            faces = np.asarray(hull.triangles)
            
            # Convert to trimesh
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=True
            )
            
            # Add metadata
            mesh.metadata = {'name': label, 'centroid': centroid.tolist()}
            
            # Set color from average of point cloud
            mean_color = np.asarray(pcd.colors).mean(0)
            color_rgba = np.append(mean_color * 255, 255).astype(np.uint8)
            mesh.visual.face_colors = color_rgba
            
            # Store the object in the dictionary
            object_meshes[label] = mesh
            _log.info(f"Created mesh for {label}: {len(pcd.points)} pts, centroid={centroid}")

        except Exception as e:
            _log.warning(f"Failed to create mesh for {label}: {e}")

    if return_pcd:
        return object_meshes, object_pcds
    else:
        return object_meshes


@cache
def get_sam_client(
    mode="local", 
    server_url=None, 
    checkpoint=None, 
    model_type="vit_h",
    api_token=None
):
    """Get a SAM client for segmentation.
    
    Args:
        mode: One of "local", "remote", or "replicate"
        server_url: URL of the SAM server if mode is "remote"
        checkpoint: Path to SAM checkpoint if mode is "local"
        model_type: SAM model type if mode is "local"
        api_token: Replicate API token if mode is "replicate"
        
    Returns:
        A configured SAM client
    """
    try:
        # Import the appropriate client based on the mode
        if mode == "replicate":
            from .sam_replicate_client import ReplicateClient
            client = ReplicateClient(api_token=api_token)
        else:
            # For local or remote modes
            from .sam_local_client import SAMClient
            client = SAMClient(
                mode=mode,
                server_url=server_url,
                checkpoint=checkpoint,
                model_type=model_type
            )
        return client
    except ImportError:
        _log.error("SAM client not available. Make sure sam_client.py is in the same directory.")
        raise
    except Exception as e:
        _log.error(f"Failed to initialize SAM client: {e}")
        raise


def save_meshes(meshes: Dict[str, trimesh.Trimesh], output_dir: str):
    """Save meshes to different formats
    
    Args:
        meshes: Dictionary mapping object names to trimesh.Trimesh objects
        output_dir: Directory to save meshes to
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Export scene with all objects
    scene = trimesh.Scene()
    for name, mesh in meshes.items():
        scene.add_geometry(mesh, node_name=name)
    
    # Save as a GLTF/GLB file (good for web visualization)
    scene.export(os.path.join(output_dir, "scene.glb"))
    
    # Save individual meshes as OBJ and PLY
    for name, mesh in meshes.items():
        safe_name = name.replace(" ", "_").replace("/", "_")
        mesh.export(os.path.join(output_dir, f"{safe_name}.obj"))
        mesh.export(os.path.join(output_dir, f"{safe_name}.ply"))
    
    _log.info(f"Saved {len(meshes)} meshes to {output_dir}")


def segment_objects(
    rgb_pil, 
    detection_results: list[dict], 
    sam_mode: str = "local", 
    sam_server_url: str = None,
    sam_checkpoint: str = None,
    replicate_api_token: str = None
) -> np.ndarray:
    """Segment objects using SAM given bounding boxes from Gemini.
    
    Args:
        rgb_pil: PIL Image to segment
        detection_results: List of detection dictionaries from Gemini
        sam_mode: One of "local", "remote", or "replicate"
        sam_server_url: URL of SAM server if mode is "remote"
        sam_checkpoint: Path to SAM checkpoint if mode is "local"
        replicate_api_token: Replicate API token if mode is "replicate"
        
    Returns:
        Array of segmentation masks
    """
    # Get SAM client based on specified mode
    client = get_sam_client(
        mode=sam_mode, 
        server_url=sam_server_url,
        checkpoint=sam_checkpoint,
        api_token=replicate_api_token
    )

    # Convert bounding boxes to SAM format
    # Gemini format: [ymin, xmin, ymax, xmax] normalized to 0-1000
    # SAM format: [x0, y0, x1, y1] in pixel coordinates
    img_height, img_width = rgb_pil.height, rgb_pil.width
    boxes = []
    for detection in detection_results:
        box_2d = detection.get("box_2d", [])
        if len(box_2d) == 4:
            ymin, xmin, ymax, xmax = box_2d
            # Convert from normalized (0-1000) to pixel coordinates
            x0 = (xmin / 1000.0) * img_width
            y0 = (ymin / 1000.0) * img_height
            x1 = (xmax / 1000.0) * img_width
            y1 = (ymax / 1000.0) * img_height
            boxes.append([x0, y0, x1, y1])
    boxes = np.array(boxes)

    # Predict masks for all boxes
    _log.info(f"Segmenting {len(boxes)} objects using SAM in {sam_mode} mode")
    masks, scores = client.segment(rgb_pil, boxes)
    _log.info(f"Generated {len(masks)} segmentation masks. Mask shape: {masks.shape}")  # (num_masks, 1, H, W)
    
    # Note: masks may contain more masks than the number of boxes
    # The segment_pointcloud_by_masks function will handle matching masks to boxes
    return masks