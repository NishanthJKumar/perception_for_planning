import logging
import numpy as np
import open3d as o3d
import trimesh
from functools import cache
from typing import Dict

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

    # Get table AABB and surface height
    table_aabb_o3d = table_pcd.get_axis_aligned_bounding_box()
    table_aabb = np.stack([table_aabb_o3d.min_bound, table_aabb_o3d.max_bound])
    surface_z = np.asarray(table_pcd.points)[:, 2].mean()

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


def segment_pointcloud_by_masks(
    xyz_world: np.ndarray, rgb: np.ndarray, masks: np.ndarray, bboxes: list[dict],
    project_points: bool = False
) -> dict[str, trimesh.Trimesh]:
    """Segment pointcloud using object masks.
    
    Args:
        xyz_world: Point cloud in world frame. Can be either:
                  - (N, 3) flattened point cloud
                  - (H, W, 3) structured point cloud matching the masks shape
        rgb: RGB colors corresponding to the point cloud
        masks: (num_objects, 1, H, W) segmentation masks from SAM
        bboxes: List of bbox dictionaries with 'label' and 'box_2d' keys
        project_points: If True and xyz_world is flattened, will attempt to project
                      points onto the image plane for segmentation
    
    Returns:
        Dictionary mapping object labels to trimesh.Trimesh objects
    """
    object_meshes = {}
    masks_2d = masks.squeeze(1).astype(bool)  # (num_objects, H, W)
    
    # Check if we have a flattened pointcloud or one that matches the image structure
    if len(xyz_world.shape) == 2:  # Flattened (N, 3)
        _log.info(f"Working with flattened pointcloud of shape {xyz_world.shape}")
        
        # Create a simple box for each object as placeholder
        # In a real implementation, you'd project the points onto the image to find correspondences
        for bbox, mask_2d in zip(bboxes, masks_2d):
            label = bbox["label"]
            
            # Create a placeholder box based on the bounding box dimensions
            box_2d = bbox.get("box_2d", [])  # [ymin, xmin, ymax, xmax] normalized to 0-1000
            if len(box_2d) == 4:
                # Create a simple box mesh
                mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
                mesh.metadata = {'name': label}
                mesh.visual.face_colors = np.random.randint(0, 255, size=4).astype(np.uint8)
                object_meshes[label] = mesh
                _log.info(f"Created placeholder mesh for {label}")
            
        return object_meshes
    
    # If we have a structured pointcloud that matches the masks
    for mask_2d, bbox in zip(masks_2d, bboxes):
        label = bbox["label"]

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

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_obj)
        pcd.colors = o3d.utility.Vector3dVector(rgb_obj)

        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)

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

    return object_meshes


@cache
def get_sam_client(
    mode="local", 
    server_url=None, 
    checkpoint=None, 
    model_type="vit_h"
):
    """Get a SAM client for segmentation.
    
    Args:
        mode: Either "local" or "remote"
        server_url: URL of the SAM server if mode is "remote"
        checkpoint: Path to SAM checkpoint if mode is "local"
        model_type: SAM model type if mode is "local"
        
    Returns:
        A configured SAM client
    """
    try:
        # Import here to allow segmentation.py to work without sam_client dependency
        from sam_client import SAMClient
        
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
    sam_checkpoint: str = None
) -> np.ndarray:
    """Segment objects using SAM given bounding boxes from Gemini.
    
    Args:
        rgb_pil: PIL Image to segment
        detection_results: List of detection dictionaries from Gemini
        sam_mode: Either "local" or "remote"
        sam_server_url: URL of SAM server if mode is "remote"
        sam_checkpoint: Path to SAM checkpoint if mode is "local"
        
    Returns:
        Array of segmentation masks
    """
    # Get SAM client based on specified mode
    client = get_sam_client(
        mode=sam_mode, 
        server_url=sam_server_url,
        checkpoint=sam_checkpoint
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
    _log.info(f"Generated {len(masks)} segmentation masks. Mask shape: {masks.shape}")  # (num_objects, 1, H, W)
    return masks