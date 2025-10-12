"""
Perception For Planning - Computer vision tools for robotic planning tasks
==========================================================================

This package provides tools for object detection, segmentation, and 3D reconstruction
to support robotic planning tasks.

Main modules:
- gemini: Interface with Gemini Vision API for object detection
- segmentation: Image and pointcloud segmentation utilities
- visualization: Visualization tools for detections and segmentations
- sam_replicate_client: SAM-2 segmentation model client via Replicate API
- sam_local_client: Local SAM model client

Copyright (c) 2023-2025 NishanthJKumar
"""

__version__ = '0.1.0'

# Core modules that should be importable directly
from . import gemini
from . import segmentation
from . import visualization

# Optional/specialized modules
from . import gemini_cache
from . import sam_replicate_client
from . import sam_local_client