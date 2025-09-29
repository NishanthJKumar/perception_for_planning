"""Utility functions for perception and planning."""
from typing import List, Dict
from structs import Object, ObjectType
from query_gemini import extract_object_types_with_vlm


def create_objects_from_detections(
    detection_results: List[dict], 
    object_types: Dict[str, ObjectType],
    client
) -> List[Object]:
    """
    Create Object instances from Gemini detection results.
    
    Args:
        detection_results: List of detection dictionaries from parse_and_display_results.
                          Expected format: [{"box_2d": [...], "label": "object_name"}, ...]
                          or [{"point": [...], "label": "object_name"}, ...]
        object_types: Dictionary mapping object type names to ObjectType instances.
                     e.g., {"cup": ObjectType("cup"), "table": ObjectType("table")}
        client: Optional GenAI client for VLM-based type extraction. If None, uses labels directly.
    
    Returns:
        List[Object]: List of Object instances created from the detection results.
    """
    objects = []
    
    # Extract all labels for batch processing with VLM if client provided
    labels = [detection.get('label', f'unknown_object_{i}') for i, detection in enumerate(detection_results)]
    # Extract allowed type names from the object_types dictionary
    allowed_types = list(object_types.keys())
    label_to_base_type = extract_object_types_with_vlm(client, labels, allowed_types)
    # Output a set of objects    
    for i, detection in enumerate(detection_results):
        label = detection.get('label', f'unknown_object_{i}')
        
        # Use the original VLM label as the name (no modification)
        object_name = label
        
        # Create a unique ID for this object instance
        unique_id = f"{object_name.replace(' ', '_')}_{i:03d}"  # e.g., obj_001, obj_002, etc.
        
        # Get base type from VLM extraction or fallback
        base_type_name = label_to_base_type.get(label, label.lower().strip())        
        
        # Get ObjectType from provided dictionary, or create a default one
        if base_type_name in object_types:
            object_type = object_types[base_type_name]
        else:
            # Create a default ObjectType if not found in provided types
            object_type = ObjectType(base_type_name)        
        
        # Create the Object instance with unique_id, name, and type
        obj = Object(unique_id=unique_id, name=object_name, type=object_type)
        objects.append(obj)    
    return objects