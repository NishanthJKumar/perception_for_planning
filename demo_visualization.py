#!/usr/bin/env python3
"""
Object Detection and Classification Demo

This script demonstrates object detection and classification into movable/immovable types.
"""

import os
from query_gemini import (
    setup_client, 
    load_and_prepare_image, 
    detect_objects_with_bounding_boxes,
    parse_and_display_results,
    visualize_detections
)
from utils import create_objects_from_detections
from structs import ObjectType


def object_detection_demo():
    """Run object detection and classify objects as movable or immovable."""
    image_path = "gemini-test-img.png"
    
    if not os.path.exists(image_path):
        print(f"Please add an image file named '{image_path}' to run this demo")
        return
    
    try:
        # 1. Load the image from image_path
        print("Setting up Gemini client...")
        client = setup_client()
        
        print(f"Loading image: {image_path}")
        image = load_and_prepare_image(image_path)
        
        # 2. Run object detection and parse results
        print("Detecting objects...")
        response = detect_objects_with_bounding_boxes(client, image)
        detection_results = parse_and_display_results(response, "bounding_boxes")
        
        if not detection_results:
            print("No objects detected")
            return
        
        # 3. Convert detections into a set of objects with types that are either "movable" or "immovable"
        print("\nClassifying objects as movable or immovable...")
        
        # Define object types - only "movable" and "immovable"
        object_types = {
            "movable": ObjectType("movable"),
            "immovable": ObjectType("immovable")
        }
        
        # Create objects from detections using VLM-based type extraction
        objects = create_objects_from_detections(detection_results, object_types, client)
        
        # 4. Print out the final list of objects
        print("\n" + "="*60)
        print("FINAL LIST OF OBJECTS")
        print("="*60)
        
        movable_objects = []
        immovable_objects = []
        
        for obj in objects:
            if obj.type.name == "movable":
                movable_objects.append(obj)
            elif obj.type.name == "immovable":
                immovable_objects.append(obj)
        
        print(f"\nMOVABLE OBJECTS ({len(movable_objects)}):")
        print("-" * 30)
        for i, obj in enumerate(movable_objects, 1):
            print(f"{i:2d}. {obj.name} (ID: {obj.unique_id}, type: {obj.type.name})")
        
        print(f"\nIMMOVABLE OBJECTS ({len(immovable_objects)}):")
        print("-" * 30)
        for i, obj in enumerate(immovable_objects, 1):
            print(f"{i:2d}. {obj.name} (ID: {obj.unique_id}, type: {obj.type.name})")
        
        print(f"\nTOTAL OBJECTS: {len(objects)}")
        print(f"  - Movable: {len(movable_objects)}")
        print(f"  - Immovable: {len(immovable_objects)}")
        
        # Create visualization with bounding boxes labeled with unique object IDs
        print("\n" + "="*60)
        print("CREATING VISUALIZATION")
        print("="*60)
        
        # Create modified detection results with unique IDs as labels
        modified_detection_results = []
        for i, (detection, obj) in enumerate(zip(detection_results, objects)):
            modified_detection = detection.copy()  # Copy original detection data
            modified_detection['label'] = obj.unique_id  # Replace label with unique ID
            modified_detection_results.append(modified_detection)
        
        # Import visualization function        
        print("Generating bounding box visualization with unique object IDs...")
        fig, ax = visualize_detections(
            image, modified_detection_results, "bounding_boxes",
            output_path="object_detection_with_unique_ids.png",
            show_plot=True
        )
        
    except Exception as e:
        print(f"Error in demo: {e}")


if __name__ == "__main__":
    object_detection_demo()