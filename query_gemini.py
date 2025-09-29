#!/usr/bin/env python3
"""
Gemini Robotics-ER 1.5 Bounding Box Detection Example

This script demonstrates how to use the Gemini Robotics-ER 1.5 model for object detection
and bounding box generation. It loads an image and uses the Gemini model to detect objects
and return their bounding boxes with labels.

Usage:
    python query_gemini.py

Requirements:
    - A valid Google API key set as the GOOGLE_API_KEY environment variable
    - An image file (gemini-test-img.png) in the same directory
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from google import genai
from google.genai import types
from PIL import Image


def setup_client():
    """Initialize the GenAI client with API key."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is required. "
            "Get your API key from https://aistudio.google.com/apikey"
        )
    
    return genai.Client(api_key=api_key)


def load_and_prepare_image(image_path):
    """Load and prepare image for processing."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        img = Image.open(image_path)
        # Resize image for faster processing while maintaining aspect ratio
        img = img.resize((800, int(800 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS)
        print(f"Loaded and resized image: {img.size}")
        return img
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")


def detect_objects_with_bounding_boxes(client, image, task_instruction=None, model_id="gemini-robotics-er-1.5-preview"):
    """
    Detect objects in the image and return bounding boxes.
    
    Args:
        client: GenAI client instance
        image: PIL Image object
        task_instruction: a string corresponding to an instruction
        model_id: Model ID to use for detection
        
    Returns:
        str: JSON string containing bounding box results
    """
    prompt = """
    Return bounding boxes as a JSON array with labels. Never return masks
    or code fencing. Limit to 25 objects. Include as many objects as you
    can identify in the image.
    
    If an object is present multiple times, name them according to their
    unique characteristic (colors, size, position, unique characteristics, etc.).
    
    The format should be as follows: [{"box_2d": [ymin, xmin, ymax, xmax],
    "label": <label for the object>}] normalized to 0-1000. The values in
    box_2d must only be integers.
    """
    if task_instruction is not None:
        prompt = f"""
        You are an expert robotics system that is attempting to perform the task:
        {task_instruction}.""" + \
        """
        Return bounding boxes as a JSON array with labels. Never return masks
        or code fencing. Limit to 25 objects. Include as many objects as you
        can identify in the image. Make sure that the objects necessary for the task
        instruction above are included.
        
        If an object is present multiple times, name them according to their
        unique characteristic (colors, size, position, unique characteristics, etc.).
        
        The format should be as follows: [{"box_2d": [ymin, xmin, ymax, xmax],
        "label": <label for the object>}] normalized to 0-1000. The values in
        box_2d must only be integers.
        """


    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                temperature=0.5,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        return response.text
    except Exception as e:
        raise RuntimeError(f"Error generating content: {e}")


def detect_specific_objects(client, image, objects_list, model_id="gemini-robotics-er-1.5-preview"):
    """
    Detect specific objects in the image and return their coordinates.
    
    Args:
        client: GenAI client instance
        image: PIL Image object
        objects_list: List of object names to detect
        model_id: Model ID to use for detection
        
    Returns:
        str: JSON string containing point results
    """
    prompt = f"""
    Get all points matching the following objects: {', '.join(objects_list)}.
    The label returned should be an identifying name for the object detected.
    
    The answer should follow the json format:
    [{{"point": <point>, "label": <label1>}}, ...]. The points are in
    [y, x] format normalized to 0-1000.
    """
    
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                temperature=0.5,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        return response.text
    except Exception as e:
        raise RuntimeError(f"Error generating content: {e}")


def extract_object_types_with_vlm(client, labels, allowed_types, model_id="gemini-robotics-er-1.5-preview"):
    """
    Use the VLM to extract base object types from detection labels.
    
    Args:
        client: GenAI client instance
        labels: List of label strings from detection results
        allowed_types: List of allowed object type names to constrain the output
        model_id: Model ID to use for extraction
        
    Returns:
        dict: Mapping from original labels to extracted base types
    """
    if not labels:
        return {}
    
    labels_str = ', '.join(labels)
    allowed_types_str = ', '.join(allowed_types)
    prompt = f"""
    Given these object labels from a detection system: {labels_str}
    
    Extract the base object type for each label, removing descriptive modifiers like colors, sizes, positions, etc.
    You must choose from ONLY these allowed object types: {allowed_types_str}
    
    For each label, find the most appropriate type from the allowed list. If no good match exists, use "unknown".
    
    Examples:
    - "red cup" -> "cup" (if "cup" is in allowed types)
    - "small wooden table" -> "table" (if "table" is in allowed types)  
    - "person in blue shirt" -> "person" (if "person" is in allowed types)
    - "green apple" -> "apple" (if "apple" is in allowed types)
    - "something not in list" -> "unknown"
    
    Return the results as a JSON object mapping each original label to its base type:
    {{"original_label1": "base_type1", "original_label2": "base_type2", ...}}
    
    Remember: Only use types from this list: {allowed_types_str}, or "unknown" if no match.
    """
    
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for consistency
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        # Clean and parse the response
        cleaned_text = response.text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text.replace('```', '')
        
        type_mapping = json.loads(cleaned_text)
        return type_mapping
        
    except Exception as e:
        print(f"Error extracting object types with VLM: {e}")
        # Fallback: return original labels as types
        return {label: label.lower().strip() for label in labels}


def parse_and_display_results(response_text, detection_type="bounding_boxes"):
    """Parse and display the detection results."""
    try:
        # Clean the response text to extract JSON
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text.replace('```', '')
        
        # Parse JSON
        results = json.loads(cleaned_text)
        
        print(f"\n{detection_type.replace('_', ' ').title()} Results:")
        print("=" * 50)
        
        for i, item in enumerate(results, 1):
            if detection_type == "bounding_boxes":
                box = item.get('box_2d', [])
                label = item.get('label', 'Unknown')
                print(f"{i:2d}. {label}")
                print(f"    Bounding Box: [ymin={box[0]}, xmin={box[1]}, ymax={box[2]}, xmax={box[3]}]")
            else:  # points
                point = item.get('point', [])
                label = item.get('label', 'Unknown')
                print(f"{i:2d}. {label}")
                print(f"    Point: [y={point[0]}, x={point[1]}]")
        
        print(f"\nTotal objects detected: {len(results)}")
        return results
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response_text}")
        return None
    except Exception as e:
        print(f"Error processing results: {e}")
        return None


def translate_task_to_predicates(client, labeled_image_path, task_description, available_predicates, objects, model_id="gemini-robotics-er-1.5-preview"):
    """
    Use VLM to translate a natural language task description into grounded atoms.
    
    Args:
        client: GenAI client instance
        labeled_image_path: Path to the image with object ID labels
        task_description: Natural language description of the task
        available_predicates: List of Predicate instances available for use
        objects: List of Object instances detected in the scene
        model_id: Model ID to use for translation
        
    Returns:
        List[Atom]: List of grounded atom instances representing the task goal
    """
    from structs import Atom
    
    # Create object reference string for the VLM
    object_references = []
    for obj in objects:
        object_references.append(f"{obj.unique_id}: {obj.name} (type: {obj.type.name})")
    object_ref_str = "\n".join(object_references)
    
    # Create predicate reference string
    predicate_references = []
    for pred in available_predicates:
        arg_str = ", ".join([obj_type.name for obj_type in pred.arg_types])
        predicate_references.append(f"{pred.name}({arg_str}): {pred.description}")
    predicate_ref_str = "\n".join(predicate_references)
    
    prompt = f"""
    You are viewing an image with labeled objects and need to translate a natural language task into formal predicates.
    
    TASK: {task_description}
    
    AVAILABLE OBJECTS (with their unique IDs visible in the image):
    {object_ref_str}
    
    AVAILABLE PREDICATES:
    {predicate_ref_str}
    
    Please analyze the task and return a JSON list of predicates that represent the goal state.
    Each predicate should specify:
    - "name": the predicate name
    - "args": list of object unique_ids that should be arguments
    """ + \
    """    
    For example, if the task is "put the red cup on the table" and you see objects "red_cup_000" and "wooden_table_001", you might return:
    [{{"name": "on", "args": ["red_cup_000", "wooden_table_001"]}}]
    
    Look carefully at the labeled image to identify the specific objects mentioned in the task.
    Only use object IDs that are visible in the image and predicate names from the available list.
    
    Return your response as a JSON array:
    """
    
    try:
        # Load the labeled image
        if os.path.exists(labeled_image_path):
            labeled_image = Image.open(labeled_image_path)
        else:
            raise FileNotFoundError(f"Labeled image not found: {labeled_image_path}")
        
        response = client.models.generate_content(
            model=model_id,
            contents=[labeled_image, prompt],
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for consistency
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        # Clean and parse the response
        cleaned_text = response.text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text.replace('```', '')
        
        predicate_specs = json.loads(cleaned_text)
        
        # Convert to grounded Atom objects
        grounded_atoms = []
        object_lookup = {obj.unique_id: obj for obj in objects}
        
        for spec in predicate_specs:
            pred_name = spec.get('name', '')
            arg_ids = spec.get('args', [])
            
            # Look up object instances
            pred_objects = []
            for arg_id in arg_ids:
                if arg_id in object_lookup:
                    pred_objects.append(object_lookup[arg_id])
                else:
                    print(f"Warning: Object ID '{arg_id}' not found in detected objects")
            
            if pred_objects:  # Only create atom if we found valid objects
                grounded_atom = Atom(name=pred_name, object_args=pred_objects)
                grounded_atoms.append(grounded_atom)
        
        return grounded_atoms
        
    except Exception as e:
        print(f"Error translating task to predicates: {e}")
        return []


def visualize_detections(image, results, detection_type="bounding_boxes", output_path=None, show_plot=True):
    """
    Visualize detection results by overlaying bounding boxes or points on the image.
    
    Args:
        image: PIL Image object
        results: List of detection results from parse_and_display_results
        detection_type: "bounding_boxes" or "points"
        output_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot interactively
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    if not results:
        print("No results to visualize")
        return None, None
    
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(image)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    
    # Get image dimensions for coordinate conversion
    img_height, img_width = img_array.shape[:2]
    
    # Generate colors for different objects (using a more vibrant colormap)
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    
    if detection_type == "bounding_boxes":
        for i, item in enumerate(results):
            box = item.get('box_2d', [])
            label = item.get('label', 'Unknown')
            
            if len(box) == 4:
                # Convert normalized coordinates (0-1000) to pixel coordinates
                ymin, xmin, ymax, xmax = box
                ymin = (ymin / 1000.0) * img_height
                xmin = (xmin / 1000.0) * img_width
                ymax = (ymax / 1000.0) * img_height
                xmax = (xmax / 1000.0) * img_width
                
                # Create rectangle patch
                width = xmax - xmin
                height = ymax - ymin
                
                rect = patches.Rectangle(
                    (xmin, ymin), width, height,
                    linewidth=2, edgecolor=colors[i], facecolor='none',
                    alpha=0.8
                )
                ax.add_patch(rect)
                
                # Add label with background
                # Add label with background
                ax.text(
                    xmin, ymin - 5, f"{i+1}. {label}",
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.8)
                )
    
    else:  # points
        for i, item in enumerate(results):
            point = item.get('point', [])
            label = item.get('label', 'Unknown')
            
            if len(point) == 2:
                # Convert normalized coordinates (0-1000) to pixel coordinates
                y, x = point
                y = (y / 1000.0) * img_height
                x = (x / 1000.0) * img_width
                
                # Draw point as a circle
                circle = patches.Circle(
                    (x, y), radius=8,
                    linewidth=2, edgecolor='white', facecolor=colors[i],
                    alpha=0.8
                )
                ax.add_patch(circle)
                
                # Add label with background
                # Add label with background
                ax.text(
                    x + 12, y, f"{i+1}. {label}",
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.8)
                )
    
    ax.set_title(f"{detection_type.replace('_', ' ').title()} Detection Results", fontsize=14, weight='bold')
    ax.axis('off')  # Hide axes
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    # Show the plot
    if show_plot:
        plt.show()
    
    return fig, ax


def main():
    """Main function to run the object detection example."""
    image_path = "gemini-test-img.png"
    
    try:
        # Setup
        print("Initializing Gemini Robotics-ER 1.5...")
        client = setup_client()
        
        # Load image
        print(f"Loading image: {image_path}")
        image = load_and_prepare_image(image_path)
        
        # Example 1: Detect all objects with bounding boxes
        print("\n" + "="*60)
        print("EXAMPLE 1: Detecting all objects with bounding boxes")
        print("="*60)
        
        bbox_response = detect_objects_with_bounding_boxes(client, image)
        bbox_results = parse_and_display_results(bbox_response, "bounding_boxes")
        
        # Visualize bounding boxes
        if bbox_results:
            print("\nGenerating bounding box visualization...")
            visualize_detections(
                image, bbox_results, "bounding_boxes", 
                output_path=f"{image_path.split('.')[0]}_bounding_boxes.png"
            )
        
        # Example 2: Detect specific objects (points)
        print("\n" + "="*60)
        print("EXAMPLE 2: Detecting specific objects (points)")
        print("="*60)
        
        # You can customize this list based on what might be in your image
        target_objects = ["person", "chair", "table", "bottle", "cup", "book", "laptop", "phone"]
        
        points_response = detect_specific_objects(client, image, target_objects)
        points_results = parse_and_display_results(points_response, "points")
        
        # Visualize points
        if points_results:
            print("\nGenerating point detection visualization...")
            visualize_detections(
                image, points_results, "points", 
                output_path=f"{image_path.split('.')[0]}_points.png"
            )
        
        # Example 3: Detect fruit (category-based detection)
        print("\n" + "="*60)
        print("EXAMPLE 3: Detecting fruit (category-based)")
        print("="*60)
        
        fruit_prompt = """
        Get all points for fruit. The label returned should be an identifying
        name for the object detected.
        The answer should follow the json format:
        [{"point": <point>, "label": <label1>}, ...]. The points are in
        [y, x] format normalized to 0-1000.
        """
        
        try:
            fruit_response = client.models.generate_content(
                model="gemini-robotics-er-1.5-preview",
                contents=[image, fruit_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            
            fruit_results = parse_and_display_results(fruit_response.text, "points")
            
            # Visualize fruit detection
            if fruit_results:
                print("\nGenerating fruit detection visualization...")
                visualize_detections(
                    image, fruit_results, "points", 
                    output_path=f"{image_path.split('.')[0]}_fruit.png"
                )
            
        except Exception as e:
            print(f"Error detecting fruit: {e}")
        
        print("\n" + "="*60)
        print("Detection complete!")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
