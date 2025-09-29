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


def setup_client() -> genai.Client:
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is required. "
            "Get your API key from https://aistudio.google.com/apikey"
        )
    
    return genai.Client(api_key=api_key)


def load_and_prepare_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = Image.open(image_path)
    img = img.resize((800, int(800 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS)
    print(f"Loaded and resized image: {img.size}")
    return img


def detect_objects_with_bounding_boxes(client: genai.Client, image: Image.Image, task_instruction: str | None = None, model_id: str = "gemini-robotics-er-1.5-preview") -> str:
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


    response = client.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    
    return response.text





def extract_object_types_with_vlm(client: genai.Client, labels: list[str], allowed_types: list[str], model_id: str = "gemini-robotics-er-1.5-preview") -> dict[str, str]:
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
    
    response = client.models.generate_content(
        model=model_id,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    
    cleaned_text = response.text.strip()
    if cleaned_text.startswith('```json'):
        cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
    elif cleaned_text.startswith('```'):
        cleaned_text = cleaned_text.replace('```', '')
    
    return json.loads(cleaned_text)


def parse_and_display_results(response_text: str) -> list[dict] | None:
    cleaned_text = response_text.strip()
    if cleaned_text.startswith('```json'):
        cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
    elif cleaned_text.startswith('```'):
        cleaned_text = cleaned_text.replace('```', '')
    
    results = json.loads(cleaned_text)
    
    print(f"\nBounding Boxes Results:")
    print("=" * 50)
    
    for i, item in enumerate(results, 1):
        box = item.get('box_2d', [])
        label = item.get('label', 'Unknown')
        print(f"{i:2d}. {label}")
        print(f"    Bounding Box: [ymin={box[0]}, xmin={box[1]}, ymax={box[2]}, xmax={box[3]}]")
    
    print(f"\nTotal objects detected: {len(results)}")
    return results


def translate_task_to_predicates(client: genai.Client, labeled_image_path: str, task_description: str, available_predicates: list, objects: list, model_id: str = "gemini-robotics-er-1.5-preview") -> list:
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
    
    if os.path.exists(labeled_image_path):
        labeled_image = Image.open(labeled_image_path)
    else:
        raise FileNotFoundError(f"Labeled image not found: {labeled_image_path}")
    
    response = client.models.generate_content(
        model=model_id,
        contents=[labeled_image, prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    
    cleaned_text = response.text.strip()
    if cleaned_text.startswith('```json'):
        cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
    elif cleaned_text.startswith('```'):
        cleaned_text = cleaned_text.replace('```', '')
    
    predicate_specs = json.loads(cleaned_text)
    
    grounded_atoms = []
    object_lookup = {obj.unique_id: obj for obj in objects}
    
    for spec in predicate_specs:
        pred_name = spec.get('name', '')
        arg_ids = spec.get('args', [])
        
        pred_objects = []
        for arg_id in arg_ids:
            if arg_id in object_lookup:
                pred_objects.append(object_lookup[arg_id])
            else:
                print(f"Warning: Object ID '{arg_id}' not found in detected objects")
        
        if pred_objects:
            grounded_atom = Atom(name=pred_name, object_args=pred_objects)
            grounded_atoms.append(grounded_atom)
    
    return grounded_atoms


def visualize_detections(image: Image.Image, results: list[dict], output_path: str | None = None, show_plot: bool = True) -> tuple:
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
            ax.text(
                xmin, ymin - 5, f"{i+1}. {label}",
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.8)
            )
    
    ax.set_title("Bounding Box Detection Results", fontsize=14, weight='bold')
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



