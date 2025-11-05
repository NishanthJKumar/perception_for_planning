import json
import os
from functools import cache
from typing import Optional, Dict, List, Tuple

import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from matplotlib import pyplot as plt, patches

# Import the caching utility
from .gemini_cache import GeminiCache


@cache
def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    prompt_path = os.path.join(prompts_dir, f"{prompt_name}.txt")
    with open(prompt_path, "r") as f:
        return f.read().strip()


@cache
def setup_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is required. Get your API key from https://aistudio.google.com/apikey"
        )

    return genai.Client(api_key=api_key)


def load_json(response_text: str) -> list | dict:
    """Extract JSON string from code fencing if present."""
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.replace("```", "")

    results = json.loads(cleaned_text)
    return results


def detect_bboxes(
    image: Image.Image,
    task_instruction: str,
    client: genai.Client | None = None,
    model_id: str = "gemini-robotics-er-1.5-preview",
    temperature: float | None = None,
    use_cache: bool = True,
    cache_dir: Optional[str] = None
) -> list:
    """Detect objects in an image using Gemini API with caching support.

    Args:
        image: The image to analyze.
        task_instruction: Natural language task instruction to focus detection on relevant objects.
        client: Gemini API client. If None, a new client will be created.
        model_id: Gemini model ID to use.
        temperature: Temperature for generation.
        use_cache: Whether to use caching.
        cache_dir: Directory to store cache files. If None, a default directory will be used.

    Returns:
        List of detected objects with bounding boxes.
    """
    # Initialize cache
    cache = GeminiCache(cache_dir=cache_dir, enabled=use_cache)

    # Create cache key based on image content, task instruction, and model settings
    cache_key = cache.compute_hash(image, task_instruction, model_id, temperature)
    
    # Try to get from cache first
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result
        
    # If not in cache, call the API
    client = client if client is not None else setup_client()
    prompt_template = load_prompt("detect_bboxes")
    prompt = prompt_template.format(task_instruction=task_instruction)

    response = client.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    bbox_list: list = load_json(response.text)
    
    # Cache the result
    cache.put(cache_key, bbox_list)
    
    return bbox_list


def translate_task(
    task_instruction: str, 
    bboxes: list[dict], 
    bbox_viz_path: str,
    use_cache: bool = True,
    cache_dir: Optional[str] = None
) -> list[dict]:
    """Translate a natural language task into formal predicates using Gemini API with caching support.
    
    Args:
        task_instruction: The natural language task to translate.
        bboxes: List of bounding boxes with labels.
        bbox_viz_path: Path to the image with labeled bounding boxes.
        use_cache: Whether to use caching.
        cache_dir: Directory to store cache files. If None, a default directory will be used.
        
    Returns:
        List of predicate specifications.
    """
    # Initialize cache
    cache = GeminiCache(cache_dir=cache_dir, enabled=use_cache)
    
    # Load the image to compute the hash
    labeled_image = Image.open(bbox_viz_path)
    
    # Create cache key based on task instruction, bboxes, and image content
    bbox_labels = [bbox["label"] for bbox in bboxes]
    cache_key = cache.compute_hash(task_instruction, bbox_labels, labeled_image)
    
    # Try to get from cache first
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # If not in cache, call the API
    client = setup_client()

    # Load prompt template and format with task-specific values
    prompt_template = load_prompt("translate_task")
    object_list = chr(10).join(f"- {bbox['label']}" for bbox in bboxes)
    prompt = prompt_template.format(task_instruction=task_instruction, object_list=object_list)

    response = client.models.generate_content(
        model="gemini-robotics-er-1.5-preview",
        contents=[labeled_image, prompt],
        config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)),
    )
    predicate_specs = load_json(response.text)

    # Convert to simple dict format: [{"predicate": "on", "args": ["obj1", "obj2"]}, ...]
    grounded_atoms = []
    for spec in predicate_specs:
        pred_name = spec.get("name", "")
        args = spec.get("args", [])
        if pred_name and args:
            grounded_atoms.append({"predicate": pred_name, "args": args})
    
    # Cache the result
    cache.put(cache_key, grounded_atoms)
    
    return grounded_atoms


def detect_and_translate(
    image: Image.Image,
    task_instruction: str,
    client: genai.Client | None = None,
    model_id: str = "gemini-robotics-er-1.5-preview",
    temperature: float | None = None,
    use_cache: bool = True,
    cache_dir: Optional[str] = None
) -> Tuple[List[Dict], List[Dict]]:
    """Detect objects and translate task in a single Gemini API call with caching support.
    
    Args:
        image: The image to analyze.
        task_instruction: The natural language task to translate.
        client: Gemini API client. If None, a new client will be created.
        model_id: Gemini model ID to use.
        temperature: Temperature for generation.
        use_cache: Whether to use caching.
        cache_dir: Directory to store cache files. If None, a default directory will be used.
        
    Returns:
        Tuple of (bboxes, grounded_atoms) where:
        - bboxes: List of detected objects with bounding boxes
        - grounded_atoms: List of predicate specifications
    """
    # Initialize cache
    cache = GeminiCache(cache_dir=cache_dir, enabled=use_cache)
    
    # Create cache key based on image content, task, and model settings
    cache_key = cache.compute_hash(image, task_instruction, model_id, temperature)
    
    # Try to get from cache first
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result["bboxes"], cached_result["grounded_atoms"]
    
    # If not in cache, call the API
    client = client if client is not None else setup_client()

    # Load prompt template and format with task-specific values
    prompt_template = load_prompt("detect_and_translate")
    prompt = prompt_template.format(task_instruction=task_instruction)

    response = client.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, 
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    
    result = load_json(response.text)
    bboxes = result.get("bboxes", [])
    predicate_specs = result.get("predicates", [])
    
    # Convert predicates to grounded atoms format
    grounded_atoms = []
    for spec in predicate_specs:
        pred_name = spec.get("name", "")
        args = spec.get("args", [])
        if pred_name and args:
            grounded_atoms.append({"predicate": pred_name, "args": args})
    
    # Cache the result
    cache.put(cache_key, {"bboxes": bboxes, "grounded_atoms": grounded_atoms})
    
    return bboxes, grounded_atoms