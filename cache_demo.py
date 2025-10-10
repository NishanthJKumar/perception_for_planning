#!/usr/bin/env python3
"""
Example script demonstrating Gemini API caching.

This script shows how to use the caching functionality with Gemini API calls.
It loads an image, makes repeated calls to the Gemini API, and shows how
caching improves performance.

Usage:
    python cache_demo.py [--no-cache] [--cache-dir /path/to/cache] [--clear-cache]
"""

import argparse
import time
from pathlib import Path
from PIL import Image

from gemini import detect_bboxes, setup_client
from gemini_cache import GeminiCache


def main(use_cache=True, cache_dir=None, clear_cache=False):
    """Run the caching demonstration.
    
    Args:
        use_cache: Whether to use caching.
        cache_dir: Directory to store cache files.
        clear_cache: Whether to clear the cache before running.
    """
    # Initialize cache
    cache = GeminiCache(cache_dir=cache_dir, enabled=use_cache)
    
    # Print cache statistics
    stats = cache.get_stats()
    print(f"Cache status:")
    print(f"  Enabled: {stats['enabled']}")
    print(f"  Directory: {stats['cache_dir']}")
    print(f"  Entries: {stats['entries']}")
    print(f"  Size: {stats.get('size_mb', 0):.2f} MB")
    
    # Clear cache if requested
    if clear_cache and stats['enabled']:
        cleared = cache.clear()
        print(f"Cleared {cleared} cache entries")
        stats = cache.get_stats()
        print(f"Cache now has {stats['entries']} entries")
    
    # Load test image
    data_dir = Path(__file__).parent
    image_path = data_dir / "example-rgb-img.png"
    if not image_path.exists():
        print(f"Error: Test image not found at {image_path}")
        return
        
    image = Image.open(image_path)
    print(f"Loaded test image: {image.width}x{image.height}")
    
    # Setup Gemini client
    client = setup_client()
    
    # Run first detection and measure time
    print("\nRunning first detection (cold)...")
    start_time = time.time()
    results = detect_bboxes(
        image, 
        client=client, 
        use_cache=use_cache,
        cache_dir=cache_dir
    )
    elapsed_time = time.time() - start_time
    print(f"First detection took {elapsed_time:.2f} seconds")
    print(f"Found {len(results)} objects")
    
    # Run second detection and measure time
    print("\nRunning second detection (should be cached if caching is enabled)...")
    start_time = time.time()
    results = detect_bboxes(
        image, 
        client=client, 
        use_cache=use_cache,
        cache_dir=cache_dir
    )
    elapsed_time = time.time() - start_time
    print(f"Second detection took {elapsed_time:.2f} seconds")
    print(f"Found {len(results)} objects")
    
    # Update cache statistics
    if use_cache:
        stats = cache.get_stats()
        print(f"\nUpdated cache statistics:")
        print(f"  Entries: {stats['entries']}")
        print(f"  Size: {stats.get('size_mb', 0):.2f} MB")
        
    print("\nDemonstration completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate Gemini API caching")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--cache-dir", type=str, help="Custom cache directory")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running")
    
    args = parser.parse_args()
    
    main(
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        clear_cache=args.clear_cache
    )