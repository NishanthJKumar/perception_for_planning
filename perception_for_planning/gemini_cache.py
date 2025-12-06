"""
Simple caching utility for Gemini API responses
"""
import hashlib
import io
import json
import pickle
from pathlib import Path
from typing import Any, Optional

from PIL import Image


class GeminiCache:
    """Simple cache for Gemini API responses."""
    
    def __init__(self, cache_dir: Optional[str] = None, enabled: bool = True):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files. If None, a default directory will be used.
            enabled: Whether caching is enabled.
        """
        self.enabled = enabled
        
        if cache_dir is None:
            # Default cache directory in user's home directory
            self.cache_dir = Path.home() / ".cache" / "gemini_api"
        else:
            self.cache_dir = Path(cache_dir)
            
        # Create cache directory if it doesn't exist
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_hash(self, *args: Any) -> str:
        """Compute a hash of the input arguments."""
        hasher = hashlib.sha256()
        
        for arg in args:
            if isinstance(arg, Image.Image):
                # Hash image data
                img_bytes = io.BytesIO()
                arg.save(img_bytes, format="PNG")
                hasher.update(img_bytes.getvalue())
            elif isinstance(arg, (str, bytes)):
                # Hash string or bytes directly
                hasher.update(str(arg).encode('utf-8'))
            elif isinstance(arg, (list, dict, tuple)):
                # Hash serializable objects
                hasher.update(json.dumps(arg, sort_keys=True).encode('utf-8'))
            else:
                # Try to hash other objects
                try:
                    hasher.update(str(arg).encode('utf-8'))
                except:
                    pass  # Skip if we can't hash it
                    
        return hasher.hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a cached response."""
        if not self.enabled:
            return None
            
        cache_file = self.cache_dir / f"{key}.pkl"
        
        # Check if cache file exists
        if not cache_file.exists():
            return None
                
        # Load cached data
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Store a response in the cache."""
        if not self.enabled:
            return
            
        cache_file = self.cache_dir / f"{key}.pkl"
        
        # Save data to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Failed to cache response: {e}")
            
    def clear(self) -> None:
        """Clear the entire cache."""
        if not self.enabled or not self.cache_dir.exists():
            return
            
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass