"""
Model downloader for Machine Gaze.

Downloads and caches pre-trained models for emotion detection and other tasks.
Provides a simple interface to download models from various sources.
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class ModelDownloader:
    """
    Downloads and manages pre-trained models for Machine Gaze.
    
    Provides caching, integrity checking, and automatic downloads
    of models needed by classifiers.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize model downloader.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        if cache_dir is None:
            # Default to ~/.machine_gaze/models/
            cache_dir = Path.home() / ".machine_gaze" / "models"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    def download_emotion_model(self, model_name: str = "fer2013_cnn") -> Path:
        """
        Download a pre-trained emotion recognition model.
        
        Args:
            model_name: Name of the emotion model to download
            
        Returns:
            Path to the downloaded model file
        """
        model_info = self._get_emotion_model_info(model_name)
        
        if not model_info:
            raise ValueError(f"Unknown emotion model: {model_name}")
        
        return self._download_model(
            name=model_name,
            url=model_info["url"],
            filename=model_info["filename"],
            expected_hash=model_info.get("sha256"),
            description=model_info.get("description", "Emotion recognition model")
        )
    
    def _get_emotion_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about available emotion models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information or None if not found
        """
        # Model registry - in production, this could be loaded from a config file
        models = {
            "fer2013_cnn": {
                "url": "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5",
                "filename": "fer2013_mini_XCEPTION.h5",
                "description": "FER2013 Mini Xception model for emotion recognition",
                "sha256": None  # Would include actual hash in production
            },
            "fer2013_simple": {
                "url": "https://example.com/simple_emotion_model.h5",  # Placeholder
                "filename": "simple_emotion_model.h5",
                "description": "Simple CNN for emotion recognition",
                "sha256": None
            }
        }
        
        return models.get(model_name)
    
    def _download_model(
        self, 
        name: str, 
        url: str, 
        filename: str,
        expected_hash: Optional[str] = None,
        description: str = "Model"
    ) -> Path:
        """
        Download a model file with caching and integrity checking.
        
        Args:
            name: Model name for logging
            url: Download URL
            filename: Local filename to save as
            expected_hash: Expected SHA256 hash for verification
            description: Human-readable description
            
        Returns:
            Path to the downloaded/cached model file
        """
        model_path = self.cache_dir / filename
        
        # Check if already cached
        if model_path.exists():
            if expected_hash and self._verify_hash(model_path, expected_hash):
                logger.info(f"Using cached {description}: {model_path}")
                return model_path
            elif not expected_hash:
                logger.info(f"Using cached {description}: {model_path}")
                return model_path
            else:
                logger.warning(f"Cached model {model_path} hash mismatch, re-downloading")
                model_path.unlink()
        
        # Download the model
        logger.info(f"Downloading {description} from {url}")
        
        try:
            # Create a temporary file for downloading
            temp_path = model_path.with_suffix(model_path.suffix + ".tmp")
            
            # Download with progress
            urllib.request.urlretrieve(url, temp_path)
            
            # Verify hash if provided
            if expected_hash and not self._verify_hash(temp_path, expected_hash):
                temp_path.unlink()
                raise RuntimeError(f"Downloaded model hash verification failed for {name}")
            
            # Move to final location
            temp_path.rename(model_path)
            
            logger.info(f"Successfully downloaded {description}: {model_path}")
            return model_path
            
        except urllib.error.URLError as e:
            logger.error(f"Failed to download {description}: {e}")
            raise RuntimeError(f"Could not download {name} from {url}: {e}")
        except Exception as e:
            logger.error(f"Error downloading {description}: {e}")
            raise
    
    def _verify_hash(self, file_path: Path, expected_hash: str) -> bool:
        """
        Verify file SHA256 hash.
        
        Args:
            file_path: Path to file to verify
            expected_hash: Expected SHA256 hash
            
        Returns:
            True if hash matches, False otherwise
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            actual_hash = sha256_hash.hexdigest()
            return actual_hash.lower() == expected_hash.lower()
            
        except Exception as e:
            logger.error(f"Error verifying hash for {file_path}: {e}")
            return False
    
    def list_cached_models(self) -> list[Path]:
        """
        List all cached model files.
        
        Returns:
            List of paths to cached model files
        """
        if not self.cache_dir.exists():
            return []
        
        # Look for common model file extensions
        model_extensions = [".h5", ".hdf5", ".pb", ".pth", ".pt", ".onnx"]
        models = []
        
        for ext in model_extensions:
            models.extend(self.cache_dir.glob(f"*{ext}"))
        
        return sorted(models)
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        if self.cache_dir.exists():
            for model_file in self.list_cached_models():
                model_file.unlink()
                logger.info(f"Removed cached model: {model_file}")


# Global downloader instance
_global_downloader = ModelDownloader()


def get_model_downloader() -> ModelDownloader:
    """Get the global model downloader instance."""
    return _global_downloader


def download_emotion_model(model_name: str = "fer2013_cnn") -> Path:
    """Convenience function to download an emotion model."""
    return _global_downloader.download_emotion_model(model_name)
