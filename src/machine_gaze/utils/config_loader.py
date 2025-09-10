"""
Configuration management for Machine Gaze.

This module handles loading and validation of configuration files,
with support for YAML configs and environment-based overrides.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Handles loading and merging of configuration files.
    
    Supports YAML configuration files with environment-specific
    overrides and runtime parameter injection.
    """
    
    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, uses default config.
            
        Returns:
            Dictionary containing configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        if config_path is None:
            # Use default config
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "default_config.yaml"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from: {config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file {config_path}: {e}")
            raise
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary
            
        Returns:
            Merged configuration with overrides applied
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                # Override value
                merged[key] = value
        
        return merged
    
    @staticmethod
    def setup_logging(config: Dict[str, Any]) -> None:
        """
        Configure logging based on config settings.
        
        Args:
            config: Configuration dictionary with logging settings
        """
        logging_config = config.get('logging', {})
        
        level = logging_config.get('level', 'INFO')
        format_str = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_str,
            force=True  # Override any existing configuration
        )
        
        logger.info(f"Logging configured at {level} level")
