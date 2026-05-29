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

# Known configuration keys, used only for typo-warnings. Unknown keys are still
# ignored silently by the pipeline (config is read with .get(key, default)); this
# schema just surfaces likely mistakes for an art project that hand-edits YAML a lot.
KNOWN_TOP_LEVEL = {"output", "video_processor", "classifiers", "logging", "grid"}
KNOWN_OUTPUT = {"fps", "codec", "quality"}
KNOWN_VIDEO_PROCESSOR = {
    "font_scale", "font_thickness", "bbox_thickness", "text_color", "bbox_color",
    "overlay_only", "background_color", "face_person_association",
    "non_max_suppression", "nms_threshold", "tracking", "grid",
    "output_fps", "output_codec", "output_quality",
}
KNOWN_TRACKING = {
    "enabled", "high_thresh", "low_thresh", "new_track_thresh", "track_buffer",
    "match_thresh", "show_track_ids", "show_trajectories", "trajectory_length",
    "temporal_window", "bbox_smoothing", "confidence_smoothing", "class_voting",
}
KNOWN_GRID = {
    "enabled", "rows", "cols", "mode", "show_gridlines", "gridline_color",
    "gridline_thickness", "background_color", "min_cells", "ghost_decay",
    "census_colormap",
}
KNOWN_LOGGING = {"level", "format"}


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

            config = config or {}
            logger.info(f"Loaded configuration from: {config_path}")
            ConfigLoader.validate_config(config)
            return config

        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file {config_path}: {e}")
            raise

    @staticmethod
    def _warn_unknown(section: str, keys, allowed) -> None:
        unknown = [k for k in keys if k not in allowed]
        for key in unknown:
            logger.warning(
                "Unknown config key '%s' in '%s' - will be ignored (typo?)",
                key, section,
            )

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Warn about unrecognized config keys to catch typos.

        Non-fatal by design: the pipeline treats unknown keys as absent, so a
        misspelled key silently disables a feature. This surfaces those as
        warnings without changing behavior. Classifier sub-configs are not
        checked here because classifier names/keys are open-ended.
        """
        if not isinstance(config, dict):
            return

        ConfigLoader._warn_unknown("(top level)", config.keys(), KNOWN_TOP_LEVEL)

        if isinstance(config.get("output"), dict):
            ConfigLoader._warn_unknown("output", config["output"].keys(), KNOWN_OUTPUT)

        if isinstance(config.get("logging"), dict):
            ConfigLoader._warn_unknown("logging", config["logging"].keys(), KNOWN_LOGGING)

        if isinstance(config.get("grid"), dict):
            ConfigLoader._warn_unknown("grid", config["grid"].keys(), KNOWN_GRID)

        vp = config.get("video_processor")
        if isinstance(vp, dict):
            ConfigLoader._warn_unknown("video_processor", vp.keys(), KNOWN_VIDEO_PROCESSOR)
            if isinstance(vp.get("tracking"), dict):
                ConfigLoader._warn_unknown(
                    "video_processor.tracking", vp["tracking"].keys(), KNOWN_TRACKING
                )
            if isinstance(vp.get("grid"), dict):
                ConfigLoader._warn_unknown("video_processor.grid", vp["grid"].keys(), KNOWN_GRID)

    @staticmethod
    def video_processor_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the dict passed to VideoProcessor.

        Merges the top-level ``output:`` block (codec/fps/quality) and a
        top-level ``grid:`` block into the ``video_processor`` config, so those
        settings actually reach the processor regardless of where the user put
        them. Explicit ``video_processor`` keys win over the top-level blocks.
        """
        vp = dict(config.get("video_processor") or {})
        output = config.get("output") or {}

        mapping = {"codec": "output_codec", "fps": "output_fps", "quality": "output_quality"}
        for src, dst in mapping.items():
            if dst not in vp and src in output:
                vp[dst] = output[src]

        if "grid" not in vp and "grid" in config:
            vp["grid"] = config["grid"]

        return vp
    
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
