"""
Advanced configuration management for MACHINE GAZE.

Provides tools for creating, validating, and managing configurations with
support for templates, inheritance, and easy classifier management.
"""

import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class ClassifierTemplate:
    """Template for classifier configuration."""
    name: str
    description: str
    enabled: bool = True
    confidence_threshold: float = 0.5
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class ConfigTemplate:
    """Template for complete configuration."""
    name: str
    description: str
    video_processor: Dict[str, Any] = None
    classifiers: Dict[str, ClassifierTemplate] = None
    logging: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.video_processor is None:
            self.video_processor = {}
        if self.classifiers is None:
            self.classifiers = {}
        if self.logging is None:
            self.logging = {"level": "INFO"}


class ConfigManager:
    """
    Advanced configuration manager for MACHINE GAZE.
    
    Provides tools for:
    - Creating configurations from templates
    - Validating configuration files
    - Managing classifier presets
    - Configuration inheritance and merging
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Built-in classifier templates
        self.classifier_templates = self._load_builtin_templates()
        
        # Built-in configuration templates
        self.config_templates = self._load_builtin_config_templates()
        
        logger.info(f"Initialized ConfigManager with {len(self.classifier_templates)} classifier templates")
    
    def _load_builtin_templates(self) -> Dict[str, ClassifierTemplate]:
        """Load built-in classifier templates."""
        templates = {}
        
        # YOLO-World template
        templates["yolo_world"] = ClassifierTemplate(
            name="yolo_world",
            description="YOLO-World object detection with open vocabulary",
            enabled=True,
            confidence_threshold=0.5,
            parameters={
                "classes": ["person", "car", "bicycle", "motorcycle"],
                "model_size": "s",  # s, m, l, x
                "device": "auto"    # auto, cpu, mps, cuda
            }
        )
        
        # Face emotion template
        templates["face_emotion"] = ClassifierTemplate(
            name="face_emotion",
            description="Face detection with emotion recognition",
            enabled=False,
            confidence_threshold=0.3,
            parameters={
                "face_confidence": 0.5,
                "emotion_confidence": 0.3,
                "face_padding": 0.2,
                "min_face_size": 30,
                "model_path": None
            }
        )
        
        # Gesture recognition template (placeholder)
        templates["gesture_recognition"] = ClassifierTemplate(
            name="gesture_recognition",
            description="Hand gesture recognition (future extension)",
            enabled=False,
            confidence_threshold=0.6,
            parameters={
                "hand_confidence": 0.7,
                "gesture_classes": ["wave", "point", "thumbs_up", "peace"],
                "tracking_enabled": True
            }
        )
        
        # Activity recognition template (placeholder)
        templates["activity_recognition"] = ClassifierTemplate(
            name="activity_recognition",
            description="Human activity recognition (future extension)",
            enabled=False,
            confidence_threshold=0.4,
            parameters={
                "activity_classes": ["walking", "running", "sitting", "standing"],
                "temporal_window": 10,
                "pose_estimation": True
            }
        )
        
        return templates
    
    def _load_builtin_config_templates(self) -> Dict[str, ConfigTemplate]:
        """Load built-in configuration templates."""
        templates = {}
        
        # Basic object detection
        templates["basic_detection"] = ConfigTemplate(
            name="basic_detection",
            description="Basic object detection with YOLO-World",
            video_processor={
                "overlay_only": False,
                "background_color": [0, 0, 0],
                "tracking": {"enabled": False}
            },
            classifiers={
                "yolo_world": self.classifier_templates["yolo_world"]
            }
        )
        
        # Emotion analysis
        templates["emotion_analysis"] = ConfigTemplate(
            name="emotion_analysis",
            description="Object detection with emotion recognition",
            video_processor={
                "overlay_only": False,
                "face_person_association": True,
                "non_max_suppression": True,
                "tracking": {"enabled": False}
            },
            classifiers={
                "yolo_world": self.classifier_templates["yolo_world"],
                "face_emotion": self.classifier_templates["face_emotion"]
            }
        )
        
        # Advanced tracking
        templates["advanced_tracking"] = ConfigTemplate(
            name="advanced_tracking",
            description="Full tracking with emotion detection and trails",
            video_processor={
                "overlay_only": True,
                "background_color": [0, 0, 0],
                "face_person_association": True,
                "non_max_suppression": True,
                "tracking": {
                    "enabled": True,
                    "show_track_ids": True,
                    "show_trajectories": True,
                    "trajectory_length": 15,
                    "high_thresh": 0.6,
                    "low_thresh": 0.1,
                    "new_track_thresh": 0.7,
                    "temporal_window": 5,
                    "bbox_smoothing": True,
                    "confidence_smoothing": True,
                    "class_voting": True
                }
            },
            classifiers={
                "yolo_world": self.classifier_templates["yolo_world"],
                "face_emotion": self.classifier_templates["face_emotion"]
            }
        )
        
        # Security monitoring
        templates["security_monitoring"] = ConfigTemplate(
            name="security_monitoring",
            description="Security-focused configuration with person tracking",
            video_processor={
                "overlay_only": False,
                "tracking": {
                    "enabled": True,
                    "show_track_ids": True,
                    "show_trajectories": True,
                    "high_thresh": 0.7,
                    "new_track_thresh": 0.8
                }
            },
            classifiers={
                "yolo_world": ClassifierTemplate(
                    name="yolo_world",
                    description="Security-focused object detection",
                    confidence_threshold=0.6,
                    parameters={
                        "classes": ["person", "car", "motorcycle", "bicycle", "backpack", "handbag"],
                        "model_size": "l"  # Larger model for better accuracy
                    }
                )
            }
        )
        
        return templates
    
    def create_config_from_template(self, template_name: str, output_path: str, 
                                  modifications: Dict[str, Any] = None) -> bool:
        """
        Create a configuration file from a template.
        
        Args:
            template_name: Name of template to use
            output_path: Path for output configuration file
            modifications: Optional modifications to apply to template
            
        Returns:
            True if successful, False otherwise
        """
        if template_name not in self.config_templates:
            logger.error(f"Template '{template_name}' not found")
            return False
        
        try:
            # Get template
            template = deepcopy(self.config_templates[template_name])
            
            # Apply modifications
            if modifications:
                config_dict = self._template_to_dict(template)
                config_dict = self._merge_configs(config_dict, modifications)
            else:
                config_dict = self._template_to_dict(template)
            
            # Save configuration
            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created configuration file: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create configuration: {e}")
            return False
    
    def _template_to_dict(self, template: ConfigTemplate) -> Dict[str, Any]:
        """Convert template to dictionary format."""
        config_dict = {
            "video_processor": template.video_processor,
            "logging": template.logging,
            "classifiers": {}
        }
        
        # Convert classifier templates
        for name, classifier_template in template.classifiers.items():
            classifier_dict = {
                "enabled": classifier_template.enabled,
                "confidence_threshold": classifier_template.confidence_threshold
            }
            classifier_dict.update(classifier_template.parameters)
            config_dict["classifiers"][name] = classifier_dict
        
        return config_dict
    
    def _merge_configs(self, base_config: Dict[str, Any], 
                      modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration modifications into base configuration."""
        result = deepcopy(base_config)
        
        def merge_dict(base: Dict[str, Any], updates: Dict[str, Any]):
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(result, modifications)
        return result
    
    def add_classifier_to_config(self, config_path: str, classifier_name: str, 
                                template_name: str = None, parameters: Dict[str, Any] = None) -> bool:
        """
        Add a classifier to an existing configuration.
        
        Args:
            config_path: Path to configuration file
            classifier_name: Name for the new classifier
            template_name: Template to use (defaults to classifier_name)
            parameters: Additional parameters to set
            
        Returns:
            True if successful, False otherwise
        """
        if template_name is None:
            template_name = classifier_name
        
        if template_name not in self.classifier_templates:
            logger.error(f"Classifier template '{template_name}' not found")
            return False
        
        try:
            # Load existing configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Add classifier
            template = self.classifier_templates[template_name]
            classifier_config = {
                "enabled": template.enabled,
                "confidence_threshold": template.confidence_threshold
            }
            classifier_config.update(template.parameters)
            
            # Apply additional parameters
            if parameters:
                classifier_config.update(parameters)
            
            # Ensure classifiers section exists
            if "classifiers" not in config:
                config["classifiers"] = {}
            
            config["classifiers"][classifier_name] = classifier_config
            
            # Save updated configuration
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Added classifier '{classifier_name}' to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add classifier: {e}")
            return False
    
    def validate_config(self, config_path: str) -> Dict[str, Any]:
        """
        Validate a configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validation results with status and issues
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ["classifiers"]
            for section in required_sections:
                if section not in config:
                    result["errors"].append(f"Missing required section: {section}")
                    result["valid"] = False
            
            # Validate classifiers
            if "classifiers" in config:
                for name, classifier_config in config["classifiers"].items():
                    if not isinstance(classifier_config, dict):
                        result["errors"].append(f"Classifier '{name}' must be a dictionary")
                        continue
                    
                    # Check if classifier is known
                    if name not in self.classifier_templates:
                        result["warnings"].append(f"Unknown classifier type: {name}")
                    
                    # Check required fields
                    if "enabled" not in classifier_config:
                        result["warnings"].append(f"Classifier '{name}' missing 'enabled' field")
                    
                    if "confidence_threshold" not in classifier_config:
                        result["warnings"].append(f"Classifier '{name}' missing 'confidence_threshold' field")
            
            # Check for enabled classifiers
            enabled_classifiers = []
            if "classifiers" in config:
                enabled_classifiers = [name for name, cfg in config["classifiers"].items() 
                                     if cfg.get("enabled", False)]
            
            if not enabled_classifiers:
                result["warnings"].append("No classifiers are enabled")
            
            # Video processor suggestions
            if "video_processor" in config:
                video_config = config["video_processor"]
                
                # Tracking suggestions
                if "tracking" in video_config and video_config["tracking"].get("enabled", False):
                    if len(enabled_classifiers) == 1:
                        result["suggestions"].append("Consider enabling multiple classifiers for better tracking")
                
                # Overlay suggestions
                if video_config.get("overlay_only", False) and not video_config.get("background_color"):
                    result["suggestions"].append("Consider setting background_color for overlay mode")
            
        except yaml.YAMLError as e:
            result["errors"].append(f"YAML parsing error: {e}")
            result["valid"] = False
        except Exception as e:
            result["errors"].append(f"Validation error: {e}")
            result["valid"] = False
        
        return result
    
    def list_templates(self) -> Dict[str, Any]:
        """
        List all available templates.
        
        Returns:
            Dictionary with classifier and configuration templates
        """
        return {
            "classifier_templates": {name: template.description 
                                   for name, template in self.classifier_templates.items()},
            "config_templates": {name: template.description 
                               for name, template in self.config_templates.items()}
        }
    
    def export_template_docs(self, output_path: str) -> bool:
        """
        Export template documentation to a file.
        
        Args:
            output_path: Path for output documentation file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            docs = {
                "machine_gaze_templates": {
                    "classifier_templates": {
                        name: {
                            "description": template.description,
                            "default_enabled": template.enabled,
                            "default_confidence": template.confidence_threshold,
                            "parameters": template.parameters
                        }
                        for name, template in self.classifier_templates.items()
                    },
                    "config_templates": {
                        name: {
                            "description": template.description,
                            "example_config": self._template_to_dict(template)
                        }
                        for name, template in self.config_templates.items()
                    }
                }
            }
            
            output_path = Path(output_path)
            if output_path.suffix.lower() == '.json':
                with open(output_path, 'w') as f:
                    json.dump(docs, f, indent=2)
            else:
                with open(output_path, 'w') as f:
                    yaml.dump(docs, f, default_flow_style=False, indent=2)
            
            logger.info(f"Exported template documentation to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export documentation: {e}")
            return False
