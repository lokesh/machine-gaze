"""
Classifier Registry for managing multiple classifiers in the Machine Gaze system.

This module implements the Registry pattern to allow dynamic registration
and management of different classifiers. It provides a clean interface
for the main processing pipeline to work with multiple classifiers without
knowing their specific implementations.
"""

from typing import Dict, List, Type, Any
import logging
from .base_classifier import BaseClassifier, Detection

logger = logging.getLogger(__name__)


class ClassifierRegistry:
    """
    Registry for managing and executing multiple classifiers.
    
    This class follows the Registry pattern and allows:
    - Dynamic registration of new classifier types
    - Configuration-driven classifier instantiation
    - Batch processing across all enabled classifiers
    - Easy addition/removal of classifiers without code changes
    """
    
    def __init__(self):
        self._classifier_types: Dict[str, Type[BaseClassifier]] = {}
        self._active_classifiers: Dict[str, BaseClassifier] = {}
    
    def register_classifier(self, name: str, classifier_class: Type[BaseClassifier]) -> None:
        """
        Register a new classifier type with the registry.
        
        Args:
            name: Unique identifier for the classifier
            classifier_class: Class implementing BaseClassifier interface
        """
        if not issubclass(classifier_class, BaseClassifier):
            raise ValueError(f"Classifier {name} must inherit from BaseClassifier")
        
        self._classifier_types[name] = classifier_class
        logger.info(f"Registered classifier: {name}")
    
    def create_classifier(self, name: str, config: Dict[str, Any]) -> BaseClassifier:
        """
        Create an instance of a registered classifier.
        
        Args:
            name: Name of the classifier type to create
            config: Configuration dictionary for the classifier
            
        Returns:
            Configured classifier instance
            
        Raises:
            KeyError: If classifier name is not registered
        """
        if name not in self._classifier_types:
            available = list(self._classifier_types.keys())
            raise KeyError(f"Classifier '{name}' not registered. Available: {available}")
        
        classifier_class = self._classifier_types[name]
        classifier = classifier_class(config)
        
        # Load the model if the classifier is enabled
        if classifier.is_enabled():
            logger.info(f"Loading model for classifier: {name}")
            classifier.load_model()
        
        return classifier
    
    def setup_from_config(self, config: Dict[str, Any]) -> None:
        """
        Set up all classifiers from a configuration dictionary.
        
        Args:
            config: Dictionary with classifier configurations.
                   Expected format: {'classifiers': {'classifier_name': {...}}}
        """
        classifiers_config = config.get('classifiers', {})
        
        for name, classifier_config in classifiers_config.items():
            try:
                classifier = self.create_classifier(name, classifier_config)
                if classifier.is_enabled():
                    self._active_classifiers[name] = classifier
                    logger.info(f"Activated classifier: {name}")
                else:
                    logger.info(f"Classifier {name} is disabled")
                    
            except Exception as e:
                logger.error(f"Failed to create classifier {name}: {e}")
                # Continue with other classifiers instead of failing completely
    
    def process_frame(self, frame) -> List[Detection]:
        """
        Run all active classifiers on a frame and aggregate results.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Combined list of all detections from all classifiers
        """
        all_detections = []
        
        for name, classifier in self._active_classifiers.items():
            try:
                detections = classifier.detect(frame)
                all_detections.extend(detections)
                logger.debug(f"Classifier {name} found {len(detections)} detections")
                
            except Exception as e:
                logger.error(f"Error in classifier {name}: {e}")
                # Continue with other classifiers
        
        return all_detections
    
    def get_active_classifiers(self) -> Dict[str, BaseClassifier]:
        """Get dictionary of currently active classifiers."""
        return self._active_classifiers.copy()
    
    def get_registered_types(self) -> List[str]:
        """Get list of registered classifier type names."""
        return list(self._classifier_types.keys())
    
    def disable_classifier(self, name: str) -> None:
        """Disable a classifier by removing it from active classifiers."""
        if name in self._active_classifiers:
            del self._active_classifiers[name]
            logger.info(f"Disabled classifier: {name}")
    
    def clear(self) -> None:
        """Clear all registered classifiers and active instances."""
        self._classifier_types.clear()
        self._active_classifiers.clear()
        logger.info("Cleared all classifiers from registry")


# Global registry instance for easy access
_global_registry = ClassifierRegistry()


def get_registry() -> ClassifierRegistry:
    """Get the global classifier registry instance."""
    return _global_registry


def register_classifier(name: str, classifier_class: Type[BaseClassifier]) -> None:
    """Convenience function to register a classifier with the global registry."""
    _global_registry.register_classifier(name, classifier_class)
