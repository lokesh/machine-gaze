#!/usr/bin/env python3
"""
Machine Gaze - Main CLI Interface

Simple command-line interface for processing videos through
the Machine Gaze computer vision pipeline.

Usage:
    python main.py input_video.mp4 output_video.mp4 [--config config.yaml]
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to Python path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from machine_gaze import VideoProcessor, ConfigLoader, get_registry


def main():
    """Main entry point for the Machine Gaze CLI."""
    
    parser = argparse.ArgumentParser(
        description="Machine Gaze - Computer Vision Analysis for Video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default config
    python main.py videos/input.mp4 output/annotated.mp4
    
    # Using custom configuration
    python main.py input.mp4 output.mp4 --config config/custom.yaml
    
    # Show only detection overlays (no original video)
    python main.py input.mp4 output/overlay.mp4 --overlay-only
    
    # Enable debug logging
    python main.py input.mp4 output.mp4 --debug
        """
    )
    
    parser.add_argument(
        "input_video",
        help="Path to input video file"
    )
    
    parser.add_argument(
        "output_video", 
        help="Path for output annotated video"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration YAML file (default: config/default_config.yaml)"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--overlay-only", "-o",
        action="store_true",
        help="Show only detection overlays without original video (black background)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = ConfigLoader.load_config(args.config)
        
        # Override logging level if debug requested
        if args.debug:
            config.setdefault('logging', {})['level'] = 'DEBUG'
        
        # Setup logging
        ConfigLoader.setup_logging(config)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Machine Gaze video processing")
        logger.info(f"Input: {args.input_video}")
        logger.info(f"Output: {args.output_video}")
        
        # Validate input file
        input_path = Path(args.input_video)
        if not input_path.exists():
            logger.error(f"Input video not found: {input_path}")
            return 1
        
        # Setup classifier registry from config
        registry = get_registry()
        registry.setup_from_config(config)
        
        active_classifiers = registry.get_active_classifiers()
        logger.info(f"Active classifiers: {list(active_classifiers.keys())}")
        
        if not active_classifiers:
            logger.error("No active classifiers found. Check configuration.")
            return 1
        
        # Create video processor
        video_config = ConfigLoader.video_processor_config(config)

        # Apply overlay-only setting from command line
        if args.overlay_only:
            video_config['overlay_only'] = True
            logger.info("Overlay-only mode enabled - showing detections without original video")
        
        processor = VideoProcessor(registry, video_config)
        
        # Set up progress callback
        def progress_callback(current: int, total: int):
            if current % 50 == 0:  # Log every 50 frames
                percentage = (current / total) * 100
                logger.info(f"Progress: {current}/{total} frames ({percentage:.1f}%)")
        
        processor.set_progress_callback(progress_callback)
        
        # Process video
        success = processor.process_video(args.input_video, args.output_video)
        
        if success:
            logger.info("Video processing completed successfully!")
            logger.info(f"Annotated video saved to: {args.output_video}")
            return 0
        else:
            logger.error("Video processing failed")
            return 1
            
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error: {e}")
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
