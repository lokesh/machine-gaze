#!/usr/bin/env python3
"""
MACHINE GAZE - Advanced Processing Script

Enhanced main script with support for multiple output formats,
detailed analysis, and comprehensive reporting.
"""

import argparse
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from machine_gaze import VideoProcessor, ConfigLoader, get_registry
from machine_gaze.output.export_manager import ExportManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedVideoProcessor:
    """
    Advanced video processor with multiple output formats and analysis.
    """
    
    def __init__(self, config_path: str, output_dir: str = "advanced_output"):
        """
        Initialize advanced processor.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory for all outputs
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config(config_path)
        
        # Set up registry and processor
        self.registry = get_registry()
        self.registry.setup_from_config(self.config)
        
        self.processor = VideoProcessor(self.registry, self.config.get('video_processor', {}))
        
        # Initialize export manager
        self.export_manager = ExportManager(str(self.output_dir))
        
        logger.info(f"Initialized AdvancedVideoProcessor with config: {config_path}")
    
    def process_video(self, input_path: str, output_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process video with advanced output options.
        
        Args:
            input_path: Path to input video
            output_options: Dictionary of output options
            
        Returns:
            Processing results and output file paths
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        logger.info(f"Processing video: {input_path}")
        start_time = time.time()
        
        # Configure export manager
        if 'video' in output_options:
            video_opts = output_options['video']
            self.export_manager.configure_video_output(
                codec=video_opts.get('codec', 'mp4v'),
                fps=video_opts.get('fps', 30.0),
                quality=video_opts.get('quality', 0)
            )
        
        if 'images' in output_options:
            image_opts = output_options['images']
            self.export_manager.configure_image_output(
                format=image_opts.get('format', 'png'),
                quality=image_opts.get('quality', 95)
            )
        
        # Open input video
        import cv2
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
            
            # Start video output if requested
            if 'video' in output_options:
                video_path = self.output_dir / output_options['video']['filename']
                self.export_manager.start_video_output(str(video_path), width, height, fps)
            
            # Process frames
            frame_count = 0
            save_images = 'images' in output_options and output_options['images'].get('save_frames', False)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, detections = self.processor.process_single_frame(frame)
                
                # Export frame
                self.export_manager.write_frame(
                    annotated_frame, detections, frame_count, save_image=save_images
                )
                
                frame_count += 1
                
                # Progress logging
                if frame_count % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
            
            # Finish video output
            self.export_manager.finish_video_output()
            
            processing_time = time.time() - start_time
            logger.info(f"Processing complete: {frame_count} frames in {processing_time:.1f} seconds")
            
            # Generate requested outputs
            output_files = self.export_manager.get_output_files().copy()
            
            # Export data if requested
            if 'data' in output_options:
                data_opts = output_options['data']
                for format in data_opts.get('formats', ['json']):
                    filename = data_opts.get('filename', f"detections.{format}")
                    data_path = self.export_manager.export_detection_data(format, filename)
                    output_files.append(data_path)
            
            # Generate reports if requested
            if 'reports' in output_options:
                report_opts = output_options['reports']
                for format in report_opts.get('formats', ['html']):
                    filename = report_opts.get('filename', f"analysis_report.{format}")
                    report_path = self.export_manager.generate_analysis_report(format, filename)
                    output_files.append(report_path)
            
            return {
                'status': 'success',
                'frames_processed': frame_count,
                'processing_time': processing_time,
                'fps': frame_count / processing_time if processing_time > 0 else 0,
                'output_files': output_files,
                'input_video': str(input_path),
                'output_directory': str(self.output_dir)
            }
            
        finally:
            cap.release()
            self.export_manager.cleanup()


def main():
    """Main CLI interface for advanced processing."""
    parser = argparse.ArgumentParser(
        description="MACHINE GAZE - Advanced Video Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Format Examples:

# Basic video output
python advanced_main.py input.mp4 --config config/tracking_config.yaml --video-output output.mp4

# Multiple formats
python advanced_main.py input.mp4 --config config/emotion_detection.yaml \\
  --video-output processed.mp4 \\
  --data-export json,csv \\
  --image-sequence png \\
  --analysis-report html,txt

# Custom settings
python advanced_main.py input.mp4 --config config/tracking_config.yaml \\
  --video-output processed.mp4 --video-codec XVID --video-fps 25 \\
  --image-sequence jpg --image-quality 85 \\
  --output-dir custom_output/

Supported Formats:
  Video: mp4, avi, mov (codecs: mp4v, XVID, etc.)
  Images: png, jpg, tiff
  Data: json, csv
  Reports: html, txt
        """
    )
    
    # Required arguments
    parser.add_argument('input', help='Input video file')
    parser.add_argument('--config', '-c', required=True, help='Configuration file')
    
    # Output options
    parser.add_argument('--output-dir', '-o', default='advanced_output', 
                       help='Output directory for all files')
    
    # Video output
    parser.add_argument('--video-output', help='Output video filename')
    parser.add_argument('--video-codec', default='mp4v', help='Video codec')
    parser.add_argument('--video-fps', type=float, help='Output video FPS')
    parser.add_argument('--video-quality', type=int, default=0, help='Video quality (0=lossless)')
    
    # Image sequence output
    parser.add_argument('--image-sequence', help='Image format for frame sequence (png, jpg, tiff)')
    parser.add_argument('--image-quality', type=int, default=95, help='Image quality for lossy formats')
    
    # Data export
    parser.add_argument('--data-export', help='Data export formats (comma-separated: json,csv)')
    parser.add_argument('--data-filename', help='Custom filename for data export')
    
    # Analysis reports
    parser.add_argument('--analysis-report', help='Report formats (comma-separated: html,txt)')
    parser.add_argument('--report-filename', help='Custom filename for analysis report')
    
    # Processing options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (errors only)')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Build output options
        output_options = {}
        
        # Video output
        if args.video_output:
            output_options['video'] = {
                'filename': args.video_output,
                'codec': args.video_codec,
                'quality': args.video_quality
            }
            if args.video_fps:
                output_options['video']['fps'] = args.video_fps
        
        # Image sequence
        if args.image_sequence:
            output_options['images'] = {
                'save_frames': True,
                'format': args.image_sequence,
                'quality': args.image_quality
            }
        
        # Data export
        if args.data_export:
            formats = [f.strip() for f in args.data_export.split(',')]
            output_options['data'] = {
                'formats': formats
            }
            if args.data_filename:
                output_options['data']['filename'] = args.data_filename
        
        # Analysis reports
        if args.analysis_report:
            formats = [f.strip() for f in args.analysis_report.split(',')]
            output_options['reports'] = {
                'formats': formats
            }
            if args.report_filename:
                output_options['reports']['filename'] = args.report_filename
        
        # Default to video output if no outputs specified
        if not output_options:
            input_path = Path(args.input)
            default_output = f"{input_path.stem}_processed.mp4"
            output_options['video'] = {
                'filename': default_output,
                'codec': args.video_codec,
                'quality': args.video_quality
            }
            print(f"No output format specified, defaulting to video: {default_output}")
        
        # Initialize processor
        processor = AdvancedVideoProcessor(args.config, args.output_dir)
        
        # Process video
        print(f"🎬 Processing: {args.input}")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"⚙️  Configuration: {args.config}")
        
        results = processor.process_video(args.input, output_options)
        
        # Print results
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE")
        print("="*60)
        print(f"📊 Frames processed: {results['frames_processed']}")
        print(f"⏱️  Processing time: {results['processing_time']:.1f} seconds")
        print(f"🚀 Processing speed: {results['fps']:.1f} FPS")
        print(f"📁 Output directory: {results['output_directory']}")
        
        print(f"\n📄 Generated files:")
        for file_path in results['output_files']:
            file_size = Path(file_path).stat().st_size / (1024*1024)  # MB
            print(f"   - {Path(file_path).name} ({file_size:.1f} MB)")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
