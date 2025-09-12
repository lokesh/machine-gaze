#!/usr/bin/env python3
"""
MACHINE GAZE - Batch Video Processor

Advanced batch processing tool for processing multiple videos with different configurations.
Supports parallel processing, progress tracking, and comprehensive reporting.
"""

import argparse
import sys
import logging
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass, asdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from machine_gaze import VideoProcessor, ConfigLoader, get_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingJob:
    """Represents a single video processing job."""
    input_path: str
    output_path: str
    config_path: str
    job_id: str
    
    # Processing results
    status: str = "pending"  # pending, processing, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    total_frames: Optional[int] = None
    processed_frames: Optional[int] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Processing duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def fps(self) -> Optional[float]:
        """Processing speed in frames per second."""
        if self.duration and self.total_frames:
            return self.total_frames / self.duration
        return None


class BatchProcessor:
    """
    Batch processor for multiple video processing jobs.
    
    Supports parallel processing, progress tracking, and comprehensive reporting.
    """
    
    def __init__(self, max_workers: int = 2, output_dir: str = "batch_output"):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum number of parallel processing jobs
            output_dir: Directory for batch processing outputs
        """
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.jobs: List[ProcessingJob] = []
        self.results: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized BatchProcessor with {max_workers} workers")
    
    def add_job(self, input_path: str, output_path: str, config_path: str, job_id: Optional[str] = None) -> str:
        """
        Add a processing job to the batch.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            config_path: Path to configuration file
            job_id: Optional job identifier
            
        Returns:
            Job ID for tracking
        """
        if job_id is None:
            job_id = f"job_{len(self.jobs) + 1:03d}"
        
        job = ProcessingJob(
            input_path=input_path,
            output_path=output_path,
            config_path=config_path,
            job_id=job_id
        )
        
        self.jobs.append(job)
        logger.info(f"Added job {job_id}: {input_path} -> {output_path}")
        return job_id
    
    def add_jobs_from_csv(self, csv_path: str) -> List[str]:
        """
        Add multiple jobs from a CSV file.
        
        CSV format: input_path,output_path,config_path,job_id
        
        Args:
            csv_path: Path to CSV file with job definitions
            
        Returns:
            List of added job IDs
        """
        job_ids = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                job_id = self.add_job(
                    input_path=row['input_path'],
                    output_path=row['output_path'],
                    config_path=row['config_path'],
                    job_id=row.get('job_id')
                )
                job_ids.append(job_id)
        
        logger.info(f"Added {len(job_ids)} jobs from {csv_path}")
        return job_ids
    
    def add_directory_batch(self, input_dir: str, output_dir: str, config_path: str, 
                          video_extensions: List[str] = None) -> List[str]:
        """
        Add all videos from a directory as batch jobs.
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory for output videos
            config_path: Configuration file to use for all videos
            video_extensions: List of video file extensions to process
            
        Returns:
            List of added job IDs
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        job_ids = []
        
        for video_file in input_path.iterdir():
            if video_file.suffix.lower() in video_extensions:
                output_file = output_path / f"{video_file.stem}_processed{video_file.suffix}"
                job_id = self.add_job(
                    input_path=str(video_file),
                    output_path=str(output_file),
                    config_path=config_path,
                    job_id=f"batch_{video_file.stem}"
                )
                job_ids.append(job_id)
        
        logger.info(f"Added {len(job_ids)} videos from directory {input_dir}")
        return job_ids
    
    def _process_single_job(self, job: ProcessingJob) -> ProcessingJob:
        """
        Process a single job.
        
        Args:
            job: Processing job to execute
            
        Returns:
            Updated job with results
        """
        job.start_time = time.time()
        job.status = "processing"
        
        try:
            logger.info(f"Starting job {job.job_id}: {job.input_path}")
            
            # Load configuration
            config_loader = ConfigLoader()
            config = config_loader.load_config(job.config_path)
            
            # Set up registry and processor
            registry = get_registry()
            registry.setup_from_config(config)
            
            processor = VideoProcessor(registry, config.get('video_processor', {}))
            
            # Process video
            success = processor.process_video(job.input_path, job.output_path)
            
            if success:
                job.status = "completed"
                logger.info(f"Completed job {job.job_id}")
            else:
                job.status = "failed"
                job.error_message = "Video processing returned False"
                logger.error(f"Failed job {job.job_id}: Processing returned False")
                
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            logger.error(f"Failed job {job.job_id}: {e}")
        
        finally:
            job.end_time = time.time()
        
        return job
    
    def process_all(self, parallel: bool = True) -> Dict[str, Any]:
        """
        Process all jobs in the batch.
        
        Args:
            parallel: Whether to process jobs in parallel
            
        Returns:
            Summary of batch processing results
        """
        if not self.jobs:
            logger.warning("No jobs to process")
            return {"status": "no_jobs", "jobs_processed": 0}
        
        logger.info(f"Processing {len(self.jobs)} jobs {'in parallel' if parallel else 'sequentially'}")
        start_time = time.time()
        
        if parallel and self.max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                future_to_job = {executor.submit(self._process_single_job, job): job for job in self.jobs}
                
                # Collect results as they complete
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        updated_job = future.result()
                        # Update job in list
                        job_index = next(i for i, j in enumerate(self.jobs) if j.job_id == updated_job.job_id)
                        self.jobs[job_index] = updated_job
                    except Exception as e:
                        logger.error(f"Error processing job {job.job_id}: {e}")
                        job.status = "failed"
                        job.error_message = str(e)
        else:
            # Sequential processing
            for i, job in enumerate(self.jobs):
                logger.info(f"Processing job {i+1}/{len(self.jobs)}: {job.job_id}")
                self.jobs[i] = self._process_single_job(job)
        
        end_time = time.time()
        
        # Generate summary
        summary = self._generate_summary(start_time, end_time)
        self._save_results()
        
        return summary
    
    def _generate_summary(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Generate processing summary."""
        total_jobs = len(self.jobs)
        completed_jobs = len([j for j in self.jobs if j.status == "completed"])
        failed_jobs = len([j for j in self.jobs if j.status == "failed"])
        
        summary = {
            "status": "completed",
            "total_duration": end_time - start_time,
            "jobs_total": total_jobs,
            "jobs_completed": completed_jobs,
            "jobs_failed": failed_jobs,
            "success_rate": completed_jobs / total_jobs if total_jobs > 0 else 0,
            "average_job_duration": sum(j.duration for j in self.jobs if j.duration) / completed_jobs if completed_jobs > 0 else 0,
            "failed_jobs": [{"job_id": j.job_id, "error": j.error_message} for j in self.jobs if j.status == "failed"]
        }
        
        logger.info(f"Batch processing complete: {completed_jobs}/{total_jobs} jobs successful")
        return summary
    
    def _save_results(self):
        """Save detailed results to files."""
        # Save as JSON
        results_json = {
            "jobs": [asdict(job) for job in self.jobs],
            "summary": self._generate_summary(0, 0)  # Quick summary
        }
        
        json_path = self.output_dir / "batch_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save as CSV
        csv_path = self.output_dir / "batch_results.csv"
        with open(csv_path, 'w', newline='') as f:
            if self.jobs:
                writer = csv.DictWriter(f, fieldnames=asdict(self.jobs[0]).keys())
                writer.writeheader()
                for job in self.jobs:
                    writer.writerow(asdict(job))
        
        logger.info(f"Results saved to {json_path} and {csv_path}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current batch processing status."""
        return {
            "total_jobs": len(self.jobs),
            "completed": len([j for j in self.jobs if j.status == "completed"]),
            "failed": len([j for j in self.jobs if j.status == "failed"]),
            "pending": len([j for j in self.jobs if j.status == "pending"]),
            "processing": len([j for j in self.jobs if j.status == "processing"])
        }


def create_example_batch_csv():
    """Create an example batch processing CSV file."""
    example_csv = Path("example_batch.csv")
    
    with open(example_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['input_path', 'output_path', 'config_path', 'job_id'])
        writer.writerow(['videos/video1.mp4', 'batch_output/video1_processed.mp4', 'config/tracking_config.yaml', 'video1_tracking'])
        writer.writerow(['videos/video2.mp4', 'batch_output/video2_processed.mp4', 'config/emotion_detection.yaml', 'video2_emotion'])
        writer.writerow(['videos/video3.mp4', 'batch_output/video3_processed.mp4', 'config/simple_tracking.yaml', 'video3_simple'])
    
    print(f"Created example batch CSV: {example_csv}")
    print("Edit this file with your actual video paths and configurations.")


def main():
    """Main CLI interface for batch processing."""
    parser = argparse.ArgumentParser(
        description="MACHINE GAZE - Batch Video Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python batch_processor.py single input.mp4 output.mp4 config/tracking_config.yaml
  
  # Process multiple videos from CSV
  python batch_processor.py csv batch_jobs.csv --workers 3
  
  # Process all videos in directory
  python batch_processor.py directory videos/ processed_videos/ config/emotion_detection.yaml
  
  # Create example CSV file
  python batch_processor.py create-example
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Processing mode')
    
    # Single video processing
    single_parser = subparsers.add_parser('single', help='Process single video')
    single_parser.add_argument('input', help='Input video path')
    single_parser.add_argument('output', help='Output video path')
    single_parser.add_argument('config', help='Configuration file path')
    single_parser.add_argument('--job-id', help='Job identifier')
    
    # CSV batch processing
    csv_parser = subparsers.add_parser('csv', help='Process videos from CSV file')
    csv_parser.add_argument('csv_file', help='CSV file with job definitions')
    csv_parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
    csv_parser.add_argument('--output-dir', default='batch_output', help='Output directory for results')
    
    # Directory batch processing
    dir_parser = subparsers.add_parser('directory', help='Process all videos in directory')
    dir_parser.add_argument('input_dir', help='Input directory with videos')
    dir_parser.add_argument('output_dir', help='Output directory for processed videos')
    dir_parser.add_argument('config', help='Configuration file to use for all videos')
    dir_parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
    dir_parser.add_argument('--extensions', nargs='+', default=['.mp4', '.mov', '.avi'], help='Video file extensions to process')
    
    # Create example
    subparsers.add_parser('create-example', help='Create example batch CSV file')
    
    # Common arguments
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.command == 'create-example':
        create_example_batch_csv()
        return
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize batch processor
    workers = getattr(args, 'workers', 1)
    output_dir = getattr(args, 'output_dir', 'batch_output')
    processor = BatchProcessor(max_workers=workers, output_dir=output_dir)
    
    try:
        # Add jobs based on command
        if args.command == 'single':
            processor.add_job(args.input, args.output, args.config, args.job_id)
        
        elif args.command == 'csv':
            processor.add_jobs_from_csv(args.csv_file)
        
        elif args.command == 'directory':
            processor.add_directory_batch(args.input_dir, args.output_dir, args.config, args.extensions)
        
        # Process all jobs
        parallel = not args.no_parallel
        summary = processor.process_all(parallel=parallel)
        
        # Print summary
        print("\n" + "="*50)
        print("BATCH PROCESSING SUMMARY")
        print("="*50)
        print(f"Total Jobs: {summary['jobs_total']}")
        print(f"Completed: {summary['jobs_completed']}")
        print(f"Failed: {summary['jobs_failed']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration']:.1f} seconds")
        print(f"Average Job Duration: {summary['average_job_duration']:.1f} seconds")
        
        if summary['failed_jobs']:
            print(f"\nFailed Jobs:")
            for failed in summary['failed_jobs']:
                print(f"  - {failed['job_id']}: {failed['error']}")
        
        print(f"\nDetailed results saved to: {output_dir}/")
        
    except KeyboardInterrupt:
        print("\nBatch processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
