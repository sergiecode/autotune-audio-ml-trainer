"""
Dataset Creation Script for AutoTune ML Trainer

This script creates training datasets from raw audio files with various
preprocessing and augmentation options.
"""

import argparse
import sys
import json
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset_creator import DatasetCreator
from data.audio_preprocessor import AudioPreprocessor
from data.augmentation import AudioAugmentation

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create AutoTune ML training dataset from audio files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing audio files"
    )
    
    parser.add_argument(
        "--output", "-o", 
        required=True,
        help="Output directory for processed dataset"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--name", 
        default="autotune_dataset",
        help="Name of the dataset"
    )
    
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Target sample rate (Hz)"
    )
    
    parser.add_argument(
        "--segment_length",
        type=float,
        default=4.0,
        help="Length of audio segments in seconds"
    )
    
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap ratio between segments (0.0-1.0)"
    )
    
    parser.add_argument(
        "--min_pitch",
        type=float,
        default=80.0,
        help="Minimum pitch frequency to consider (Hz)"
    )
    
    parser.add_argument(
        "--max_pitch",
        type=float,
        default=2000.0,
        help="Maximum pitch frequency to consider (Hz)"
    )
    
    # File filtering
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".wav", ".flac", ".mp3", ".m4a"],
        help="Audio file extensions to process"
    )
    
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1.0,
        help="Minimum audio file duration in seconds"
    )
    
    # Processing options
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize audio amplitude"
    )
    
    parser.add_argument(
        "--apply_preprocessing",
        action="store_true",
        default=True,
        help="Apply audio preprocessing (filtering, DC removal)"
    )
    
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help="Apply audio augmentation for data diversity"
    )
    
    # Augmentation options
    parser.add_argument(
        "--pitch_shift_range",
        nargs=2,
        type=float,
        default=[-1.0, 1.0],
        help="Pitch shift range in semitones [min, max]"
    )
    
    parser.add_argument(
        "--time_stretch_range", 
        nargs=2,
        type=float,
        default=[0.95, 1.05],
        help="Time stretch factor range [min, max]"
    )
    
    parser.add_argument(
        "--noise_levels",
        nargs="+",
        type=float,
        default=[0.01, 0.02],
        help="Noise levels for augmentation"
    )
    
    # Output options
    parser.add_argument(
        "--save_config",
        action="store_true",
        default=True,
        help="Save dataset configuration"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    errors = []
    
    # Check input directory
    input_path = Path(args.input)
    if not input_path.exists():
        errors.append(f"Input directory does not exist: {args.input}")
    elif not input_path.is_dir():
        errors.append(f"Input path is not a directory: {args.input}")
        
    # Check overlap range
    if not 0.0 <= args.overlap <= 1.0:
        errors.append(f"Overlap must be between 0.0 and 1.0, got {args.overlap}")
        
    # Check segment length
    if args.segment_length <= 0:
        errors.append(f"Segment length must be positive, got {args.segment_length}")
        
    # Check pitch range
    if args.min_pitch >= args.max_pitch:
        errors.append(f"Min pitch ({args.min_pitch}) must be less than max pitch ({args.max_pitch})")
        
    # Check sample rate
    if args.sample_rate <= 0:
        errors.append(f"Sample rate must be positive, got {args.sample_rate}")
        
    if errors:
        logger.error("Argument validation errors:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
        
    return True


def create_augmentation_config(args):
    """Create augmentation configuration from arguments."""
    return {
        'pitch_shift': True,
        'pitch_shift_range': list(args.pitch_shift_range),
        'time_stretch': True,
        'stretch_factors': list(args.time_stretch_range),
        'add_noise': True,
        'noise_levels': args.noise_levels,
        'add_reverb': True,
        'reverb_params': [
            {'room_size': 0.3, 'damping': 0.5, 'wet_level': 0.2},
            {'room_size': 0.7, 'damping': 0.3, 'wet_level': 0.3}
        ]
    }


def save_dataset_config(args, output_path):
    """Save dataset creation configuration."""
    config = {
        'input_directory': args.input,
        'output_directory': args.output,
        'dataset_name': args.name,
        'creation_parameters': {
            'sample_rate': args.sample_rate,
            'segment_length': args.segment_length,
            'overlap': args.overlap,
            'min_pitch': args.min_pitch,
            'max_pitch': args.max_pitch,
            'extensions': args.extensions,
            'min_duration': args.min_duration,
            'normalize': args.normalize,
            'apply_preprocessing': args.apply_preprocessing,
            'augment': args.augment
        }
    }
    
    if args.augment:
        config['augmentation'] = create_augmentation_config(args)
        
    config_path = output_path / f"{args.name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    logger.info(f"Saved dataset configuration to {config_path}")


def count_audio_files(input_path, extensions):
    """Count audio files in input directory."""
    count = 0
    for ext in extensions:
        count += len(list(input_path.rglob(f'*{ext}')))
    return count


def main():
    """Main dataset creation function."""
    print("AutoTune ML Trainer - Dataset Creator")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Validate arguments
    if not validate_arguments(args):
        return 1
        
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Count input files
    input_path = Path(args.input)
    file_count = count_audio_files(input_path, args.extensions)
    
    if file_count == 0:
        logger.error(f"No audio files found in {args.input} with extensions {args.extensions}")
        return 1
        
    logger.info(f"Found {file_count} audio files to process")
    
    # Create dataset creator
    creator = DatasetCreator(
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        overlap=args.overlap,
        min_pitch=args.min_pitch,
        max_pitch=args.max_pitch
    )
    
    # Create dataset
    try:
        logger.info("Starting dataset creation...")
        dataset_info = creator.create_dataset(
            input_dir=args.input,
            output_dir=args.output,
            dataset_name=args.name,
            file_extensions=args.extensions
        )
        
        # Save configuration
        if args.save_config:
            save_dataset_config(args, output_path)
            
        # Print summary
        print(f"\nDataset Creation Complete!")
        print(f"Dataset name: {args.name}")
        print(f"Total segments: {dataset_info['total_segments']:,}")
        print(f"Processed files: {dataset_info['processed_files']}/{dataset_info['total_files']}")
        print(f"Total duration: {dataset_info['statistics']['total_duration']:.1f} seconds")
        print(f"Pitch detection rate: {dataset_info['statistics']['pitch_detection_rate']:.1%}")
        print(f"Output directory: {args.output}")
        
        if dataset_info['failed_files']:
            print(f"\nWarning: {len(dataset_info['failed_files'])} files failed to process")
            if args.verbose:
                print("Failed files:")
                for failed_file in dataset_info['failed_files']:
                    print(f"  - {failed_file}")
                    
        print(f"\nNext steps:")
        print(f"1. Review the dataset with: jupyter lab notebooks/01_data_exploration.ipynb")
        print(f"2. Train a model with: python scripts/train_pitch_model.py --dataset {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
