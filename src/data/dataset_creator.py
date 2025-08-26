"""
Dataset Creator for AutoTune ML Trainer

This module handles the creation of training datasets from raw audio files,
including segmentation, labeling, and preparation for neural network training.
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetCreator:
    """
    Creates training datasets from raw audio files for AutoTune ML training.
    
    This class processes audio files, extracts features, and creates
    structured datasets suitable for training pitch correction and
    timing adjustment models.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 segment_length: float = 4.0,
                 overlap: float = 0.5,
                 min_pitch: float = 80.0,
                 max_pitch: float = 2000.0):
        """
        Initialize the DatasetCreator.
        
        Args:
            sample_rate: Target sample rate for audio processing (Hz)
            segment_length: Length of audio segments in seconds
            overlap: Overlap ratio between segments (0.0-1.0)
            min_pitch: Minimum pitch frequency to consider (Hz)
            max_pitch: Maximum pitch frequency to consider (Hz)
        """
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.overlap = overlap
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        
        # Calculate segment parameters
        self.segment_samples = int(segment_length * sample_rate)
        self.hop_samples = int(self.segment_samples * (1 - overlap))
        
        logger.info(f"DatasetCreator initialized:")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Segment length: {segment_length} seconds ({self.segment_samples} samples)")
        logger.info(f"  Overlap: {overlap * 100:.1f}%")
        
    def create_dataset(self, 
                      input_dir: Union[str, Path],
                      output_dir: Union[str, Path],
                      dataset_name: str = "autotune_dataset",
                      file_extensions: List[str] = ['.wav', '.flac', '.mp3', '.m4a']) -> Dict:
        """
        Create a complete dataset from audio files in input directory.
        
        Args:
            input_dir: Directory containing raw audio files
            output_dir: Directory to save processed dataset
            dataset_name: Name of the dataset
            file_extensions: Supported audio file extensions
            
        Returns:
            Dictionary with dataset statistics and metadata
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(list(input_path.rglob(f'*{ext}')))
            
        if not audio_files:
            raise ValueError(f"No audio files found in {input_dir} with extensions {file_extensions}")
            
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Process files and create dataset
        dataset_info = {
            'name': dataset_name,
            'total_files': len(audio_files),
            'processed_files': 0,
            'total_segments': 0,
            'failed_files': [],
            'statistics': {}
        }
        
        all_segments = []
        all_metadata = []
        
        for file_path in tqdm(audio_files, desc="Processing audio files"):
            try:
                segments, metadata = self._process_audio_file(file_path)
                all_segments.extend(segments)
                all_metadata.extend(metadata)
                dataset_info['processed_files'] += 1
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {str(e)}")
                dataset_info['failed_files'].append(str(file_path))
                
        dataset_info['total_segments'] = len(all_segments)
        
        # Save processed data
        self._save_dataset(all_segments, all_metadata, output_path, dataset_name)
        
        # Calculate and save statistics
        dataset_info['statistics'] = self._calculate_statistics(all_metadata)
        self._save_dataset_info(dataset_info, output_path, dataset_name)
        
        logger.info(f"Dataset creation completed:")
        logger.info(f"  Processed files: {dataset_info['processed_files']}/{dataset_info['total_files']}")
        logger.info(f"  Total segments: {dataset_info['total_segments']}")
        logger.info(f"  Output directory: {output_path}")
        
        return dataset_info
        
    def _process_audio_file(self, file_path: Path) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Process a single audio file into segments with metadata.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_segments, metadata_list)
        """
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            if len(audio) < self.segment_samples:
                logger.warning(f"Audio file {file_path} too short, skipping")
                return [], []
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return [], []
            
        # Create segments
        segments = []
        metadata = []
        
        start_idx = 0
        segment_id = 0
        
        while start_idx + self.segment_samples <= len(audio):
            end_idx = start_idx + self.segment_samples
            segment = audio[start_idx:end_idx]
            
            # Extract basic metadata
            segment_metadata = {
                'file_path': str(file_path),
                'segment_id': segment_id,
                'start_time': start_idx / self.sample_rate,
                'end_time': end_idx / self.sample_rate,
                'duration': self.segment_length,
                'sample_rate': self.sample_rate,
                'rms_energy': np.sqrt(np.mean(segment**2)),
                'max_amplitude': np.max(np.abs(segment)),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(segment)),
            }
            
            # Add pitch information (basic)
            try:
                pitch_data = self._extract_basic_pitch_info(segment)
                segment_metadata.update(pitch_data)
            except Exception as e:
                logger.warning(f"Pitch extraction failed for segment {segment_id}: {str(e)}")
                segment_metadata.update({
                    'has_pitch': False,
                    'mean_pitch': 0.0,
                    'pitch_confidence': 0.0
                })
            
            segments.append(segment)
            metadata.append(segment_metadata)
            
            start_idx += self.hop_samples
            segment_id += 1
            
        return segments, metadata
        
    def _extract_basic_pitch_info(self, audio_segment: np.ndarray) -> Dict:
        """
        Extract basic pitch information from an audio segment.
        
        Args:
            audio_segment: Audio data as numpy array
            
        Returns:
            Dictionary with pitch information
        """
        # Use librosa's pitch detection (basic autocorrelation)
        pitches, magnitudes = librosa.piptrack(
            y=audio_segment,
            sr=self.sample_rate,
            fmin=self.min_pitch,
            fmax=self.max_pitch,
            threshold=0.1
        )
        
        # Extract fundamental frequency estimates
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
                
        if pitch_values:
            mean_pitch = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            pitch_confidence = len(pitch_values) / pitches.shape[1]  # Proportion of frames with pitch
            has_pitch = True
        else:
            mean_pitch = 0.0
            pitch_std = 0.0
            pitch_confidence = 0.0
            has_pitch = False
            
        return {
            'has_pitch': has_pitch,
            'mean_pitch': mean_pitch,
            'pitch_std': pitch_std,
            'pitch_confidence': pitch_confidence,
            'num_pitch_frames': len(pitch_values)
        }
        
    def _save_dataset(self, 
                     segments: List[np.ndarray], 
                     metadata: List[Dict],
                     output_path: Path,
                     dataset_name: str):
        """
        Save processed segments and metadata to disk.
        
        Args:
            segments: List of audio segments
            metadata: List of metadata dictionaries
            output_path: Output directory path
            dataset_name: Name of the dataset
        """
        # Create subdirectories
        audio_dir = output_path / f"{dataset_name}_audio"
        audio_dir.mkdir(exist_ok=True)
        
        # Save audio segments
        logger.info("Saving audio segments...")
        for i, segment in enumerate(tqdm(segments, desc="Saving segments")):
            filename = f"segment_{i:06d}.wav"
            filepath = audio_dir / filename
            sf.write(filepath, segment, self.sample_rate)
            
            # Update metadata with file path
            metadata[i]['segment_file'] = str(filepath)
            
        # Save metadata as CSV
        metadata_df = pd.DataFrame(metadata)
        metadata_path = output_path / f"{dataset_name}_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        logger.info(f"Saved {len(segments)} segments to {audio_dir}")
        logger.info(f"Saved metadata to {metadata_path}")
        
    def _calculate_statistics(self, metadata: List[Dict]) -> Dict:
        """
        Calculate dataset statistics from metadata.
        
        Args:
            metadata: List of metadata dictionaries
            
        Returns:
            Dictionary with dataset statistics
        """
        df = pd.DataFrame(metadata)
        
        statistics = {
            'total_duration': df['duration'].sum(),
            'mean_rms_energy': df['rms_energy'].mean(),
            'std_rms_energy': df['rms_energy'].std(),
            'mean_max_amplitude': df['max_amplitude'].mean(),
            'std_max_amplitude': df['max_amplitude'].std(),
            'mean_zero_crossing_rate': df['zero_crossing_rate'].mean(),
            'std_zero_crossing_rate': df['zero_crossing_rate'].std(),
            'segments_with_pitch': df['has_pitch'].sum(),
            'pitch_detection_rate': df['has_pitch'].mean(),
        }
        
        # Pitch statistics (only for segments with pitch)
        pitched_segments = df[df['has_pitch']]
        if len(pitched_segments) > 0:
            statistics.update({
                'mean_pitch': pitched_segments['mean_pitch'].mean(),
                'std_pitch': pitched_segments['mean_pitch'].std(),
                'min_pitch': pitched_segments['mean_pitch'].min(),
                'max_pitch': pitched_segments['mean_pitch'].max(),
                'mean_pitch_confidence': pitched_segments['pitch_confidence'].mean(),
            })
        
        return statistics
        
    def _save_dataset_info(self, 
                          dataset_info: Dict,
                          output_path: Path,
                          dataset_name: str):
        """
        Save dataset information and statistics.
        
        Args:
            dataset_info: Dataset information dictionary
            output_path: Output directory path
            dataset_name: Name of the dataset
        """
        import json
        
        info_path = output_path / f"{dataset_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
            
        logger.info(f"Saved dataset info to {info_path}")


def create_dataset_from_config(config_path: str) -> Dict:
    """
    Create dataset from configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dataset information dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    creator = DatasetCreator(
        sample_rate=config.get('sample_rate', 44100),
        segment_length=config.get('segment_length', 4.0),
        overlap=config.get('overlap', 0.5),
        min_pitch=config.get('min_pitch', 80.0),
        max_pitch=config.get('max_pitch', 2000.0)
    )
    
    return creator.create_dataset(
        input_dir=config['input_dir'],
        output_dir=config['output_dir'],
        dataset_name=config.get('dataset_name', 'autotune_dataset'),
        file_extensions=config.get('file_extensions', ['.wav', '.flac', '.mp3'])
    )


if __name__ == "__main__":
    # Example usage
    creator = DatasetCreator()
    
    # Create dataset from command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Create AutoTune ML training dataset")
    parser.add_argument("--input", required=True, help="Input directory with audio files")
    parser.add_argument("--output", required=True, help="Output directory for dataset")
    parser.add_argument("--name", default="autotune_dataset", help="Dataset name")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Sample rate")
    parser.add_argument("--segment_length", type=float, default=4.0, help="Segment length in seconds")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio")
    
    args = parser.parse_args()
    
    creator = DatasetCreator(
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        overlap=args.overlap
    )
    
    dataset_info = creator.create_dataset(
        input_dir=args.input,
        output_dir=args.output,
        dataset_name=args.name
    )
    
    print(f"Dataset created successfully!")
    print(f"Total segments: {dataset_info['total_segments']}")
    print(f"Total duration: {dataset_info['statistics']['total_duration']:.2f} seconds")
