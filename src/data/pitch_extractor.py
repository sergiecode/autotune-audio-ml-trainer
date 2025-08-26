"""
Pitch Extractor for AutoTune ML Trainer

This module provides various pitch detection algorithms including CREPE,
autocorrelation, and spectral methods for training data preparation.
"""

import numpy as np
import librosa
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# Try to import CREPE
try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False
    logger.warning("CREPE not available. Install with: pip install crepe")


class PitchExtractor:
    """
    Multi-algorithm pitch extraction for AutoTune ML training.
    
    Supports CREPE, autocorrelation, YIN, and spectral pitch detection
    methods for creating training datasets.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 frame_length: int = 2048,
                 hop_length: int = 512,
                 min_frequency: float = 80.0,
                 max_frequency: float = 2000.0):
        """
        Initialize the PitchExtractor.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            frame_length: Frame length for analysis
            hop_length: Hop length between frames
            min_frequency: Minimum pitch frequency (Hz)
            max_frequency: Maximum pitch frequency (Hz)
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        
    def extract_pitch_crepe(self, 
                           audio: np.ndarray,
                           model_capacity: str = 'medium') -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch using CREPE deep learning model.
        
        Args:
            audio: Input audio array
            model_capacity: CREPE model size ('tiny', 'small', 'medium', 'large', 'full')
            
        Returns:
            Tuple of (pitch_frequencies, confidence_scores)
        """
        if not CREPE_AVAILABLE:
            raise ImportError("CREPE not available. Install with: pip install crepe")
            
        # CREPE expects specific sample rate
        if self.sample_rate != 16000:
            audio_resampled = librosa.resample(audio, 
                                             orig_sr=self.sample_rate, 
                                             target_sr=16000)
        else:
            audio_resampled = audio
            
        # Extract pitch with CREPE
        time, frequency, confidence, activation = crepe.predict(
            audio_resampled,
            sr=16000,
            model_capacity=model_capacity,
            viterbi=True,
            step_size=self.hop_length * 16000 // self.sample_rate
        )
        
        # Filter by frequency range and confidence
        valid_mask = (frequency >= self.min_frequency) & \
                    (frequency <= self.max_frequency) & \
                    (confidence > 0.5)
                    
        frequency[~valid_mask] = 0.0
        confidence[~valid_mask] = 0.0
        
        return frequency, confidence
        
    def extract_pitch_autocorr(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch using autocorrelation method.
        
        Args:
            audio: Input audio array
            
        Returns:
            Tuple of (pitch_frequencies, confidence_scores)
        """
        # Frame the audio
        frames = librosa.util.frame(audio, 
                                   frame_length=self.frame_length,
                                   hop_length=self.hop_length,
                                   axis=0)
        
        pitches = []
        confidences = []
        
        for frame in frames.T:
            pitch, confidence = self._autocorr_pitch_single_frame(frame)
            pitches.append(pitch)
            confidences.append(confidence)
            
        return np.array(pitches), np.array(confidences)
        
    def _autocorr_pitch_single_frame(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Extract pitch from single frame using autocorrelation.
        
        Args:
            frame: Single audio frame
            
        Returns:
            Tuple of (pitch_frequency, confidence)
        """
        # Apply window
        windowed = frame * np.hanning(len(frame))
        
        # Autocorrelation
        autocorr = np.correlate(windowed, windowed, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find valid lag range
        min_lag = int(self.sample_rate / self.max_frequency)
        max_lag = int(self.sample_rate / self.min_frequency)
        
        if max_lag >= len(autocorr):
            return 0.0, 0.0
            
        # Find peak in valid range
        valid_autocorr = autocorr[min_lag:max_lag]
        if len(valid_autocorr) == 0:
            return 0.0, 0.0
            
        peak_idx = np.argmax(valid_autocorr) + min_lag
        
        # Calculate pitch frequency
        if peak_idx > 0:
            pitch_freq = self.sample_rate / peak_idx
            
            # Calculate confidence as ratio of peak to RMS
            peak_value = autocorr[peak_idx]
            rms_value = np.sqrt(np.mean(autocorr[min_lag:max_lag]**2))
            confidence = peak_value / rms_value if rms_value > 0 else 0.0
            confidence = min(confidence / 10.0, 1.0)  # Normalize
            
            return pitch_freq, confidence
        else:
            return 0.0, 0.0
            
    def extract_pitch_yin(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch using YIN algorithm (via librosa).
        
        Args:
            audio: Input audio array
            
        Returns:
            Tuple of (pitch_frequencies, confidence_scores)
        """
        # Use librosa's implementation of YIN
        pitches = librosa.yin(audio,
                             fmin=self.min_frequency,
                             fmax=self.max_frequency,
                             sr=self.sample_rate,
                             frame_length=self.frame_length,
                             hop_length=self.hop_length)
        
        # Calculate confidence based on pitch stability
        confidences = self._calculate_pitch_confidence(pitches)
        
        return pitches, confidences
        
    def extract_pitch_spectral(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch using spectral peak detection.
        
        Args:
            audio: Input audio array
            
        Returns:
            Tuple of (pitch_frequencies, confidence_scores)
        """
        # Compute STFT
        stft = librosa.stft(audio, 
                           n_fft=self.frame_length,
                           hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
        
        pitches = []
        confidences = []
        
        for frame_idx in range(magnitude.shape[1]):
            frame_mag = magnitude[:, frame_idx]
            
            # Find frequency range indices
            min_bin = np.argmin(np.abs(freqs - self.min_frequency))
            max_bin = np.argmin(np.abs(freqs - self.max_frequency))
            
            # Find peak in valid range
            valid_mag = frame_mag[min_bin:max_bin]
            if len(valid_mag) == 0:
                pitches.append(0.0)
                confidences.append(0.0)
                continue
                
            peak_idx = np.argmax(valid_mag) + min_bin
            peak_freq = freqs[peak_idx]
            
            # Calculate confidence
            peak_magnitude = frame_mag[peak_idx]
            mean_magnitude = np.mean(valid_mag)
            confidence = (peak_magnitude / mean_magnitude - 1.0) if mean_magnitude > 0 else 0.0
            confidence = min(confidence / 10.0, 1.0)  # Normalize
            
            pitches.append(peak_freq)
            confidences.append(confidence)
            
        return np.array(pitches), np.array(confidences)
        
    def _calculate_pitch_confidence(self, pitches: np.ndarray) -> np.ndarray:
        """
        Calculate confidence scores based on pitch stability.
        
        Args:
            pitches: Array of pitch values
            
        Returns:
            Array of confidence scores
        """
        confidences = np.zeros_like(pitches)
        
        for i in range(len(pitches)):
            if pitches[i] == 0:
                confidences[i] = 0.0
                continue
                
            # Look at neighboring frames
            start_idx = max(0, i - 2)
            end_idx = min(len(pitches), i + 3)
            neighborhood = pitches[start_idx:end_idx]
            
            # Remove zeros
            valid_neighbors = neighborhood[neighborhood > 0]
            
            if len(valid_neighbors) < 2:
                confidences[i] = 0.5
                continue
                
            # Calculate stability (inverse of relative standard deviation)
            mean_pitch = np.mean(valid_neighbors)
            std_pitch = np.std(valid_neighbors)
            
            if mean_pitch > 0:
                relative_std = std_pitch / mean_pitch
                confidence = np.exp(-relative_std * 5)  # Exponential decay
                confidences[i] = confidence
            else:
                confidences[i] = 0.0
                
        return confidences
        
    def extract_multi_algorithm(self, 
                               audio: np.ndarray,
                               algorithms: List[str] = ['crepe', 'yin', 'autocorr']) -> Dict:
        """
        Extract pitch using multiple algorithms and combine results.
        
        Args:
            audio: Input audio array
            algorithms: List of algorithms to use
            
        Returns:
            Dictionary with results from each algorithm plus combined result
        """
        results = {}
        
        # Extract with each algorithm
        if 'crepe' in algorithms and CREPE_AVAILABLE:
            try:
                pitches, confidences = self.extract_pitch_crepe(audio)
                results['crepe'] = {'pitches': pitches, 'confidences': confidences}
            except Exception as e:
                logger.warning(f"CREPE extraction failed: {e}")
                
        if 'yin' in algorithms:
            try:
                pitches, confidences = self.extract_pitch_yin(audio)
                results['yin'] = {'pitches': pitches, 'confidences': confidences}
            except Exception as e:
                logger.warning(f"YIN extraction failed: {e}")
                
        if 'autocorr' in algorithms:
            try:
                pitches, confidences = self.extract_pitch_autocorr(audio)
                results['autocorr'] = {'pitches': pitches, 'confidences': confidences}
            except Exception as e:
                logger.warning(f"Autocorrelation extraction failed: {e}")
                
        if 'spectral' in algorithms:
            try:
                pitches, confidences = self.extract_pitch_spectral(audio)
                results['spectral'] = {'pitches': pitches, 'confidences': confidences}
            except Exception as e:
                logger.warning(f"Spectral extraction failed: {e}")
                
        # Combine results using weighted average
        if len(results) > 1:
            combined_pitches, combined_confidences = self._combine_pitch_results(results)
            results['combined'] = {
                'pitches': combined_pitches, 
                'confidences': combined_confidences
            }
            
        return results
        
    def _combine_pitch_results(self, results: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine pitch results from multiple algorithms.
        
        Args:
            results: Dictionary of results from different algorithms
            
        Returns:
            Tuple of (combined_pitches, combined_confidences)
        """
        # Get all pitch and confidence arrays
        all_pitches = []
        all_confidences = []
        
        for alg_name, alg_results in results.items():
            if alg_name != 'combined':
                all_pitches.append(alg_results['pitches'])
                all_confidences.append(alg_results['confidences'])
                
        if not all_pitches:
            return np.array([]), np.array([])
            
        # Find minimum length
        min_length = min(len(p) for p in all_pitches)
        
        # Truncate all arrays to minimum length
        all_pitches = [p[:min_length] for p in all_pitches]
        all_confidences = [c[:min_length] for c in all_confidences]
        
        # Convert to numpy arrays
        pitches_array = np.array(all_pitches)
        confidences_array = np.array(all_confidences)
        
        # Weighted average
        combined_pitches = np.zeros(min_length)
        combined_confidences = np.zeros(min_length)
        
        for i in range(min_length):
            frame_pitches = pitches_array[:, i]
            frame_confidences = confidences_array[:, i]
            
            # Only consider non-zero pitches
            valid_mask = frame_pitches > 0
            
            if np.any(valid_mask):
                valid_pitches = frame_pitches[valid_mask]
                valid_confidences = frame_confidences[valid_mask]
                
                # Weighted average
                total_confidence = np.sum(valid_confidences)
                if total_confidence > 0:
                    combined_pitches[i] = np.sum(valid_pitches * valid_confidences) / total_confidence
                    combined_confidences[i] = np.mean(valid_confidences)
                else:
                    combined_pitches[i] = np.mean(valid_pitches)
                    combined_confidences[i] = 0.5
            else:
                combined_pitches[i] = 0.0
                combined_confidences[i] = 0.0
                
        return combined_pitches, combined_confidences
