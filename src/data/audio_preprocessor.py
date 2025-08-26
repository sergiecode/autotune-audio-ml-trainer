"""
Audio Preprocessor for AutoTune ML Trainer

This module provides audio preprocessing utilities including normalization,
filtering, and format conversion for neural network training.
"""

import numpy as np
import librosa
import scipy.signal
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Audio preprocessing utilities for AutoTune ML training.
    
    Handles normalization, filtering, resampling, and other audio
    preprocessing tasks required for training neural networks.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 target_lufs: float = -23.0,
                 normalize: bool = True):
        """
        Initialize the AudioPreprocessor.
        
        Args:
            sample_rate: Target sample rate (Hz)
            target_lufs: Target loudness in LUFS
            normalize: Whether to normalize audio amplitude
        """
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.normalize = normalize
        
    def preprocess(self, 
                  audio: np.ndarray,
                  sr: Optional[int] = None) -> np.ndarray:
        """
        Apply full preprocessing pipeline to audio.
        
        Args:
            audio: Input audio array
            sr: Original sample rate (if different from target)
            
        Returns:
            Preprocessed audio array
        """
        # Resample if necessary
        if sr is not None and sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
        # Normalize
        if self.normalize:
            audio = self.normalize_audio(audio)
            
        # Remove DC offset
        audio = self.remove_dc_offset(audio)
        
        # Apply gentle high-pass filter
        audio = self.high_pass_filter(audio, cutoff=20.0)
        
        return audio
        
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target amplitude.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            # Normalize to -3dB peak to avoid clipping
            target_peak = 0.7
            audio = audio * (target_peak / max_amp)
        return audio
        
    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove DC offset from audio.
        
        Args:
            audio: Input audio array
            
        Returns:
            Audio with DC offset removed
        """
        return audio - np.mean(audio)
        
    def high_pass_filter(self, 
                        audio: np.ndarray, 
                        cutoff: float = 20.0,
                        order: int = 4) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise.
        
        Args:
            audio: Input audio array
            cutoff: Cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Filtered audio array
        """
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        # Design Butterworth high-pass filter
        sos = scipy.signal.butter(order, normalized_cutoff, 
                                 btype='high', output='sos')
        
        # Apply filter
        filtered_audio = scipy.signal.sosfilt(sos, audio)
        
        return filtered_audio
        
    def apply_window(self, 
                    audio: np.ndarray,
                    window_type: str = 'hann') -> np.ndarray:
        """
        Apply windowing function to audio segment.
        
        Args:
            audio: Input audio array
            window_type: Type of window ('hann', 'hamming', 'blackman')
            
        Returns:
            Windowed audio array
        """
        if window_type == 'hann':
            window = np.hanning(len(audio))
        elif window_type == 'hamming':
            window = np.hamming(len(audio))
        elif window_type == 'blackman':
            window = np.blackman(len(audio))
        else:
            raise ValueError(f"Unsupported window type: {window_type}")
            
        return audio * window
        
    def extract_features(self, audio: np.ndarray) -> dict:
        """
        Extract audio features for analysis.
        
        Args:
            audio: Input audio array
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic statistics
        features['rms'] = np.sqrt(np.mean(audio**2))
        features['peak'] = np.max(np.abs(audio))
        features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] > 0 else 0
        
        # Spectral features
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(
            S=magnitude, sr=self.sample_rate))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(
            S=magnitude, sr=self.sample_rate))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(
            S=magnitude, sr=self.sample_rate))
            
        # Zero crossing rate
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}'] = np.mean(mfccs[i])
            
        return features
