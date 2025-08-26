"""
Audio Augmentation for AutoTune ML Trainer

This module provides various audio augmentation techniques to increase
training data diversity and model robustness.
"""

import numpy as np
import librosa
import scipy.signal
from typing import List, Dict, Tuple, Union
import random
import logging

logger = logging.getLogger(__name__)


class AudioAugmentation:
    """
    Audio augmentation utilities for robust model training.
    
    Provides various augmentation techniques including pitch shifting,
    time stretching, noise addition, and more to improve model generalization.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the AudioAugmentation.
        
        Args:
            sample_rate: Audio sample rate (Hz)
        """
        self.sample_rate = sample_rate
        
    def apply_augmentations(self, 
                          audio: np.ndarray,
                          augmentation_config: Dict) -> List[np.ndarray]:
        """
        Apply multiple augmentations to audio.
        
        Args:
            audio: Input audio array
            augmentation_config: Configuration for augmentations
            
        Returns:
            List of augmented audio arrays
        """
        augmented_samples = [audio]  # Include original
        
        config = augmentation_config
        
        # Pitch shifting
        if config.get('pitch_shift', False):
            pitch_shifts = config.get('pitch_shift_range', [-2, 2])
            for shift in pitch_shifts:
                augmented = self.pitch_shift(audio, shift)
                augmented_samples.append(augmented)
                
        # Time stretching
        if config.get('time_stretch', False):
            stretch_factors = config.get('stretch_factors', [0.9, 1.1])
            for factor in stretch_factors:
                augmented = self.time_stretch(audio, factor)
                augmented_samples.append(augmented)
                
        # Noise addition
        if config.get('add_noise', False):
            noise_levels = config.get('noise_levels', [0.01, 0.02])
            for level in noise_levels:
                augmented = self.add_noise(audio, level)
                augmented_samples.append(augmented)
                
        # Reverb simulation
        if config.get('add_reverb', False):
            reverb_params = config.get('reverb_params', [{'room_size': 0.5, 'damping': 0.5}])
            for params in reverb_params:
                augmented = self.add_reverb(audio, **params)
                augmented_samples.append(augmented)
                
        # Formant shifting
        if config.get('formant_shift', False):
            formant_shifts = config.get('formant_shift_range', [-0.1, 0.1])
            for shift in formant_shifts:
                augmented = self.formant_shift(audio, shift)
                augmented_samples.append(augmented)
                
        # Dynamic range compression
        if config.get('compress', False):
            compression_ratios = config.get('compression_ratios', [2.0, 4.0])
            for ratio in compression_ratios:
                augmented = self.compress(audio, ratio)
                augmented_samples.append(augmented)
                
        return augmented_samples
        
    def pitch_shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """
        Shift pitch without changing duration.
        
        Args:
            audio: Input audio array
            semitones: Number of semitones to shift (can be fractional)
            
        Returns:
            Pitch-shifted audio array
        """
        return librosa.effects.pitch_shift(
            audio, 
            sr=self.sample_rate, 
            n_steps=semitones
        )
        
    def time_stretch(self, audio: np.ndarray, stretch_factor: float) -> np.ndarray:
        """
        Stretch or compress time without changing pitch.
        
        Args:
            audio: Input audio array
            stretch_factor: Time stretch factor (>1 = slower, <1 = faster)
            
        Returns:
            Time-stretched audio array
        """
        return librosa.effects.time_stretch(audio, rate=1.0/stretch_factor)
        
    def add_noise(self, 
                  audio: np.ndarray, 
                  noise_level: float,
                  noise_type: str = 'white') -> np.ndarray:
        """
        Add noise to audio signal.
        
        Args:
            audio: Input audio array
            noise_level: Noise level relative to signal RMS
            noise_type: Type of noise ('white', 'pink', 'brown')
            
        Returns:
            Audio with added noise
        """
        signal_rms = np.sqrt(np.mean(audio**2))
        
        if noise_type == 'white':
            noise = np.random.normal(0, 1, len(audio))
        elif noise_type == 'pink':
            noise = self._generate_pink_noise(len(audio))
        elif noise_type == 'brown':
            noise = self._generate_brown_noise(len(audio))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
            
        # Scale noise to desired level
        noise_rms = np.sqrt(np.mean(noise**2))
        if noise_rms > 0:
            noise = noise * (signal_rms * noise_level / noise_rms)
            
        return audio + noise
        
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink noise (1/f noise)."""
        white_noise = np.random.normal(0, 1, length)
        
        # Create pink noise filter (approximate)
        freqs = np.fft.fftfreq(length, 1/self.sample_rate)
        pink_filter = np.sqrt(1 / (np.abs(freqs) + 1e-10))
        pink_filter[0] = 0  # Remove DC
        
        # Apply filter
        white_fft = np.fft.fft(white_noise)
        pink_fft = white_fft * pink_filter
        pink_noise = np.real(np.fft.ifft(pink_fft))
        
        return pink_noise
        
    def _generate_brown_noise(self, length: int) -> np.ndarray:
        """Generate brown noise (1/f^2 noise)."""
        white_noise = np.random.normal(0, 1, length)
        
        # Create brown noise filter
        freqs = np.fft.fftfreq(length, 1/self.sample_rate)
        brown_filter = 1 / (np.abs(freqs) + 1e-10)
        brown_filter[0] = 0  # Remove DC
        
        # Apply filter
        white_fft = np.fft.fft(white_noise)
        brown_fft = white_fft * brown_filter
        brown_noise = np.real(np.fft.ifft(brown_fft))
        
        # Normalize
        brown_noise = brown_noise / np.std(brown_noise)
        
        return brown_noise
        
    def add_reverb(self, 
                   audio: np.ndarray,
                   room_size: float = 0.5,
                   damping: float = 0.5,
                   wet_level: float = 0.3) -> np.ndarray:
        """
        Add simple reverb effect to audio.
        
        Args:
            audio: Input audio array
            room_size: Size of simulated room (0-1)
            damping: Amount of high-frequency damping (0-1)
            wet_level: Amount of reverb to mix (0-1)
            
        Returns:
            Audio with reverb effect
        """
        # Simple reverb using multiple delayed copies
        delay_times = [0.03, 0.05, 0.08, 0.12, 0.15]  # seconds
        delay_gains = [0.5, 0.4, 0.3, 0.2, 0.1]
        
        reverb_signal = np.zeros_like(audio)
        
        for delay_time, gain in zip(delay_times, delay_gains):
            # Scale delay time by room size
            actual_delay = delay_time * room_size
            delay_samples = int(actual_delay * self.sample_rate)
            
            if delay_samples < len(audio):
                # Create delayed copy
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples]
                
                # Apply damping (low-pass filter)
                if damping > 0:
                    cutoff_freq = 8000 * (1 - damping)
                    delayed = self._low_pass_filter(delayed, cutoff_freq)
                
                reverb_signal += delayed * gain
                
        # Mix dry and wet signals
        dry_level = 1 - wet_level
        return audio * dry_level + reverb_signal * wet_level
        
    def _low_pass_filter(self, audio: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """Apply low-pass filter for reverb damping."""
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        if normalized_cutoff >= 1.0:
            return audio
            
        # Design butterworth low-pass filter
        sos = scipy.signal.butter(4, normalized_cutoff, btype='low', output='sos')
        return scipy.signal.sosfilt(sos, audio)
        
    def formant_shift(self, audio: np.ndarray, shift_factor: float) -> np.ndarray:
        """
        Shift formants to simulate different vocal tract characteristics.
        
        Args:
            audio: Input audio array
            shift_factor: Formant shift factor (-1 to 1)
            
        Returns:
            Audio with shifted formants
        """
        # Simple formant shifting using spectral envelope manipulation
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply formant shift to magnitude spectrum
        shifted_magnitude = np.zeros_like(magnitude)
        
        for frame_idx in range(magnitude.shape[1]):
            frame_mag = magnitude[:, frame_idx]
            
            # Shift frequency bins
            shift_bins = int(shift_factor * 50)  # Shift by up to 50 bins
            
            for freq_bin in range(len(frame_mag)):
                shifted_bin = freq_bin + shift_bins
                if 0 <= shifted_bin < len(frame_mag):
                    shifted_magnitude[freq_bin, frame_idx] = frame_mag[shifted_bin]
                    
        # Reconstruct audio
        shifted_stft = shifted_magnitude * np.exp(1j * phase)
        return librosa.istft(shifted_stft, hop_length=512)
        
    def compress(self, 
                 audio: np.ndarray,
                 ratio: float = 4.0,
                 threshold: float = -20.0,
                 attack: float = 0.003,
                 release: float = 0.1) -> np.ndarray:
        """
        Apply dynamic range compression.
        
        Args:
            audio: Input audio array
            ratio: Compression ratio
            threshold: Threshold in dB
            attack: Attack time in seconds
            release: Release time in seconds
            
        Returns:
            Compressed audio
        """
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
        
        # Calculate gain reduction
        gain_reduction = np.zeros_like(audio_db)
        over_threshold = audio_db > threshold
        gain_reduction[over_threshold] = (audio_db[over_threshold] - threshold) * (1 - 1/ratio)
        
        # Apply attack/release smoothing
        attack_samples = int(attack * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        smoothed_gain = np.zeros_like(gain_reduction)
        for i in range(len(gain_reduction)):
            if i == 0:
                smoothed_gain[i] = gain_reduction[i]
            else:
                if gain_reduction[i] > smoothed_gain[i-1]:
                    # Attack
                    alpha = 1 - np.exp(-1 / attack_samples) if attack_samples > 0 else 1.0
                else:
                    # Release
                    alpha = 1 - np.exp(-1 / release_samples) if release_samples > 0 else 1.0
                    
                smoothed_gain[i] = smoothed_gain[i-1] + alpha * (gain_reduction[i] - smoothed_gain[i-1])
                
        # Apply compression
        compressed_audio = audio * (10 ** (-smoothed_gain / 20))
        
        return compressed_audio
        
    def random_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random augmentation for training data variety.
        
        Args:
            audio: Input audio array
            
        Returns:
            Randomly augmented audio
        """
        augmentations = [
            lambda x: self.pitch_shift(x, random.uniform(-1, 1)),
            lambda x: self.time_stretch(x, random.uniform(0.95, 1.05)),
            lambda x: self.add_noise(x, random.uniform(0.005, 0.02)),
            lambda x: self.compress(x, random.uniform(2, 6)),
        ]
        
        # Randomly select and apply augmentation
        selected_aug = random.choice(augmentations)
        return selected_aug(audio)
