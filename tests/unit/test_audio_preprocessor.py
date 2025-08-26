"""
Unit tests for audio preprocessing functionality
Created by Sergie Code
"""
import pytest
import numpy as np
import torch
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    from src.data.audio_preprocessor import AudioPreprocessor
except ImportError:
    # Fallback import
    import sys
    sys.path.append(str(ROOT_DIR / "src" / "data"))
    from audio_preprocessor import AudioPreprocessor

class TestAudioPreprocessor:
    """Test suite for AudioPreprocessor class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.preprocessor = AudioPreprocessor(
            sample_rate=44100,
            target_lufs=-23.0,
            normalize=True
        )
    
    def test_init(self):
        """Test AudioPreprocessor initialization"""
        assert self.preprocessor.sample_rate == 44100
        assert self.preprocessor.target_lufs == -23.0
        assert self.preprocessor.normalize == True
        assert self.preprocessor.hop_length == 256
        assert self.preprocessor.n_mels == 128
    
    def test_normalize_audio(self, test_audio_mono):
        """Test audio normalization"""
        # Test with normal audio
        normalized = self.preprocessor.normalize_audio(test_audio_mono)
        assert np.max(np.abs(normalized)) <= 1.0
        assert normalized.dtype == np.float32
        
        # Test with audio that needs scaling
        loud_audio = test_audio_mono * 10.0
        normalized_loud = self.preprocessor.normalize_audio(loud_audio)
        assert np.max(np.abs(normalized_loud)) <= 1.0
        
        # Test with silent audio
        silent_audio = np.zeros(1000, dtype=np.float32)
        normalized_silent = self.preprocessor.normalize_audio(silent_audio)
        assert np.allclose(normalized_silent, silent_audio)
    
    def test_extract_features(self, test_audio_mono):
        """Test feature extraction"""
        features = self.preprocessor.extract_features(test_audio_mono)
        
        # Check feature dict structure
        assert isinstance(features, dict)
        expected_keys = ['mfcc', 'spectral_centroid', 'spectral_rolloff', 
                        'zero_crossing_rate', 'mel_spectrogram']
        for key in expected_keys:
            assert key in features
        
        # Check MFCC shape
        mfcc = features['mfcc']
        assert mfcc.ndim == 2
        assert mfcc.shape[0] == 13  # Default n_mfcc
        
        # Check mel spectrogram shape
        mel_spec = features['mel_spectrogram']
        assert mel_spec.ndim == 2
        assert mel_spec.shape[0] == self.preprocessor.n_mels
    
    def test_compute_stft(self, test_audio_mono):
        """Test STFT computation"""
        stft = self.preprocessor.compute_stft(test_audio_mono)
        
        # Check output shape
        expected_freq_bins = self.preprocessor.n_fft // 2 + 1
        assert stft.shape[0] == expected_freq_bins
        
        # Check data type
        assert stft.dtype == np.complex64
        
        # Check that we get reasonable magnitude values
        magnitude = np.abs(stft)
        assert np.max(magnitude) > 0
    
    def test_compute_mel_spectrogram(self, test_audio_mono):
        """Test mel spectrogram computation"""
        mel_spec = self.preprocessor.compute_mel_spectrogram(test_audio_mono)
        
        # Check shape
        assert mel_spec.shape[0] == self.preprocessor.n_mels
        assert mel_spec.ndim == 2
        
        # Check that values are reasonable (log scale)
        assert np.all(mel_spec >= -80)  # Typical minimum dB value
        assert np.max(mel_spec) > -80
    
    def test_pre_emphasis(self, test_audio_mono):
        """Test pre-emphasis filter"""
        emphasized = self.preprocessor.pre_emphasis(test_audio_mono)
        
        # Check shape preservation
        assert emphasized.shape == test_audio_mono.shape
        
        # Check that pre-emphasis actually changed the signal
        assert not np.allclose(emphasized, test_audio_mono)
        
        # Test with different coefficients
        emphasized_strong = self.preprocessor.pre_emphasis(test_audio_mono, coeff=0.98)
        emphasized_weak = self.preprocessor.pre_emphasis(test_audio_mono, coeff=0.95)
        
        # Stronger pre-emphasis should create more difference
        diff_strong = np.mean(np.abs(emphasized_strong - test_audio_mono))
        diff_weak = np.mean(np.abs(emphasized_weak - test_audio_mono))
        assert diff_strong > diff_weak
    
    def test_apply_window(self, test_audio_mono):
        """Test windowing function"""
        windowed = self.preprocessor.apply_window(test_audio_mono, window_type='hann')
        
        # Check shape preservation
        assert windowed.shape == test_audio_mono.shape
        
        # Check that windowing reduced edge values
        assert np.abs(windowed[0]) <= np.abs(test_audio_mono[0])
        assert np.abs(windowed[-1]) <= np.abs(test_audio_mono[-1])
        
        # Test different window types
        for window_type in ['hamming', 'blackman', 'bartlett']:
            windowed_test = self.preprocessor.apply_window(test_audio_mono, window_type=window_type)
            assert windowed_test.shape == test_audio_mono.shape
    
    def test_process_stereo_to_mono(self, test_audio_stereo):
        """Test stereo to mono conversion"""
        mono = self.preprocessor.process_stereo_to_mono(test_audio_stereo)
        
        # Check output shape
        assert mono.ndim == 1
        assert mono.shape[0] == test_audio_stereo.shape[1]
        
        # Check that values are reasonable average
        expected_max = np.max(np.abs(test_audio_stereo))
        assert np.max(np.abs(mono)) <= expected_max
    
    def test_resample_audio(self, test_audio_mono, sample_rate):
        """Test audio resampling"""
        target_sr = 22050
        resampled = self.preprocessor.resample_audio(test_audio_mono, sample_rate, target_sr)
        
        # Check that length changed appropriately
        expected_length = int(len(test_audio_mono) * target_sr / sample_rate)
        assert abs(len(resampled) - expected_length) <= 1  # Allow for rounding
        
        # Test upsampling
        upsampled = self.preprocessor.resample_audio(test_audio_mono, sample_rate, 88200)
        assert len(upsampled) > len(test_audio_mono)
    
    def test_segment_audio(self, test_audio_mono, sample_rate):
        """Test audio segmentation"""
        segment_length = 1.0  # seconds
        hop_length = 0.5      # seconds
        
        segments = self.preprocessor.segment_audio(
            test_audio_mono, segment_length, hop_length, sample_rate
        )
        
        # Check that we get segments
        assert len(segments) > 0
        
        # Check segment length
        expected_samples = int(segment_length * sample_rate)
        for segment in segments:
            assert len(segment) == expected_samples
    
    def test_remove_dc_offset(self, test_audio_mono):
        """Test DC offset removal"""
        # Add DC offset
        offset_audio = test_audio_mono + 0.1
        assert np.mean(offset_audio) > 0.05  # Verify offset exists
        
        # Remove offset
        clean_audio = self.preprocessor.remove_dc_offset(offset_audio)
        
        # Check that mean is near zero
        assert abs(np.mean(clean_audio)) < 1e-10
        
        # Check that AC component is preserved
        ac_component_original = test_audio_mono - np.mean(test_audio_mono)
        ac_component_cleaned = clean_audio
        np.testing.assert_allclose(ac_component_original, ac_component_cleaned, rtol=1e-10)
    
    def test_process_pipeline(self, test_audio_mono):
        """Test complete preprocessing pipeline"""
        processed = self.preprocessor.process_pipeline(test_audio_mono)
        
        # Check that output is a dictionary
        assert isinstance(processed, dict)
        
        # Check for required keys
        required_keys = ['audio', 'features', 'metadata']
        for key in required_keys:
            assert key in processed
        
        # Check processed audio properties
        processed_audio = processed['audio']
        assert processed_audio.dtype == np.float32
        assert np.max(np.abs(processed_audio)) <= 1.0
        
        # Check features
        features = processed['features']
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check metadata
        metadata = processed['metadata']
        assert isinstance(metadata, dict)
        assert 'sample_rate' in metadata
        assert 'duration' in metadata
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Test with empty audio
        with pytest.raises((ValueError, RuntimeError)):
            self.preprocessor.extract_features(np.array([]))
        
        # Test with invalid window type
        test_signal = np.random.randn(1000).astype(np.float32)
        with pytest.raises(ValueError):
            self.preprocessor.apply_window(test_signal, window_type='invalid_window')
        
        # Test with invalid segment parameters
        with pytest.raises(ValueError):
            self.preprocessor.segment_audio(test_signal, segment_length=-1.0, hop_length=0.5, sample_rate=44100)
    
    def test_batch_processing(self, test_audio_mono):
        """Test batch processing capabilities"""
        # Create multiple audio segments
        segments = [test_audio_mono[:22050], test_audio_mono[22050:44100], test_audio_mono[:44100]]
        
        # Process batch
        results = self.preprocessor.process_batch(segments)
        
        # Check results
        assert len(results) == len(segments)
        for result in results:
            assert isinstance(result, dict)
            assert 'audio' in result
            assert 'features' in result
    
    @pytest.mark.parametrize("n_fft", [512, 1024, 2048])
    def test_different_fft_sizes(self, test_audio_mono, n_fft):
        """Test preprocessing with different FFT sizes"""
        preprocessor = AudioPreprocessor(
            sample_rate=44100,
            n_fft=n_fft,
            hop_length=n_fft//4,
            n_mels=128
        )
        
        stft = preprocessor.compute_stft(test_audio_mono)
        expected_freq_bins = n_fft // 2 + 1
        assert stft.shape[0] == expected_freq_bins
    
    def test_memory_efficiency(self, test_audio_mono):
        """Test that preprocessing doesn't use excessive memory"""
        # Create a longer audio signal
        long_audio = np.tile(test_audio_mono, 10)  # 20 seconds
        
        # Process in chunks to test memory efficiency
        chunk_size = len(test_audio_mono)
        features_list = []
        
        for i in range(0, len(long_audio), chunk_size):
            chunk = long_audio[i:i+chunk_size]
            if len(chunk) == chunk_size:  # Only process full chunks
                features = self.preprocessor.extract_features(chunk)
                features_list.append(features)
        
        assert len(features_list) == 10
        
        # Verify that all features have consistent shapes
        mfcc_shapes = [f['mfcc'].shape for f in features_list]
        assert all(shape[0] == mfcc_shapes[0][0] for shape in mfcc_shapes)  # Same number of MFCC coefficients
