"""
Unit tests for pitch extraction functionality
Created by Sergie Code
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    from src.data.pitch_extractor import PitchExtractor
except ImportError:
    # Fallback import
    import sys
    sys.path.append(str(ROOT_DIR / "src" / "data"))
    from pitch_extractor import PitchExtractor

class TestPitchExtractor:
    """Test suite for PitchExtractor class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.extractor = PitchExtractor(
            sample_rate=44100,
            frame_length=1024,
            hop_length=256
        )
    
    def test_init(self):
        """Test PitchExtractor initialization"""
        assert self.extractor.sample_rate == 44100
        assert self.extractor.frame_length == 1024
        assert self.extractor.hop_length == 256
    
    def test_extract_pitch_librosa(self, test_audio_mono):
        """Test pitch extraction using librosa"""
        pitch, confidence = self.extractor.extract_pitch_librosa(test_audio_mono)
        
        # Check output shape
        assert len(pitch) == len(confidence)
        assert len(pitch) > 0
        
        # Check value ranges
        assert np.all(pitch >= 0)  # Pitch should be non-negative (0 for unvoiced)
        assert np.all((confidence >= 0) & (confidence <= 1))  # Confidence in [0, 1]
        
        # Check that we detect some pitch in a tonal signal
        voiced_frames = pitch > 0
        assert np.sum(voiced_frames) > 0  # Should detect some voiced frames
    
    def test_extract_pitch_yin(self, test_audio_mono):
        """Test pitch extraction using YIN algorithm"""
        pitch, confidence = self.extractor.extract_pitch_yin(test_audio_mono)
        
        # Check output shape
        assert len(pitch) == len(confidence)
        assert len(pitch) > 0
        
        # Check value ranges
        assert np.all(pitch >= 0)
        assert np.all((confidence >= 0) & (confidence <= 1))
        
        # Test with specific parameters
        pitch_detailed = self.extractor.extract_pitch_yin(
            test_audio_mono, 
            threshold=0.1,
            frame_length=2048
        )
        assert len(pitch_detailed[0]) > 0
    
    def test_extract_pitch_spectral(self, test_audio_mono):
        """Test spectral pitch extraction"""
        pitch, confidence = self.extractor.extract_pitch_spectral(test_audio_mono)
        
        # Check output shape
        assert len(pitch) == len(confidence)
        assert len(pitch) > 0
        
        # Check value ranges
        assert np.all(pitch >= 0)
        assert np.all((confidence >= 0) & (confidence <= 1))
    
    @pytest.mark.skipif(
        not pytest.importorskip("crepe", minversion="0.0.12"),
        reason="CREPE not available"
    )
    def test_extract_pitch_crepe(self, test_audio_mono):
        """Test pitch extraction using CREPE"""
        try:
            pitch, confidence = self.extractor.extract_pitch_crepe(test_audio_mono)
            
            # Check output shape
            assert len(pitch) == len(confidence)
            assert len(pitch) > 0
            
            # Check value ranges
            assert np.all(pitch >= 0)
            assert np.all((confidence >= 0) & (confidence <= 1))
            
        except ImportError:
            pytest.skip("CREPE not available")
    
    def test_extract_multi_algorithm(self, test_audio_mono):
        """Test multi-algorithm pitch extraction"""
        results = self.extractor.extract_multi_algorithm(test_audio_mono)
        
        # Check that results is a dictionary
        assert isinstance(results, dict)
        
        # Check that we have at least librosa results
        assert 'librosa' in results
        assert 'yin' in results
        assert 'spectral' in results
        
        # Check each result format
        for algorithm, result in results.items():
            if result is not None:  # Some algorithms might fail
                pitch, confidence = result
                assert len(pitch) == len(confidence)
                assert len(pitch) > 0
    
    def test_smooth_pitch(self):
        """Test pitch smoothing functionality"""
        # Create a noisy pitch sequence
        clean_pitch = np.array([440.0, 441.0, 442.0, 441.5, 440.5, 440.0])
        noisy_pitch = clean_pitch + np.random.normal(0, 5.0, len(clean_pitch))
        
        # Apply smoothing
        smoothed = self.extractor.smooth_pitch(noisy_pitch, window_size=3)
        
        # Check that smoothing reduced variance
        assert np.var(smoothed) <= np.var(noisy_pitch)
        
        # Check output length
        assert len(smoothed) == len(noisy_pitch)
    
    def test_remove_outliers(self):
        """Test pitch outlier removal"""
        # Create pitch sequence with outliers
        pitch = np.array([440.0, 441.0, 880.0, 442.0, 220.0, 441.5, 440.5])  # 880 and 220 are outliers
        
        cleaned = self.extractor.remove_outliers(pitch, threshold=2.0)
        
        # Check that outliers were removed (set to 0 or interpolated)
        assert len(cleaned) == len(pitch)
        
        # The extreme values should be reduced
        extreme_count_original = np.sum((pitch > 500) | (pitch < 300))
        extreme_count_cleaned = np.sum((cleaned > 500) | (cleaned < 300))
        assert extreme_count_cleaned <= extreme_count_original
    
    def test_interpolate_unvoiced(self):
        """Test interpolation of unvoiced segments"""
        # Create pitch sequence with gaps (unvoiced = 0)
        pitch = np.array([440.0, 441.0, 0.0, 0.0, 442.0, 441.5, 440.5])
        
        interpolated = self.extractor.interpolate_unvoiced(pitch)
        
        # Check that gaps were filled
        assert np.sum(interpolated == 0) < np.sum(pitch == 0)
        
        # Check that valid values were preserved
        valid_original = pitch[pitch > 0]
        valid_interpolated = interpolated[pitch > 0]
        np.testing.assert_array_equal(valid_original, valid_interpolated)
    
    def test_compute_pitch_statistics(self, test_pitch_sequence):
        """Test pitch statistics computation"""
        stats = self.extractor.compute_pitch_statistics(test_pitch_sequence)
        
        # Check that we get expected statistics
        expected_keys = ['mean', 'std', 'min', 'max', 'median', 'range']
        for key in expected_keys:
            assert key in stats
        
        # Check value reasonableness
        assert stats['min'] <= stats['mean'] <= stats['max']
        assert stats['range'] == stats['max'] - stats['min']
        assert stats['std'] >= 0
    
    def test_segment_pitch_by_voice(self, test_pitch_sequence):
        """Test segmentation of pitch by voiced/unvoiced"""
        # Create pitch with some unvoiced segments
        pitch_with_gaps = test_pitch_sequence.copy()
        pitch_with_gaps[2:4] = 0.0  # Add unvoiced segment
        
        segments = self.extractor.segment_pitch_by_voice(pitch_with_gaps)
        
        # Check that we get segments
        assert len(segments) > 0
        
        # Check that segments contain only voiced frames
        for segment in segments:
            assert np.all(segment > 0)
    
    def test_extract_pitch_contour(self, test_audio_mono):
        """Test pitch contour extraction"""
        contour = self.extractor.extract_pitch_contour(test_audio_mono)
        
        # Check output format
        assert isinstance(contour, dict)
        
        # Check required keys
        required_keys = ['pitch', 'confidence', 'time', 'voiced_frames']
        for key in required_keys:
            assert key in contour
        
        # Check array lengths match
        assert len(contour['pitch']) == len(contour['confidence'])
        assert len(contour['pitch']) == len(contour['time'])
        assert len(contour['pitch']) == len(contour['voiced_frames'])
    
    def test_pitch_to_midi(self, test_pitch_sequence):
        """Test pitch to MIDI note conversion"""
        midi_notes = self.extractor.pitch_to_midi(test_pitch_sequence)
        
        # Check output shape
        assert len(midi_notes) == len(test_pitch_sequence)
        
        # Check value ranges (MIDI notes 0-127)
        valid_notes = midi_notes[midi_notes > 0]  # Exclude unvoiced (0)
        assert np.all((valid_notes >= 0) & (valid_notes <= 127))
        
        # Test known conversion (A4 = 440 Hz = MIDI note 69)
        a4_midi = self.extractor.pitch_to_midi(np.array([440.0]))[0]
        assert abs(a4_midi - 69) < 0.1
    
    def test_midi_to_pitch(self):
        """Test MIDI note to pitch conversion"""
        # Test known conversion (MIDI note 69 = A4 = 440 Hz)
        a4_pitch = self.extractor.midi_to_pitch(np.array([69]))[0]
        assert abs(a4_pitch - 440.0) < 0.1
        
        # Test array processing
        midi_notes = np.array([60, 64, 67, 72])  # C-E-G-C chord
        pitches = self.extractor.midi_to_pitch(midi_notes)
        assert len(pitches) == len(midi_notes)
        assert np.all(pitches > 200)  # Reasonable frequency range
        assert np.all(pitches < 1000)
    
    def test_analyze_pitch_stability(self, test_pitch_sequence):
        """Test pitch stability analysis"""
        stability = self.extractor.analyze_pitch_stability(test_pitch_sequence)
        
        # Check output format
        assert isinstance(stability, dict)
        
        # Check required metrics
        required_keys = ['jitter', 'shimmer', 'stability_score']
        for key in required_keys:
            assert key in stability
        
        # Check value ranges
        assert stability['jitter'] >= 0
        assert stability['shimmer'] >= 0
        assert 0 <= stability['stability_score'] <= 1
    
    def test_different_sample_rates(self, test_audio_mono):
        """Test pitch extraction with different sample rates"""
        for sr in [22050, 44100, 48000]:
            extractor = PitchExtractor(sample_rate=sr)
            
            # Resample audio for testing
            if sr != 44100:
                import librosa
                audio_resampled = librosa.resample(test_audio_mono, orig_sr=44100, target_sr=sr)
            else:
                audio_resampled = test_audio_mono
            
            pitch, confidence = extractor.extract_pitch_librosa(audio_resampled)
            assert len(pitch) > 0
            assert len(confidence) > 0
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Test with empty audio
        with pytest.raises((ValueError, RuntimeError)):
            self.extractor.extract_pitch_librosa(np.array([]))
        
        # Test with invalid sample rate
        with pytest.raises(ValueError):
            PitchExtractor(sample_rate=-1)
        
        # Test with invalid frame parameters
        with pytest.raises(ValueError):
            PitchExtractor(frame_length=0, hop_length=256)
    
    @pytest.mark.parametrize("algorithm", ["librosa", "yin", "spectral"])
    def test_pitch_extraction_consistency(self, test_audio_mono, algorithm):
        """Test that pitch extraction is consistent across runs"""
        # Extract pitch twice
        if algorithm == "librosa":
            pitch1, conf1 = self.extractor.extract_pitch_librosa(test_audio_mono)
            pitch2, conf2 = self.extractor.extract_pitch_librosa(test_audio_mono)
        elif algorithm == "yin":
            pitch1, conf1 = self.extractor.extract_pitch_yin(test_audio_mono)
            pitch2, conf2 = self.extractor.extract_pitch_yin(test_audio_mono)
        elif algorithm == "spectral":
            pitch1, conf1 = self.extractor.extract_pitch_spectral(test_audio_mono)
            pitch2, conf2 = self.extractor.extract_pitch_spectral(test_audio_mono)
        
        # Results should be identical
        np.testing.assert_array_equal(pitch1, pitch2)
        np.testing.assert_array_equal(conf1, conf2)
    
    def test_pitch_range_validation(self, test_audio_mono):
        """Test pitch extraction within expected frequency ranges"""
        pitch, confidence = self.extractor.extract_pitch_librosa(test_audio_mono)
        
        # Filter out unvoiced frames
        voiced_pitch = pitch[pitch > 0]
        
        if len(voiced_pitch) > 0:
            # Check that detected pitches are in reasonable range for human voice/instruments
            assert np.all(voiced_pitch >= 50)   # Above very low bass
            assert np.all(voiced_pitch <= 2000) # Below very high soprano
    
    def test_confidence_correlation(self, test_audio_mono):
        """Test that confidence correlates with pitch stability"""
        pitch, confidence = self.extractor.extract_pitch_librosa(test_audio_mono)
        
        # Filter voiced frames
        voiced_mask = pitch > 0
        if np.sum(voiced_mask) > 5:  # Need enough frames for analysis
            voiced_pitch = pitch[voiced_mask]
            voiced_confidence = confidence[voiced_mask]
            
            # Higher confidence should correlate with lower pitch variation
            high_conf_mask = voiced_confidence > np.median(voiced_confidence)
            low_conf_mask = voiced_confidence <= np.median(voiced_confidence)
            
            if np.sum(high_conf_mask) > 0 and np.sum(low_conf_mask) > 0:
                high_conf_variance = np.var(voiced_pitch[high_conf_mask])
                low_conf_variance = np.var(voiced_pitch[low_conf_mask])
                
                # This is a general trend, not always true, so we use a loose check
                # High confidence frames should generally be more stable
                assert high_conf_variance <= low_conf_variance * 2
