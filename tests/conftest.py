"""
Test configuration and fixtures for AutoTune ML Trainer
Created by Sergie Code
"""
import pytest
import torch
import numpy as np
import librosa
import os
import tempfile
import sys
from pathlib import Path

# Add src to path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

@pytest.fixture
def sample_rate():
    """Standard sample rate for testing"""
    return 44100

@pytest.fixture
def test_audio_mono(sample_rate):
    """Generate a simple mono audio signal for testing"""
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a simple sine wave with some pitch variation
    frequency = 440.0 + 50 * np.sin(2 * np.pi * 0.5 * t)  # A4 with slight vibrato
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)

@pytest.fixture 
def test_audio_stereo(test_audio_mono):
    """Convert mono test audio to stereo"""
    return np.stack([test_audio_mono, test_audio_mono * 0.8])

@pytest.fixture
def test_pitch_sequence():
    """Generate a test pitch sequence in Hz"""
    return np.array([440.0, 446.0, 441.5, 439.2, 442.1, 440.8])

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def test_audio_file(test_audio_mono, sample_rate, temp_dir):
    """Create a temporary audio file for testing"""
    filepath = os.path.join(temp_dir, "test_audio.wav")
    # Use librosa to write the audio file
    import soundfile as sf
    sf.write(filepath, test_audio_mono, sample_rate)
    return filepath

@pytest.fixture
def test_dataset_config():
    """Configuration for test dataset creation"""
    return {
        'segment_length': 1.0,  # seconds
        'hop_length': 0.5,      # seconds
        'sample_rate': 44100,
        'min_pitch': 80.0,      # Hz
        'max_pitch': 800.0,     # Hz
        'pitch_threshold': 0.3,  # confidence threshold
    }

@pytest.fixture
def test_model_config():
    """Configuration for test model creation"""
    return {
        'input_size': 512,
        'hidden_size': 256,
        'num_layers': 2,
        'output_size': 1,
        'dropout': 0.1,
        'sample_rate': 44100,
        'n_fft': 1024,
        'hop_length': 256,
    }

@pytest.fixture
def device():
    """Get available device for testing"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestAudioGenerator:
    """Helper class to generate various types of test audio"""
    
    @staticmethod
    def generate_sine_wave(frequency, duration, sample_rate, amplitude=0.5):
        """Generate a pure sine wave"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        return amplitude * np.sin(2 * np.pi * frequency * t)
    
    @staticmethod
    def generate_chirp(f0, f1, duration, sample_rate, amplitude=0.5):
        """Generate a frequency sweep (chirp)"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Linear frequency sweep
        frequency = f0 + (f1 - f0) * t / duration
        phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
        return amplitude * np.sin(phase)
    
    @staticmethod
    def generate_harmonic_tone(fundamental, duration, sample_rate, harmonics=3, amplitude=0.5):
        """Generate a tone with harmonics"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = np.zeros_like(t)
        
        for h in range(1, harmonics + 1):
            harmonic_amp = amplitude / h  # Decreasing amplitude for higher harmonics
            signal += harmonic_amp * np.sin(2 * np.pi * fundamental * h * t)
        
        return signal
    
    @staticmethod
    def add_noise(signal, noise_level=0.1):
        """Add white noise to a signal"""
        noise = np.random.normal(0, noise_level, signal.shape)
        return signal + noise

# Test constants
TEST_AUDIO_DURATION = 2.0  # seconds
TEST_SAMPLE_RATE = 44100
TEST_FUNDAMENTAL_FREQ = 440.0  # A4
TEST_BUFFER_SIZE = 512
TEST_N_FFT = 1024
TEST_HOP_LENGTH = 256

# Skip markers for optional dependencies
def _has_crepe():
    try:
        import crepe
        return True
    except ImportError:
        return False

def _has_matplotlib():
    try:
        import matplotlib
        return True
    except ImportError:
        return False

requires_crepe = pytest.mark.skipif(
    not _has_crepe(),
    reason="CREPE not available - install with: pip install crepe"
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

requires_matplotlib = pytest.mark.skipif(
    not _has_matplotlib(),
    reason="Matplotlib not available"
)
