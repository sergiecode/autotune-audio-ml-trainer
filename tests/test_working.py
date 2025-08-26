"""
Working Unit Tests for AutoTune ML Trainer
Created by Sergie Code

Simple tests that actually work with the existing code.
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.data.audio_preprocessor import AudioPreprocessor
from src.data.pitch_extractor import PitchExtractor
from src.models.pitch_correction_net import PitchCorrectionNet


class TestAudioPreprocessorWorking:
    """Working tests for AudioPreprocessor"""
    
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
    
    def test_normalize_audio(self):
        """Test audio normalization"""
        # Generate test audio
        test_audio = np.random.randn(44100).astype(np.float32) * 2.0  # Loud audio
        
        normalized = self.preprocessor.normalize_audio(test_audio)
        
        # Check that audio is normalized
        assert np.max(np.abs(normalized)) <= 1.0
        assert normalized.dtype == np.float32
    
    def test_extract_features(self):
        """Test feature extraction"""
        # Generate test audio
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)
        
        features = self.preprocessor.extract_features(test_audio)
        
        # Check that features is a dict with expected keys
        assert isinstance(features, dict)
        assert 'rms' in features
        assert 'peak' in features
        assert 'spectral_centroid' in features


class TestPitchExtractorWorking:
    """Working tests for PitchExtractor"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.extractor = PitchExtractor(
            sample_rate=44100,
            frame_length=2048,
            hop_length=512
        )
    
    def test_init(self):
        """Test PitchExtractor initialization"""
        assert self.extractor.sample_rate == 44100
        assert self.extractor.frame_length == 2048
        assert self.extractor.hop_length == 512
    
    def test_extract_pitch_yin(self):
        """Test YIN pitch extraction"""
        # Generate test audio with known pitch
        duration = 2.0
        sample_rate = 44100
        frequency = 440.0  # A4
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        test_audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        pitch, confidence = self.extractor.extract_pitch_yin(test_audio)
        
        # Check output format
        assert isinstance(pitch, np.ndarray)
        assert isinstance(confidence, np.ndarray)
        assert len(pitch) == len(confidence)
        assert len(pitch) > 0
    
    def test_extract_pitch_autocorr(self):
        """Test autocorrelation pitch extraction"""
        # Generate test audio
        duration = 1.0
        sample_rate = 44100
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        test_audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        pitch, confidence = self.extractor.extract_pitch_autocorr(test_audio)
        
        # Check output format
        assert isinstance(pitch, np.ndarray)
        assert isinstance(confidence, np.ndarray)
        assert len(pitch) == len(confidence)


class TestPitchCorrectionNetWorking:
    """Working tests for PitchCorrectionNet"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.device = torch.device("cpu")
        self.model = PitchCorrectionNet(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            dropout=0.1
        ).to(self.device)
    
    def test_model_initialization(self):
        """Test model creation"""
        assert isinstance(self.model, PitchCorrectionNet)
        assert self.model.input_size == 512
        assert self.model.hidden_size == 256
        assert self.model.num_layers == 2
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        assert total_params > 0
        print(f"Model has {total_params:,} parameters")
    
    def test_forward_pass_correct_interface(self):
        """Test forward pass with correct interface"""
        batch_size = 2
        buffer_size = 512
        
        # Create inputs with correct format
        audio_buffer = torch.randn(batch_size, buffer_size).to(self.device)
        target_pitch = torch.randint(200, 800, (batch_size, 1)).float().to(self.device)
        correction_strength = torch.rand(batch_size, 1).to(self.device)
        
        # Forward pass
        output = self.model(audio_buffer, target_pitch, correction_strength)
        
        # Check output format
        assert isinstance(output, tuple)
        assert len(output) == 2
        
        corrected_audio, confidence = output
        assert corrected_audio.shape == (batch_size, buffer_size)
        assert confidence.shape == (batch_size, 1)
        
        # Check output ranges
        assert torch.all(torch.isfinite(corrected_audio))
        assert torch.all(torch.isfinite(confidence))
    
    def test_different_batch_sizes(self):
        """Test model with different batch sizes"""
        buffer_size = 512
        
        for batch_size in [1, 2, 4, 8]:
            audio_buffer = torch.randn(batch_size, buffer_size).to(self.device)
            target_pitch = torch.randint(200, 800, (batch_size, 1)).float().to(self.device)
            correction_strength = torch.rand(batch_size, 1).to(self.device)
            
            output = self.model(audio_buffer, target_pitch, correction_strength)
            corrected_audio, confidence = output
            
            assert corrected_audio.shape == (batch_size, buffer_size)
            assert confidence.shape == (batch_size, 1)
    
    def test_training_mode(self):
        """Test model in training vs eval mode"""
        batch_size = 2
        buffer_size = 512
        
        audio_buffer = torch.randn(batch_size, buffer_size).to(self.device)
        target_pitch = torch.randint(200, 800, (batch_size, 1)).float().to(self.device)
        correction_strength = torch.rand(batch_size, 1).to(self.device)
        
        # Test training mode
        self.model.train()
        output_train = self.model(audio_buffer, target_pitch, correction_strength)
        
        # Test eval mode
        self.model.eval()
        with torch.no_grad():
            output_eval = self.model(audio_buffer, target_pitch, correction_strength)
        
        # Both should work (outputs may differ due to dropout)
        assert isinstance(output_train, tuple)
        assert isinstance(output_eval, tuple)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly"""
        batch_size = 2
        buffer_size = 512
        
        audio_buffer = torch.randn(batch_size, buffer_size, requires_grad=True).to(self.device)
        target_pitch = torch.randint(200, 800, (batch_size, 1)).float().to(self.device)
        correction_strength = torch.rand(batch_size, 1).to(self.device)
        
        # Forward pass
        output = self.model(audio_buffer, target_pitch, correction_strength)
        corrected_audio, confidence = output
        
        # Compute loss and backprop
        loss = torch.mean(corrected_audio ** 2) + torch.mean(confidence ** 2)
        loss.backward()
        
        # Check that gradients exist
        assert audio_buffer.grad is not None
        assert torch.any(audio_buffer.grad != 0)
        
        # Check model gradients
        has_grad = False
        for param in self.model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_grad = True
                break
        assert has_grad, "Model should have gradients after backprop"


def test_integration_simple():
    """Simple integration test"""
    # Create components
    preprocessor = AudioPreprocessor()
    extractor = PitchExtractor()
    model = PitchCorrectionNet(input_size=512, hidden_size=128, num_layers=1)
    
    # Generate test audio
    test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)
    
    # Preprocess
    normalized = preprocessor.normalize_audio(test_audio)
    features = preprocessor.extract_features(normalized)
    
    # Extract pitch
    pitch, confidence = extractor.extract_pitch_yin(normalized)
    
    # Test model (simplified)
    audio_buffer = torch.randn(1, 512)
    target_pitch = torch.tensor([[440.0]])
    correction_strength = torch.tensor([[0.5]])
    
    output = model(audio_buffer, target_pitch, correction_strength)
    
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(features, dict)
    assert len(pitch) > 0


if __name__ == "__main__":
    # Run tests directly
    import subprocess
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v"
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    exit(result.returncode)
