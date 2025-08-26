"""
Unit tests for neural network models
Created by Sergie Code
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    from src.models.pitch_correction_net import (
        PitchCorrectionNet,
        MultiScalePitchCorrectionNet,
        ConvolutionalPitchCorrector
    )
except ImportError:
    # Fallback import
    sys.path.append(str(ROOT_DIR / "src" / "models"))
    from pitch_correction_net import (
        PitchCorrectionNet,
        MultiScalePitchCorrectionNet,
        ConvolutionalPitchCorrector
    )

class TestPitchCorrectionNet:
    """Test suite for PitchCorrectionNet model"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PitchCorrectionNet(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            dropout=0.1
        ).to(self.device)
        
        # Set model to eval mode for consistent testing
        self.model.eval()
    
    def test_model_initialization(self):
        """Test model initialization"""
        assert self.model.input_size == 512
        assert self.model.hidden_size == 256
        assert self.model.num_layers == 2
        assert self.model.dropout == 0.1
    
    def test_forward_pass(self):
        """Test forward pass with various input shapes"""
        batch_size = 4
        buffer_size = 512
        
        # Test normal input
        audio_buffer = torch.randn(batch_size, buffer_size).to(self.device)
        target_pitch = torch.randint(200, 800, (batch_size, 1)).float().to(self.device)  # Hz
        correction_strength = torch.rand(batch_size, 1).to(self.device)  # 0-1
        
        output = self.model(audio_buffer, target_pitch, correction_strength)
        
        # Check output tuple format
        assert isinstance(output, tuple)
        assert len(output) == 2
        
        corrected_audio, confidence = output
        
        # Check output shapes
        assert corrected_audio.shape == (batch_size, buffer_size)
        assert confidence.shape == (batch_size, 1)
        
        # Check output values are finite
        assert torch.all(torch.isfinite(corrected_audio))
        assert torch.all(torch.isfinite(confidence))
    
    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths"""
        batch_size = 2
        input_size = 512
        
        for seq_len in [10, 50, 100, 200]:
            x = torch.randn(batch_size, seq_len, input_size).to(self.device)
            output = self.model(x)
            
            expected_shape = (batch_size, seq_len, 1)
            assert output.shape == expected_shape
    
    def test_different_batch_sizes(self):
        """Test model with different batch sizes"""
        sequence_length = 100
        input_size = 512
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, sequence_length, input_size).to(self.device)
            output = self.model(x)
            
            expected_shape = (batch_size, sequence_length, 1)
            assert output.shape == expected_shape
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly"""
        batch_size = 2
        sequence_length = 50
        input_size = 512
        
        x = torch.randn(batch_size, sequence_length, input_size, requires_grad=True).to(self.device)
        target = torch.randn(batch_size, sequence_length, 1).to(self.device)
        
        # Forward pass
        output = self.model(x)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are finite
        for param in self.model.parameters():
            assert param.grad is not None
            assert torch.all(torch.isfinite(param.grad))
    
    def test_model_training_mode(self):
        """Test model behavior in training vs eval mode"""
        batch_size = 2
        sequence_length = 50
        input_size = 512
        
        x = torch.randn(batch_size, sequence_length, input_size).to(self.device)
        
        # Test in training mode
        self.model.train()
        output_train = self.model(x)
        
        # Test in eval mode
        self.model.eval()
        output_eval = self.model(x)
        
        # Outputs might be different due to dropout
        # But shapes should be the same
        assert output_train.shape == output_eval.shape
    
    def test_save_and_load_state_dict(self, tmp_path):
        """Test model state dict saving and loading"""
        # Save model state
        save_path = tmp_path / "model_state.pth"
        torch.save(self.model.state_dict(), save_path)
        
        # Create new model and load state
        new_model = PitchCorrectionNet(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            output_size=1,
            dropout=0.1
        ).to(self.device)
        
        new_model.load_state_dict(torch.load(save_path, map_location=self.device))
        
        # Test that models produce same output
        x = torch.randn(1, 10, 512).to(self.device)
        
        self.model.eval()
        new_model.eval()
        
        with torch.no_grad():
            output1 = self.model(x)
            output2 = new_model(x)
        
        torch.testing.assert_close(output1, output2)


class TestMultiScalePitchCorrectionNet:
    """Test suite for MultiScalePitchCorrectionNet model"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultiScalePitchCorrectionNet(
            input_size=512,
            hidden_sizes=[256, 128, 64]
        ).to(self.device)
        
        self.model.eval()
    
    def test_multi_scale_forward_pass(self):
        """Test forward pass of multi-scale model"""
        batch_size = 4
        sequence_length = 100
        input_size = 512
        
        x = torch.randn(batch_size, sequence_length, input_size).to(self.device)
        output = self.model(x)
        
        expected_shape = (batch_size, sequence_length, 1)
        assert output.shape == expected_shape
        assert torch.all(torch.isfinite(output))
    
    def test_multi_scale_architecture(self):
        """Test multi-scale architecture properties"""
        # Check that model has multiple scales
        assert hasattr(self.model, 'scales')
        assert len(self.model.scales) == 3  # Based on hidden_sizes
        
        # Check that fusion layer exists
        assert hasattr(self.model, 'fusion')


class TestConvolutionalPitchCorrector:
    """Test suite for ConvolutionalPitchCorrector model"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ConvolutionalPitchCorrector(
            input_channels=1,
            hidden_channels=64,
            num_layers=4,
            kernel_size=3,
            dropout=0.1
        ).to(self.device)
        
        self.model.eval()
    
    def test_conv_forward_pass(self):
        """Test forward pass of convolutional model"""
        batch_size = 4
        channels = 1
        height = 128  # Frequency bins
        width = 100   # Time frames
        
        x = torch.randn(batch_size, channels, height, width).to(self.device)
        output = self.model(x)
        
        # Output should maintain spatial dimensions
        expected_shape = (batch_size, 1, height, width)
        assert output.shape == expected_shape
        assert torch.all(torch.isfinite(output))
    
    def test_conv_different_input_sizes(self):
        """Test convolutional model with different input sizes"""
        batch_size = 2
        channels = 1
        
        for height, width in [(64, 50), (128, 100), (256, 200)]:
            x = torch.randn(batch_size, channels, height, width).to(self.device)
            output = self.model(x)
            
            expected_shape = (batch_size, 1, height, width)
            assert output.shape == expected_shape


class TestModelComparison:
    """Test suite for comparing different model architectures"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 512
        self.batch_size = 4
        self.sequence_length = 100
        
        # Create test input
        self.test_input = torch.randn(
            self.batch_size, self.sequence_length, self.input_size
        ).to(self.device)
        
        # Initialize all models
        self.models = {
            'basic': PitchCorrectionNet(
                input_size=self.input_size,
                hidden_size=256,
                num_layers=2,
                dropout=0.1
            ).to(self.device),
            'multi_scale': MultiScalePitchCorrectionNet(
                input_size=self.input_size,
                hidden_sizes=[256, 128, 64]
            ).to(self.device)
        }
        
        # Set all models to eval mode
        for model in self.models.values():
            model.eval()
    
    def test_all_models_forward_pass(self):
        """Test that all models can process the same input"""
        expected_shape = (self.batch_size, self.sequence_length, 1)
        
        for name, model in self.models.items():
            with torch.no_grad():
                if name == 'conv':
                    # Skip convolutional model as it needs different input format
                    continue
                
                output = model(self.test_input)
                assert output.shape == expected_shape, f"Model {name} output shape mismatch"
                assert torch.all(torch.isfinite(output)), f"Model {name} produced non-finite output"
    
    def test_model_parameter_counts(self):
        """Test and compare parameter counts of different models"""
        param_counts = {}
        
        for name, model in self.models.items():
            if name == 'conv':
                continue  # Skip conv model for this test
                
            param_count = sum(p.numel() for p in model.parameters())
            param_counts[name] = param_count
            
            # Ensure models have reasonable number of parameters
            assert param_count > 1000, f"Model {name} has too few parameters"
            assert param_count < 10_000_000, f"Model {name} has too many parameters"
        
        # Print parameter counts for comparison (captured in test output)
        print("\nModel parameter counts:")
        for name, count in param_counts.items():
            print(f"  {name}: {count:,} parameters")
    
    def test_model_computational_requirements(self):
        """Test computational requirements of different models"""
        for name, model in self.models.items():
            if name == 'conv':
                continue
                
            # Measure forward pass time (rough estimate)
            import time
            
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                _ = model(self.test_input)
                end_time = time.time()
                
                forward_time = end_time - start_time
                
                # Ensure forward pass is reasonably fast
                assert forward_time < 1.0, f"Model {name} forward pass too slow: {forward_time:.3f}s"
    
    def test_model_memory_usage(self):
        """Test memory usage of different models"""
        if torch.cuda.is_available():
            for name, model in self.models.items():
                if name == 'conv':
                    continue
                    
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                
                # Forward pass
                with torch.no_grad():
                    _ = model(self.test_input)
                
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - initial_memory
                
                # Ensure reasonable memory usage (less than 1GB for these small models)
                assert memory_used < 1e9, f"Model {name} uses too much memory: {memory_used/1e6:.1f}MB"
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_model_precision_support(self, dtype):
        """Test model support for different precisions"""
        if dtype == torch.float16 and not torch.cuda.is_available():
            pytest.skip("float16 testing requires CUDA")
        
        for name, model in self.models.items():
            if name == 'conv':
                continue
                
            # Convert model and input to specified precision
            model_test = model.to(dtype)
            input_test = self.test_input.to(dtype)
            
            with torch.no_grad():
                output = model_test(input_test)
                
                assert output.dtype == dtype
                assert torch.all(torch.isfinite(output))


class TestModelRobustness:
    """Test suite for model robustness and edge cases"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PitchCorrectionNet(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            output_size=1
        ).to(self.device)
        self.model.eval()
    
    def test_extreme_input_values(self):
        """Test model with extreme input values"""
        batch_size = 2
        sequence_length = 50
        input_size = 512
        
        # Test with very large values
        large_input = torch.full((batch_size, sequence_length, input_size), 1000.0).to(self.device)
        with torch.no_grad():
            output_large = self.model(large_input)
        assert torch.all(torch.isfinite(output_large))
        
        # Test with very small values
        small_input = torch.full((batch_size, sequence_length, input_size), 1e-6).to(self.device)
        with torch.no_grad():
            output_small = self.model(small_input)
        assert torch.all(torch.isfinite(output_small))
        
        # Test with zeros
        zero_input = torch.zeros(batch_size, sequence_length, input_size).to(self.device)
        with torch.no_grad():
            output_zero = self.model(zero_input)
        assert torch.all(torch.isfinite(output_zero))
    
    def test_single_sample_inference(self):
        """Test model with single sample (batch_size=1, sequence_length=1)"""
        single_input = torch.randn(1, 1, 512).to(self.device)
        
        with torch.no_grad():
            output = self.model(single_input)
        
        assert output.shape == (1, 1, 1)
        assert torch.all(torch.isfinite(output))
    
    def test_very_long_sequences(self):
        """Test model with very long sequences"""
        batch_size = 1
        long_sequence_length = 1000
        input_size = 512
        
        long_input = torch.randn(batch_size, long_sequence_length, input_size).to(self.device)
        
        with torch.no_grad():
            output = self.model(long_input)
        
        expected_shape = (batch_size, long_sequence_length, 1)
        assert output.shape == expected_shape
        assert torch.all(torch.isfinite(output))
    
    def test_model_determinism(self):
        """Test that model produces deterministic outputs"""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        batch_size = 2
        sequence_length = 50
        input_size = 512
        
        # Create identical inputs
        x1 = torch.randn(batch_size, sequence_length, input_size).to(self.device)
        x2 = x1.clone()
        
        self.model.eval()
        
        with torch.no_grad():
            output1 = self.model(x1)
            output2 = self.model(x2)
        
        # Outputs should be identical
        torch.testing.assert_close(output1, output2)
    
    def test_input_perturbation_sensitivity(self):
        """Test model sensitivity to small input perturbations"""
        batch_size = 2
        sequence_length = 50
        input_size = 512
        
        # Original input
        x_original = torch.randn(batch_size, sequence_length, input_size).to(self.device)
        
        # Slightly perturbed input
        perturbation = torch.randn_like(x_original) * 1e-6
        x_perturbed = x_original + perturbation
        
        self.model.eval()
        
        with torch.no_grad():
            output_original = self.model(x_original)
            output_perturbed = self.model(x_perturbed)
        
        # Outputs should be close but not necessarily identical
        difference = torch.abs(output_original - output_perturbed)
        max_difference = torch.max(difference)
        
        # The difference should be small relative to the perturbation
        assert max_difference < 1.0  # Reasonable threshold for stability
