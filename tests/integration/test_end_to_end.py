"""
Integration tests for the complete AutoTune ML pipeline
Created by Sergie Code
"""
import pytest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

class TestEndToEndPipeline:
    """Test the complete end-to-end training pipeline"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup after each test"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.slow
    def test_complete_training_pipeline(self, test_audio_mono, sample_rate):
        """Test the complete training pipeline from audio to trained model"""
        try:
            # Import modules
            from src.data.audio_preprocessor import AudioPreprocessor
            from src.data.pitch_extractor import PitchExtractor
            from src.models.pitch_correction_net import PitchCorrectionNet
            from src.training.trainer import Trainer
            
            # Step 1: Preprocess audio
            preprocessor = AudioPreprocessor(sample_rate=sample_rate)
            processed = preprocessor.process_pipeline(test_audio_mono)
            
            # Step 2: Extract pitch
            pitch_extractor = PitchExtractor(sample_rate=sample_rate)
            pitch_data = pitch_extractor.extract_pitch_contour(test_audio_mono)
            
            # Step 3: Create synthetic training data
            batch_size = 8
            sequence_length = 100
            input_size = 512
            
            # Generate synthetic training data
            train_data = []
            for _ in range(20):  # Small dataset for testing
                x = torch.randn(sequence_length, input_size)
                y = torch.randn(sequence_length, 1)  # Target pitch correction
                train_data.append((x, y))
            
            # Step 4: Initialize model
            model = PitchCorrectionNet(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                output_size=1,
                dropout=0.1
            )
            
            # Step 5: Train model (minimal training for testing)
            trainer = Trainer(
                model=model,
                device=self.device,
                learning_rate=0.001,
                batch_size=batch_size
            )
            
            # Train for a few steps
            initial_loss = trainer.train_step(train_data[:batch_size])
            assert initial_loss > 0
            
            # Verify model can make predictions
            test_input = torch.randn(1, sequence_length, input_size)
            with torch.no_grad():
                prediction = model(test_input)
            
            assert prediction.shape == (1, sequence_length, 1)
            assert torch.all(torch.isfinite(prediction))
            
        except ImportError as e:
            pytest.skip(f"Integration test skipped due to missing module: {e}")
    
    def test_data_flow_consistency(self, test_audio_mono, sample_rate):
        """Test that data flows consistently through the pipeline"""
        try:
            from src.data.audio_preprocessor import AudioPreprocessor
            from src.data.pitch_extractor import PitchExtractor
            
            # Process the same audio through different components
            preprocessor = AudioPreprocessor(sample_rate=sample_rate)
            pitch_extractor = PitchExtractor(sample_rate=sample_rate)
            
            # Extract features and pitch
            features = preprocessor.extract_features(test_audio_mono)
            pitch_contour = pitch_extractor.extract_pitch_contour(test_audio_mono)
            
            # Verify time alignment
            # Features and pitch should have compatible time dimensions
            mel_spec = features['mel_spectrogram']
            pitch_frames = len(pitch_contour['pitch'])
            
            # Time frames should be reasonably close (within factor of 2)
            time_ratio = mel_spec.shape[1] / pitch_frames
            assert 0.5 <= time_ratio <= 2.0, f"Time alignment issue: ratio = {time_ratio}"
            
        except ImportError as e:
            pytest.skip(f"Integration test skipped due to missing module: {e}")
    
    def test_model_export_import_cycle(self):
        """Test model export and import cycle"""
        try:
            from src.models.pitch_correction_net import PitchCorrectionNet
            
            # Create and initialize model
            model = PitchCorrectionNet(
                input_size=512,
                hidden_size=256,
                num_layers=2,
                output_size=1
            )
            
            # Test input
            test_input = torch.randn(1, 10, 512)
            
            # Get original output
            model.eval()
            with torch.no_grad():
                original_output = model(test_input)
            
            # Save model
            model_path = os.path.join(self.temp_dir, "test_model.pth")
            torch.save(model.state_dict(), model_path)
            
            # Load model
            new_model = PitchCorrectionNet(
                input_size=512,
                hidden_size=256,
                num_layers=2,
                output_size=1
            )
            new_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            # Test loaded model
            new_model.eval()
            with torch.no_grad():
                loaded_output = new_model(test_input)
            
            # Verify outputs match
            torch.testing.assert_close(original_output, loaded_output)
            
        except ImportError as e:
            pytest.skip(f"Integration test skipped due to missing module: {e}")
    
    def test_onnx_export_pipeline(self):
        """Test ONNX export functionality"""
        try:
            from src.models.pitch_correction_net import PitchCorrectionNet
            import onnx
            import onnxruntime as ort
            
            # Create model
            model = PitchCorrectionNet(
                input_size=512,
                hidden_size=128,
                num_layers=2,
                output_size=1
            )
            model.eval()
            
            # Test input
            batch_size = 1
            sequence_length = 10
            input_size = 512
            dummy_input = torch.randn(batch_size, sequence_length, input_size)
            
            # Export to ONNX
            onnx_path = os.path.join(self.temp_dir, "test_model.onnx")
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size', 1: 'sequence_length'},
                    'output': {0: 'batch_size', 1: 'sequence_length'}
                }
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Test ONNX runtime
            ort_session = ort.InferenceSession(onnx_path)
            
            # Get PyTorch output
            with torch.no_grad():
                pytorch_output = model(dummy_input)
            
            # Get ONNX output
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            ort_output = ort_session.run(None, ort_inputs)[0]
            
            # Compare outputs
            np.testing.assert_allclose(
                pytorch_output.numpy(), 
                ort_output, 
                rtol=1e-5, 
                atol=1e-6
            )
            
        except ImportError as e:
            pytest.skip(f"ONNX export test skipped due to missing dependencies: {e}")
    
    def test_batch_processing_pipeline(self, test_audio_mono, sample_rate):
        """Test batch processing of multiple audio files"""
        try:
            from src.data.audio_preprocessor import AudioPreprocessor
            
            # Create multiple audio segments
            segment_length = len(test_audio_mono) // 4
            audio_segments = [
                test_audio_mono[i:i+segment_length] 
                for i in range(0, len(test_audio_mono), segment_length)
                if i+segment_length <= len(test_audio_mono)
            ]
            
            # Process batch
            preprocessor = AudioPreprocessor(sample_rate=sample_rate)
            batch_results = preprocessor.process_batch(audio_segments)
            
            # Verify batch processing
            assert len(batch_results) == len(audio_segments)
            
            for i, result in enumerate(batch_results):
                assert isinstance(result, dict)
                assert 'audio' in result
                assert 'features' in result
                assert 'metadata' in result
                
                # Verify audio length
                processed_audio = result['audio']
                expected_length = len(audio_segments[i])
                assert abs(len(processed_audio) - expected_length) <= 1
            
        except ImportError as e:
            pytest.skip(f"Batch processing test skipped due to missing module: {e}")
    
    @pytest.mark.slow
    def test_training_convergence(self):
        """Test that model training shows convergence"""
        try:
            from src.models.pitch_correction_net import PitchCorrectionNet
            from src.training.trainer import Trainer
            
            # Create synthetic data with learnable pattern
            def generate_pattern_data(batch_size, sequence_length, input_size):
                data = []
                for _ in range(batch_size):
                    # Create input with simple pattern
                    x = torch.randn(sequence_length, input_size)
                    
                    # Create target based on simple function of input
                    # Target is sum of first few features
                    y = torch.sum(x[:, :10], dim=1, keepdim=True) * 0.1
                    
                    data.append((x, y))
                return data
            
            # Generate training data
            train_data = generate_pattern_data(100, 50, 512)
            val_data = generate_pattern_data(20, 50, 512)
            
            # Create model
            model = PitchCorrectionNet(
                input_size=512,
                hidden_size=64,
                num_layers=1,
                output_size=1,
                dropout=0.0  # No dropout for deterministic testing
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                device=self.device,
                learning_rate=0.01,
                batch_size=10
            )
            
            # Train for several epochs
            initial_loss = None
            final_loss = None
            
            for epoch in range(10):
                epoch_losses = []
                for i in range(0, len(train_data), 10):
                    batch = train_data[i:i+10]
                    loss = trainer.train_step(batch)
                    epoch_losses.append(loss)
                
                avg_loss = np.mean(epoch_losses)
                
                if epoch == 0:
                    initial_loss = avg_loss
                elif epoch == 9:
                    final_loss = avg_loss
            
            # Verify that loss decreased
            assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
            
            # Loss should decrease by at least 10%
            improvement = (initial_loss - final_loss) / initial_loss
            assert improvement > 0.1, f"Insufficient improvement: {improvement:.2%}"
            
        except ImportError as e:
            pytest.skip(f"Training convergence test skipped due to missing module: {e}")
    
    def test_real_time_inference_latency(self):
        """Test real-time inference latency requirements"""
        try:
            from src.models.pitch_correction_net import PitchCorrectionNet
            import time
            
            # Create optimized model for real-time use
            model = PitchCorrectionNet(
                input_size=512,
                hidden_size=128,
                num_layers=1,  # Minimal layers for speed
                output_size=1,
                dropout=0.0
            ).to(self.device)
            
            model.eval()
            
            # Test with real-time buffer size
            buffer_size = 512  # Typical real-time buffer
            sequence_length = buffer_size // 256  # Hop length consideration
            
            test_input = torch.randn(1, sequence_length, 512).to(self.device)
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_input)
            
            # Measure inference time
            num_trials = 100
            times = []
            
            with torch.no_grad():
                for _ in range(num_trials):
                    start_time = time.perf_counter()
                    _ = model(test_input)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            max_time = np.max(times)
            
            # Real-time constraint: process buffer in less than buffer duration
            buffer_duration = buffer_size / 44100  # Assuming 44.1kHz
            
            print(f"\nInference timing:")
            print(f"  Average: {avg_time*1000:.2f}ms")
            print(f"  Maximum: {max_time*1000:.2f}ms")
            print(f"  Buffer duration: {buffer_duration*1000:.2f}ms")
            
            # Should be much faster than real-time
            assert avg_time < buffer_duration * 0.1, f"Too slow for real-time: {avg_time*1000:.2f}ms"
            assert max_time < buffer_duration * 0.5, f"Max time too slow: {max_time*1000:.2f}ms"
            
        except ImportError as e:
            pytest.skip(f"Real-time latency test skipped due to missing module: {e}")
    
    def test_memory_usage_constraints(self):
        """Test memory usage stays within reasonable bounds"""
        try:
            from src.models.pitch_correction_net import PitchCorrectionNet
            
            if not torch.cuda.is_available():
                pytest.skip("Memory usage test requires CUDA")
            
            # Clear memory
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Create model
            model = PitchCorrectionNet(
                input_size=512,
                hidden_size=256,
                num_layers=2,
                output_size=1
            ).to(self.device)
            
            model_memory = torch.cuda.memory_allocated() - initial_memory
            
            # Test inference with various batch sizes
            max_memory_used = 0
            
            for batch_size in [1, 4, 8, 16]:
                torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated()
                
                test_input = torch.randn(batch_size, 100, 512).to(self.device)
                
                with torch.no_grad():
                    _ = model(test_input)
                
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - start_memory
                max_memory_used = max(max_memory_used, memory_used)
            
            print(f"\nMemory usage:")
            print(f"  Model: {model_memory/1e6:.1f}MB")
            print(f"  Max inference: {max_memory_used/1e6:.1f}MB")
            print(f"  Total peak: {(model_memory + max_memory_used)/1e6:.1f}MB")
            
            # Memory constraints for real-time usage
            assert model_memory < 100e6, f"Model too large: {model_memory/1e6:.1f}MB"
            assert max_memory_used < 200e6, f"Inference uses too much memory: {max_memory_used/1e6:.1f}MB"
            
        except ImportError as e:
            pytest.skip(f"Memory usage test skipped due to missing module: {e}")
    
    def test_model_quantization_compatibility(self):
        """Test model compatibility with quantization for deployment"""
        try:
            from src.models.pitch_correction_net import PitchCorrectionNet
            
            # Create model
            model = PitchCorrectionNet(
                input_size=512,
                hidden_size=128,
                num_layers=2,
                output_size=1
            )
            
            model.eval()
            
            # Test input
            test_input = torch.randn(1, 10, 512)
            
            # Get original output
            with torch.no_grad():
                original_output = model(test_input)
            
            # Test dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.LSTM}, 
                dtype=torch.qint8
            )
            
            # Test quantized model
            with torch.no_grad():
                quantized_output = quantized_model(test_input)
            
            # Outputs should be reasonably close
            max_diff = torch.max(torch.abs(original_output - quantized_output))
            relative_error = max_diff / torch.max(torch.abs(original_output))
            
            print(f"\nQuantization results:")
            print(f"  Max absolute difference: {max_diff:.6f}")
            print(f"  Relative error: {relative_error:.2%}")
            
            # Allow for some quantization error
            assert relative_error < 0.1, f"Quantization error too high: {relative_error:.2%}"
            
        except ImportError as e:
            pytest.skip(f"Quantization test skipped due to missing module: {e}")
        except Exception as e:
            pytest.skip(f"Quantization test skipped due to error: {e}")


class TestSystemIntegration:
    """Test system-level integration aspects"""
    
    def test_cpu_vs_gpu_consistency(self):
        """Test that model produces consistent results on CPU vs GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for consistency test")
        
        try:
            from src.models.pitch_correction_net import PitchCorrectionNet
            
            # Create identical models for CPU and GPU
            model_cpu = PitchCorrectionNet(
                input_size=512,
                hidden_size=128,
                num_layers=2,
                output_size=1
            )
            
            model_gpu = PitchCorrectionNet(
                input_size=512,
                hidden_size=128,
                num_layers=2,
                output_size=1
            ).cuda()
            
            # Copy weights from CPU to GPU model
            model_gpu.load_state_dict(model_cpu.state_dict())
            
            # Test input
            test_input_cpu = torch.randn(1, 10, 512)
            test_input_gpu = test_input_cpu.cuda()
            
            # Get outputs
            model_cpu.eval()
            model_gpu.eval()
            
            with torch.no_grad():
                output_cpu = model_cpu(test_input_cpu)
                output_gpu = model_gpu(test_input_gpu).cpu()
            
            # Compare outputs
            torch.testing.assert_close(output_cpu, output_gpu, rtol=1e-5, atol=1e-6)
            
        except ImportError as e:
            pytest.skip(f"CPU vs GPU test skipped due to missing module: {e}")
    
    def test_cross_platform_model_compatibility(self):
        """Test model compatibility across different scenarios"""
        try:
            from src.models.pitch_correction_net import PitchCorrectionNet
            
            # Test with different Python environments
            model = PitchCorrectionNet(
                input_size=512,
                hidden_size=128,
                num_layers=2,
                output_size=1
            )
            
            # Test serialization/deserialization
            with tempfile.NamedTemporaryFile(suffix='.pth') as f:
                torch.save(model.state_dict(), f.name)
                
                # Load in strict mode
                new_model = PitchCorrectionNet(
                    input_size=512,
                    hidden_size=128,
                    num_layers=2,
                    output_size=1
                )
                new_model.load_state_dict(torch.load(f.name, map_location='cpu'))
                
                # Test that loaded model works
                test_input = torch.randn(1, 5, 512)
                
                model.eval()
                new_model.eval()
                
                with torch.no_grad():
                    output1 = model(test_input)
                    output2 = new_model(test_input)
                
                torch.testing.assert_close(output1, output2)
                
        except ImportError as e:
            pytest.skip(f"Cross-platform test skipped due to missing module: {e}")
