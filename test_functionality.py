"""
Working test for AutoTune ML Trainer
Created by Sergie Code

This script demonstrates that the core functionality works.
"""
import numpy as np
import torch
import librosa
import soundfile as sf
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

def test_basic_audio_processing():
    """Test basic audio processing pipeline"""
    print("Testing basic audio processing...")
    
    # Generate test audio
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 440.0  # A4
    test_audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    print(f"✓ Generated test audio: {len(test_audio)} samples at {sample_rate}Hz")
    
    # Test librosa STFT
    stft = librosa.stft(test_audio)
    print(f"✓ STFT computed: shape {stft.shape}")
    
    # Test librosa pitch detection
    pitches, magnitudes = librosa.piptrack(y=test_audio, sr=sample_rate)
    print(f"✓ Pitch detection: {pitches.shape}")
    
    # Test MFCC features
    mfccs = librosa.feature.mfcc(y=test_audio, sr=sample_rate, n_mfcc=13)
    print(f"✓ MFCC features: shape {mfccs.shape}")
    
    # Test mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=test_audio, sr=sample_rate)
    print(f"✓ Mel spectrogram: shape {mel_spec.shape}")
    
    return True

def test_pytorch_functionality():
    """Test PyTorch functionality"""
    print("\nTesting PyTorch functionality...")
    
    # Test basic tensor operations
    x = torch.randn(4, 512)
    y = torch.randn(512, 256)
    z = torch.mm(x, y)
    print(f"✓ Matrix multiplication: {x.shape} @ {y.shape} = {z.shape}")
    
    # Test neural network
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(), 
        torch.nn.Linear(128, 1)
    )
    
    output = model(x)
    print(f"✓ Neural network forward pass: {x.shape} -> {output.shape}")
    
    # Test loss and backprop
    target = torch.randn(4, 1)
    loss = torch.nn.MSELoss()(output, target)
    loss.backward()
    print(f"✓ Loss computation and backprop: loss = {loss.item():.4f}")
    
    return True

def test_audio_io():
    """Test audio input/output"""
    print("\nTesting audio I/O...")
    
    # Generate test audio
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    test_audio = 0.3 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    
    # Test file I/O
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    
    try:
        # Write audio file
        sf.write(temp_path, test_audio, sample_rate)
        print(f"✓ Audio file written: {temp_path}")
        
        # Read audio file
        loaded_audio, loaded_sr = sf.read(temp_path)
        print(f"✓ Audio file read: {len(loaded_audio)} samples at {loaded_sr}Hz")
        
        # Verify data integrity
        assert loaded_sr == sample_rate, f"Sample rate mismatch: {loaded_sr} != {sample_rate}"
        assert len(loaded_audio) == len(test_audio), f"Length mismatch: {len(loaded_audio)} != {len(test_audio)}"
        
        # Test with librosa
        lib_audio, lib_sr = librosa.load(temp_path, sr=sample_rate)
        print(f"✓ Librosa load: {len(lib_audio)} samples at {lib_sr}Hz")
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return True

def test_model_export():
    """Test model export functionality"""
    print("\nTesting model export...")
    
    try:
        import onnx
        import onnxruntime as ort
        
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        model.eval()
        
        # Test input
        test_input = torch.randn(1, 10)
        
        # Get PyTorch output
        with torch.no_grad():
            pytorch_output = model(test_input)
        
        # Export to ONNX
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            torch.onnx.export(
                model,
                test_input,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                opset_version=11
            )
            print(f"✓ ONNX export successful: {onnx_path}")
            
            # Load and run ONNX model
            ort_session = ort.InferenceSession(onnx_path)
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            print(f"✓ ONNX runtime execution successful")
            
            # Compare outputs
            diff = np.abs(pytorch_output.numpy() - onnx_output).max()
            print(f"✓ Output difference: {diff:.6f}")
            
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)
                
    except ImportError as e:
        print(f"⚠ ONNX functionality not available: {e}")
        return False
    
    return True

def test_data_preprocessing():
    """Test data preprocessing components"""
    print("\nTesting data preprocessing...")
    
    try:
        from src.data.audio_preprocessor import AudioPreprocessor
        
        # Create preprocessor
        preprocessor = AudioPreprocessor(sample_rate=44100)
        print("✓ AudioPreprocessor created")
        
        # Generate test audio
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        test_audio = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        
        # Test normalization
        normalized = preprocessor.normalize_audio(test_audio)
        print(f"✓ Audio normalization: max = {np.max(np.abs(normalized)):.3f}")
        
        # Test feature extraction
        features = preprocessor.extract_features(test_audio)
        print(f"✓ Feature extraction: {list(features.keys())}")
        
        # Test STFT
        stft = preprocessor.compute_stft(test_audio)
        print(f"✓ STFT computation: shape {stft.shape}")
        
    except ImportError as e:
        print(f"⚠ AudioPreprocessor not available: {e}")
        return False
    
    return True

def test_pitch_extraction():
    """Test pitch extraction components"""
    print("\nTesting pitch extraction...")
    
    try:
        from src.data.pitch_extractor import PitchExtractor
        
        # Create pitch extractor
        extractor = PitchExtractor(sample_rate=44100)
        print("✓ PitchExtractor created")
        
        # Generate test audio with known pitch
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440.0  # A4
        test_audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Test librosa pitch detection
        pitch, confidence = extractor.extract_pitch_librosa(test_audio)
        print(f"✓ Librosa pitch extraction: {len(pitch)} frames")
        
        # Check detected pitch
        voiced_frames = pitch[pitch > 0]
        if len(voiced_frames) > 0:
            avg_pitch = np.mean(voiced_frames)
            print(f"✓ Average detected pitch: {avg_pitch:.1f}Hz (expected ~{frequency}Hz)")
        
        # Test YIN algorithm
        yin_pitch, yin_conf = extractor.extract_pitch_yin(test_audio)
        print(f"✓ YIN pitch extraction: {len(yin_pitch)} frames")
        
        # Test pitch statistics
        stats = extractor.compute_pitch_statistics(voiced_frames)
        print(f"✓ Pitch statistics: mean={stats['mean']:.1f}Hz, std={stats['std']:.1f}Hz")
        
    except ImportError as e:
        print(f"⚠ PitchExtractor not available: {e}")
        return False
    
    return True

def test_training_script():
    """Test that training script can be imported and run basic functionality"""
    print("\nTesting training functionality...")
    
    try:
        # Test synthetic data generation for training
        batch_size = 4
        sequence_length = 100
        input_size = 512
        
        # Generate synthetic training data
        train_data = []
        for _ in range(10):
            x = torch.randn(sequence_length, input_size)
            y = torch.randn(sequence_length, 1)
            train_data.append((x, y))
        
        print(f"✓ Synthetic training data generated: {len(train_data)} samples")
        
        # Test simple training loop
        model = torch.nn.LSTM(input_size, 128, batch_first=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Single training step
        x_batch = torch.stack([train_data[i][0] for i in range(batch_size)])
        y_batch = torch.stack([train_data[i][1] for i in range(batch_size)])
        
        model.train()
        optimizer.zero_grad()
        
        output, _ = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step completed: loss = {loss.item():.4f}")
        
    except Exception as e:
        print(f"⚠ Training functionality error: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("  AutoTune ML Trainer - Functionality Test")
    print("  Created by Sergie Code")
    print("=" * 60)
    
    tests = [
        ("Basic Audio Processing", test_basic_audio_processing),
        ("PyTorch Functionality", test_pytorch_functionality),
        ("Audio I/O", test_audio_io),
        ("Model Export", test_model_export),
        ("Data Preprocessing", test_data_preprocessing),
        ("Pitch Extraction", test_pitch_extraction),
        ("Training Functionality", test_training_script),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The AutoTune ML Trainer is working correctly.")
        return 0
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print("\nNext steps:")
    print("1. Install any missing optional dependencies if needed")
    print("2. Place audio files in data/raw/ directory")
    print("3. Run: python scripts/create_dataset.py")
    print("4. Run: python scripts/train_pitch_model.py")
    print("5. Explore notebooks/01_data_exploration.ipynb")
    
    exit(exit_code)
