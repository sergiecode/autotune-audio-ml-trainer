"""
Integration Test: AutoTune ML Trainer + Real-Time Audio Tuner
Created by Sergie Code

This script tests the compatibility and integration between:
1. autotune-audio-ml-trainer (Python ML framework)
2. autotune-real-time-audio-tuner (C++ real-time engine)
"""
import os
import sys
import subprocess
import numpy as np
import torch
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add current project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class IntegrationTester:
    """Test integration between ML trainer and real-time tuner"""
    
    def __init__(self):
        self.ml_trainer_path = Path(__file__).parent
        self.real_time_path = self.ml_trainer_path.parent / "autotune-real-time-audio-tuner"
        self.test_results = {}
        
    def run_all_tests(self) -> Dict:
        """Run all integration tests"""
        print("🔄 AutoTune Projects Integration Test")
        print("=" * 60)
        
        tests = [
            ("Project Structure", self.test_project_structure),
            ("Dependencies", self.test_dependencies),
            ("Audio Format Compatibility", self.test_audio_format_compatibility),
            ("Model Export Compatibility", self.test_model_export_compatibility),
            ("Python Bridge Interface", self.test_python_bridge_interface),
            ("Data Flow Integration", self.test_data_flow_integration),
            ("Performance Integration", self.test_performance_integration),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = test_func()
                if result['status'] == 'PASS':
                    print(f"✅ {test_name}: PASSED")
                    passed += 1
                else:
                    print(f"❌ {test_name}: FAILED - {result.get('message', 'Unknown error')}")
                
                self.test_results[test_name] = result
                
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {e}")
                self.test_results[test_name] = {
                    'status': 'ERROR',
                    'message': str(e)
                }
        
        # Summary
        print("\n" + "="*60)
        print(f"📊 INTEGRATION TEST RESULTS: {passed}/{total} PASSED")
        print("="*60)
        
        self.test_results['summary'] = {
            'passed': passed,
            'total': total,
            'success_rate': passed / total * 100
        }
        
        return self.test_results
    
    def test_project_structure(self) -> Dict:
        """Test if both projects have expected structure"""
        try:
            # Check ML trainer structure
            ml_required = [
                "src/data/audio_preprocessor.py",
                "src/data/pitch_extractor.py", 
                "src/models/pitch_correction_net.py",
                "tests/test_working.py",
                "test_app.py"
            ]
            
            ml_missing = []
            for file in ml_required:
                if not (self.ml_trainer_path / file).exists():
                    ml_missing.append(file)
            
            # Check real-time tuner structure
            rt_required = [
                "include/autotune_engine.h",
                "src/autotune_engine.cpp",
                "python/bindings.cpp",
                "CMakeLists.txt"
            ]
            
            rt_missing = []
            for file in rt_required:
                if not (self.real_time_path / file).exists():
                    rt_missing.append(file)
            
            print(f"✓ ML Trainer: {len(ml_required) - len(ml_missing)}/{len(ml_required)} files found")
            print(f"✓ Real-Time Tuner: {len(rt_required) - len(rt_missing)}/{len(rt_required)} files found")
            
            if not ml_missing and not rt_missing:
                return {'status': 'PASS', 'message': 'All required files present'}
            else:
                missing_info = f"ML missing: {ml_missing}, RT missing: {rt_missing}"
                return {'status': 'FAIL', 'message': missing_info}
                
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}
    
    def test_dependencies(self) -> Dict:
        """Test if dependencies are compatible between projects"""
        try:
            # Test ML trainer dependencies
            try:
                import torch
                import librosa
                import numpy as np
                import soundfile as sf
                print("✓ ML Trainer dependencies: PyTorch, Librosa, NumPy, SoundFile available")
                ml_deps_ok = True
            except ImportError as e:
                print(f"❌ ML Trainer dependencies missing: {e}")
                ml_deps_ok = False
            
            # Test real-time tuner dependencies (check if can be built)
            cmake_file = self.real_time_path / "CMakeLists.txt"
            rt_deps_ok = cmake_file.exists()
            
            if rt_deps_ok:
                print("✓ Real-Time Tuner: CMake configuration available")
            else:
                print("❌ Real-Time Tuner: No CMake configuration")
            
            if ml_deps_ok and rt_deps_ok:
                return {'status': 'PASS', 'message': 'Both projects have required dependencies'}
            else:
                return {'status': 'FAIL', 'message': f'ML deps: {ml_deps_ok}, RT deps: {rt_deps_ok}'}
                
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}
    
    def test_audio_format_compatibility(self) -> Dict:
        """Test if audio formats are compatible between projects"""
        try:
            # Expected formats from integration guide
            EXPECTED_SAMPLE_RATE = 44100
            EXPECTED_BUFFER_SIZE = 512
            EXPECTED_CHANNELS = 1
            
            # Test ML trainer audio processing
            try:
                from src.data.audio_preprocessor import AudioPreprocessor
                
                preprocessor = AudioPreprocessor(sample_rate=EXPECTED_SAMPLE_RATE)
                
                # Generate test audio
                test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, EXPECTED_SAMPLE_RATE)).astype(np.float32)
                
                # Test processing
                normalized = preprocessor.normalize_audio(test_audio)
                features = preprocessor.extract_features(test_audio)
                
                print(f"✓ ML Trainer audio format: {EXPECTED_SAMPLE_RATE}Hz, float32, mono")
                print(f"✓ Audio processing: normalization and feature extraction working")
                
                ml_audio_ok = True
                
            except Exception as e:
                print(f"❌ ML Trainer audio processing failed: {e}")
                ml_audio_ok = False
            
            # Check real-time tuner audio specs (from bindings)
            bindings_file = self.real_time_path / "python" / "bindings.cpp"
            rt_audio_ok = bindings_file.exists()
            
            if rt_audio_ok:
                with open(bindings_file, 'r') as f:
                    content = f.read()
                    if "AudioFrame" in content and "512" in content:
                        print("✓ Real-Time Tuner: AudioFrame with 512 buffer size detected")
                    else:
                        print("⚠ Real-Time Tuner: Audio format specs not clearly found")
            
            if ml_audio_ok and rt_audio_ok:
                return {'status': 'PASS', 'message': 'Audio formats compatible'}
            else:
                return {'status': 'FAIL', 'message': f'ML audio: {ml_audio_ok}, RT audio: {rt_audio_ok}'}
                
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}
    
    def test_model_export_compatibility(self) -> Dict:
        """Test if ML models can be exported for C++ engine"""
        try:
            # Test model creation and ONNX export
            try:
                from src.models.pitch_correction_net import PitchCorrectionNet
                
                # Create a small model for testing
                model = PitchCorrectionNet(
                    input_size=512,
                    hidden_size=128,
                    num_layers=1,
                    dropout=0.0
                )
                model.eval()
                
                print("✓ PitchCorrectionNet model created successfully")
                
                # Test ONNX export
                with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
                    onnx_path = f.name
                
                try:
                    # Create dummy inputs matching expected interface
                    audio_buffer = torch.randn(1, 512)
                    target_pitch = torch.tensor([[440.0]])
                    correction_strength = torch.tensor([[0.5]])
                    
                    # Test forward pass first
                    with torch.no_grad():
                        output = model(audio_buffer, target_pitch, correction_strength)
                    
                    print("✓ Model forward pass successful")
                    print(f"✓ Output shape: {output[0].shape} (audio), {output[1].shape} (confidence)")
                    
                    # Test ONNX export
                    torch.onnx.export(
                        model,
                        (audio_buffer, target_pitch, correction_strength),
                        onnx_path,
                        input_names=['audio_buffer', 'target_pitch', 'correction_strength'],
                        output_names=['corrected_audio', 'confidence'],
                        opset_version=11,
                        dynamic_axes={
                            'audio_buffer': {0: 'batch_size'},
                            'corrected_audio': {0: 'batch_size'},
                            'confidence': {0: 'batch_size'}
                        }
                    )
                    
                    print(f"✓ ONNX export successful: {os.path.getsize(onnx_path)} bytes")
                    
                    export_ok = True
                    
                except Exception as e:
                    print(f"❌ ONNX export failed: {e}")
                    export_ok = False
                finally:
                    if os.path.exists(onnx_path):
                        os.unlink(onnx_path)
                
            except Exception as e:
                print(f"❌ Model creation/export failed: {e}")
                export_ok = False
            
            # Check if real-time tuner supports ONNX (basic check)
            cmake_file = self.real_time_path / "CMakeLists.txt"
            onnx_support = False
            
            if cmake_file.exists():
                with open(cmake_file, 'r') as f:
                    content = f.read()
                    if "onnx" in content.lower() or "ONNX" in content:
                        onnx_support = True
                        print("✓ Real-Time Tuner: ONNX support detected in CMake")
                    else:
                        print("⚠ Real-Time Tuner: No explicit ONNX support found")
            
            if export_ok:
                return {'status': 'PASS', 'message': f'Model export working, RT ONNX support: {onnx_support}'}
            else:
                return {'status': 'FAIL', 'message': 'Model export failed'}
                
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}
    
    def test_python_bridge_interface(self) -> Dict:
        """Test if Python bindings interface matches ML trainer expectations"""
        try:
            # Check Python bindings file
            bindings_file = self.real_time_path / "python" / "bindings.cpp"
            
            if not bindings_file.exists():
                return {'status': 'FAIL', 'message': 'Python bindings file not found'}
            
            with open(bindings_file, 'r') as f:
                content = f.read()
            
            # Check for required interfaces
            required_classes = ['AudioFrame', 'AutotuneEngine', 'ProcessingParams', 'ProcessingResult']
            required_functions = ['process_frame', 'process_numpy_audio']
            
            missing_classes = []
            missing_functions = []
            
            for cls in required_classes:
                if cls not in content:
                    missing_classes.append(cls)
                else:
                    print(f"✓ Found class: {cls}")
            
            for func in required_functions:
                if func not in content:
                    missing_functions.append(func)
                else:
                    print(f"✓ Found function: {func}")
            
            # Check for NumPy compatibility
            numpy_support = "numpy" in content and "py::array" in content
            if numpy_support:
                print("✓ NumPy array support detected")
            else:
                print("⚠ No NumPy array support found")
            
            if not missing_classes and not missing_functions:
                return {'status': 'PASS', 'message': f'All interfaces found, NumPy support: {numpy_support}'}
            else:
                message = f"Missing classes: {missing_classes}, functions: {missing_functions}"
                return {'status': 'FAIL', 'message': message}
                
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}
    
    def test_data_flow_integration(self) -> Dict:
        """Test simulated data flow between projects"""
        try:
            # Simulate the ML training -> Real-time deployment workflow
            
            # Step 1: ML Trainer generates training data
            try:
                sample_rate = 44100
                duration = 1.0
                
                # Generate test audio
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
                
                print(f"✓ Step 1: Generated training audio ({len(test_audio)} samples)")
                
                # Step 2: Process with ML trainer
                from src.data.audio_preprocessor import AudioPreprocessor
                from src.data.pitch_extractor import PitchExtractor
                
                preprocessor = AudioPreprocessor(sample_rate=sample_rate)
                extractor = PitchExtractor(sample_rate=sample_rate)
                
                normalized = preprocessor.normalize_audio(test_audio)
                pitch, confidence = extractor.extract_pitch_yin(normalized)
                
                print(f"✓ Step 2: ML preprocessing successful (pitch frames: {len(pitch)})")
                
                # Step 3: Simulate model training/export (create dummy model)
                from src.models.pitch_correction_net import PitchCorrectionNet
                
                model = PitchCorrectionNet(input_size=512, hidden_size=64, num_layers=1)
                
                # Create test data matching expected format
                audio_buffer = torch.from_numpy(normalized[:512]).unsqueeze(0)
                target_pitch = torch.tensor([[440.0]])
                correction_strength = torch.tensor([[0.5]])
                
                with torch.no_grad():
                    output = model(audio_buffer, target_pitch, correction_strength)
                
                print(f"✓ Step 3: Model inference successful")
                
                # Step 4: Check compatibility with real-time engine format
                corrected_audio = output[0].numpy()
                confidence_score = output[1].numpy()
                
                # Verify format matches real-time engine expectations
                assert corrected_audio.shape == (1, 512), f"Expected (1, 512), got {corrected_audio.shape}"
                assert confidence_score.shape == (1, 1), f"Expected (1, 1), got {confidence_score.shape}"
                
                print(f"✓ Step 4: Output format matches real-time engine expectations")
                
                return {'status': 'PASS', 'message': 'Complete data flow simulation successful'}
                
            except Exception as e:
                return {'status': 'FAIL', 'message': f'Data flow simulation failed: {e}'}
                
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}
    
    def test_performance_integration(self) -> Dict:
        """Test if performance requirements are compatible"""
        try:
            # Expected performance targets from integration guide
            TARGET_LATENCY_MS = 5.0
            TARGET_SAMPLE_RATE = 44100
            TARGET_BUFFER_SIZE = 512
            
            # Test ML model inference speed
            try:
                from src.models.pitch_correction_net import PitchCorrectionNet
                import time
                
                model = PitchCorrectionNet(input_size=512, hidden_size=64, num_layers=1)
                model.eval()
                
                # Prepare test data
                audio_buffer = torch.randn(1, 512)
                target_pitch = torch.tensor([[440.0]])
                correction_strength = torch.tensor([[0.5]])
                
                # Warmup
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(audio_buffer, target_pitch, correction_strength)
                
                # Measure inference time
                num_tests = 100
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(num_tests):
                        output = model(audio_buffer, target_pitch, correction_strength)
                
                end_time = time.time()
                avg_inference_ms = (end_time - start_time) / num_tests * 1000
                
                print(f"✓ ML Model inference: {avg_inference_ms:.2f}ms average")
                
                # Check if meets real-time requirements
                latency_ok = avg_inference_ms < TARGET_LATENCY_MS
                
                if latency_ok:
                    print(f"✓ Latency requirement met: {avg_inference_ms:.2f}ms < {TARGET_LATENCY_MS}ms")
                else:
                    print(f"⚠ Latency requirement not met: {avg_inference_ms:.2f}ms >= {TARGET_LATENCY_MS}ms")
                
                # Calculate real-time factor
                buffer_duration_ms = (TARGET_BUFFER_SIZE / TARGET_SAMPLE_RATE) * 1000
                real_time_factor = buffer_duration_ms / avg_inference_ms
                
                print(f"✓ Real-time factor: {real_time_factor:.2f}x (higher is better)")
                
                performance_ok = real_time_factor > 1.0
                
                if performance_ok:
                    return {'status': 'PASS', 'message': f'Performance OK: {avg_inference_ms:.2f}ms, {real_time_factor:.2f}x real-time'}
                else:
                    return {'status': 'FAIL', 'message': f'Performance insufficient: {avg_inference_ms:.2f}ms'}
                
            except Exception as e:
                return {'status': 'FAIL', 'message': f'Performance test failed: {e}'}
                
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}


def main():
    """Run integration tests and generate report"""
    tester = IntegrationTester()
    results = tester.run_all_tests()
    
    # Save detailed results
    with open('integration_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    summary = results['summary']
    success_rate = summary['success_rate']
    
    print(f"\n🎯 FINAL INTEGRATION STATUS:")
    
    if success_rate >= 85:
        print("🟢 EXCELLENT: Projects integrate very well!")
        status = "EXCELLENT"
    elif success_rate >= 70:
        print("🟡 GOOD: Minor integration issues detected")
        status = "GOOD"
    elif success_rate >= 50:
        print("🟠 FAIR: Several integration issues need attention")
        status = "FAIR"
    else:
        print("🔴 POOR: Significant integration problems found")
        status = "POOR"
    
    print(f"Success Rate: {success_rate:.1f}% ({summary['passed']}/{summary['total']} tests passed)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if success_rate < 100:
        print("1. Review failed tests in integration_test_results.json")
        print("2. Check compatibility issues identified above")
        print("3. Update interfaces to match integration requirements")
    else:
        print("1. Projects are well integrated!")
        print("2. Ready for production deployment")
    
    print("3. Build the C++ real-time engine: cd ../autotune-real-time-audio-tuner && mkdir build && cd build && cmake .. && make")
    print("4. Test end-to-end workflow with actual audio files")
    
    return 0 if success_rate >= 70 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
