"""
Test runner script for AutoTune ML Trainer
Created by Sergie Code

This script runs all tests and provides a comprehensive test report.
"""
import pytest
import sys
import os
from pathlib import Path
import subprocess
import time

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

def run_tests():
    """Run all tests with appropriate configuration"""
    
    print("=" * 80)
    print("  AutoTune ML Trainer - Test Suite")
    print("  Created by Sergie Code")
    print("=" * 80)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"⚠ Missing dependencies: {', '.join(missing_deps)}")
        print("Some tests may be skipped.")
    else:
        print("✓ All dependencies available")
    print()
    
    # Test discovery
    test_dir = ROOT_DIR / "tests"
    print(f"Test directory: {test_dir}")
    print(f"Running tests from: {os.getcwd()}")
    print()
    
    # Configure pytest arguments
    pytest_args = [
        str(test_dir),
        "-v",                    # Verbose output
        "--tb=short",           # Short traceback format
        "--strict-markers",     # Strict marker validation
        "-ra",                  # Show short test summary for all except passed
        "--durations=10",       # Show 10 slowest tests
        "-x",                   # Stop on first failure (for quick debugging)
    ]
    
    # Add coverage if available
    try:
        import coverage
        pytest_args.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
        print("✓ Coverage reporting enabled")
    except ImportError:
        print("ℹ Coverage reporting not available (install pytest-cov)")
    
    print()
    print("Running tests...")
    print("-" * 40)
    
    start_time = time.time()
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print()
    print("-" * 40)
    print(f"Tests completed in {duration:.2f} seconds")
    
    # Interpret results
    if exit_code == 0:
        print("🎉 All tests passed!")
    elif exit_code == 1:
        print("❌ Some tests failed")
    elif exit_code == 2:
        print("⚠ Test execution was interrupted")
    elif exit_code == 3:
        print("❌ Internal pytest error")
    elif exit_code == 4:
        print("❌ pytest command line usage error")
    elif exit_code == 5:
        print("📝 No tests were collected")
    
    return exit_code

def check_dependencies():
    """Check for required dependencies"""
    required_deps = [
        'torch',
        'torchaudio', 
        'librosa',
        'numpy',
        'scipy',
        'soundfile',
    ]
    
    optional_deps = [
        'crepe',
        'onnx',
        'onnxruntime',
        'matplotlib',
        'seaborn',
    ]
    
    missing = []
    
    print("Required dependencies:")
    for dep in required_deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep}")
            missing.append(dep)
    
    print("\nOptional dependencies:")
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  - {dep} (optional)")
    
    return missing

def run_specific_test_suite(suite_name):
    """Run a specific test suite"""
    
    test_suites = {
        'unit': 'tests/unit/',
        'integration': 'tests/integration/',
        'audio': 'tests/unit/test_audio_preprocessor.py',
        'pitch': 'tests/unit/test_pitch_extractor.py',
        'models': 'tests/unit/test_models.py',
        'end2end': 'tests/integration/test_end_to_end.py',
    }
    
    if suite_name not in test_suites:
        print(f"Unknown test suite: {suite_name}")
        print(f"Available suites: {', '.join(test_suites.keys())}")
        return 1
    
    test_path = test_suites[suite_name]
    
    print(f"Running {suite_name} tests: {test_path}")
    
    pytest_args = [
        test_path,
        "-v",
        "--tb=short",
        "-ra"
    ]
    
    return pytest.main(pytest_args)

def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality"""
    
    print("Running quick smoke test...")
    print("-" * 40)
    
    try:
        # Test basic imports
        print("Testing imports...")
        
        import numpy as np
        print("  ✓ numpy")
        
        import torch
        print("  ✓ torch")
        
        import librosa
        print("  ✓ librosa")
        
        # Test basic audio processing
        print("\nTesting basic functionality...")
        
        # Generate test audio
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        test_audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        print("  ✓ Generated test audio")
        
        # Test torch functionality
        tensor = torch.from_numpy(test_audio)
        assert tensor.shape[0] == len(test_audio)
        print("  ✓ PyTorch tensor creation")
        
        # Test librosa functionality
        stft = librosa.stft(test_audio)
        assert stft.shape[0] > 0
        print("  ✓ Librosa STFT")
        
        print()
        print("✅ Smoke test passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        return 1

def benchmark_performance():
    """Run performance benchmarks"""
    
    print("Running performance benchmarks...")
    print("-" * 40)
    
    try:
        import torch
        import time
        import numpy as np
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        # Benchmark tensor operations
        print("\nTensor operations:")
        
        sizes = [1000, 10000, 100000]
        for size in sizes:
            # Create random tensors
            a = torch.randn(size, size).to(device)
            b = torch.randn(size, size).to(device)
            
            # Warm up
            for _ in range(5):
                _ = torch.mm(a, b)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(10):
                _ = torch.mm(a, b)
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / 10
            print(f"  Matrix multiply {size}x{size}: {avg_time*1000:.2f}ms")
        
        # Benchmark audio processing
        print("\nAudio processing:")
        
        sample_rate = 44100
        durations = [1.0, 5.0, 10.0]
        
        for duration in durations:
            # Generate test audio
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
            
            # Benchmark STFT
            start_time = time.perf_counter()
            stft = librosa.stft(audio)
            end_time = time.perf_counter()
            
            stft_time = end_time - start_time
            print(f"  STFT {duration}s audio: {stft_time*1000:.2f}ms")
        
        print("\n✅ Performance benchmarks completed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1

def main():
    """Main test runner"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "smoke":
            return run_quick_smoke_test()
        elif command == "benchmark":
            return benchmark_performance()
        elif command in ["unit", "integration", "audio", "pitch", "models", "end2end"]:
            return run_specific_test_suite(command)
        elif command == "help":
            print_help()
            return 0
        else:
            print(f"Unknown command: {command}")
            print_help()
            return 1
    else:
        # Run full test suite
        return run_tests()

def print_help():
    """Print help information"""
    print("AutoTune ML Trainer Test Runner")
    print("Created by Sergie Code")
    print()
    print("Usage:")
    print("  python run_tests.py [command]")
    print()
    print("Commands:")
    print("  (no command)  Run full test suite")
    print("  smoke        Run quick smoke test")
    print("  benchmark    Run performance benchmarks")
    print("  unit         Run unit tests only")
    print("  integration  Run integration tests only")
    print("  audio        Run audio preprocessing tests")
    print("  pitch        Run pitch extraction tests")
    print("  models       Run model tests")
    print("  end2end      Run end-to-end tests")
    print("  help         Show this help")
    print()
    print("Examples:")
    print("  python run_tests.py smoke")
    print("  python run_tests.py unit")
    print("  python run_tests.py benchmark")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
