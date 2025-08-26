"""
Installation Test Script for AutoTune ML Trainer

This script verifies that all dependencies are correctly installed
and the environment is properly configured.
"""

import sys
import importlib
import platform
import subprocess
from pathlib import Path

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    """Print script header."""
    print(f"{Colors.BLUE}{Colors.BOLD}")
    print("=" * 60)
    print("  AutoTune ML Trainer - Installation Test")
    print("  Created by Sergie Code")
    print("=" * 60)
    print(f"{Colors.END}")

def print_success(message):
    """Print success message in green."""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_error(message):
    """Print error message in red."""
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def print_warning(message):
    """Print warning message in yellow."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")

def print_info(message):
    """Print info message in blue."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")

def check_python_version():
    """Check Python version compatibility."""
    print(f"\n{Colors.BOLD}Checking Python Version...{Colors.END}")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print_info(f"Python version: {version_str}")
    print_info(f"Platform: {platform.platform()}")
    
    if version.major == 3 and version.minor >= 8:
        print_success("Python version is compatible (3.8+)")
        return True
    else:
        print_error(f"Python 3.8+ required, found {version_str}")
        return False

def check_package(package_name, import_name=None, version_attr=None):
    """Check if a package is installed and optionally get version."""
    if import_name is None:
        import_name = package_name
        
    try:
        module = importlib.import_module(import_name)
        
        # Try to get version
        version = "unknown"
        if version_attr:
            version = getattr(module, version_attr, "unknown")
        elif hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version
            
        print_success(f"{package_name} - version {version}")
        return True, version
        
    except ImportError:
        print_error(f"{package_name} - not installed")
        return False, None

def check_core_dependencies():
    """Check core machine learning dependencies."""
    print(f"\n{Colors.BOLD}Checking Core Dependencies...{Colors.END}")
    
    packages = [
        ("PyTorch", "torch"),
        ("TorchAudio", "torchaudio"),
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("Pandas", "pandas"),
    ]
    
    all_installed = True
    for package_name, import_name in packages:
        installed, version = check_package(package_name, import_name)
        if not installed:
            all_installed = False
            
    return all_installed

def check_audio_dependencies():
    """Check audio processing dependencies."""
    print(f"\n{Colors.BOLD}Checking Audio Processing Dependencies...{Colors.END}")
    
    packages = [
        ("Librosa", "librosa"),
        ("SoundFile", "soundfile"),
        ("Resampy", "resampy"),
        ("PyDub", "pydub"),
    ]
    
    all_installed = True
    for package_name, import_name in packages:
        installed, version = check_package(package_name, import_name)
        if not installed:
            all_installed = False
            
    return all_installed

def check_optional_dependencies():
    """Check optional dependencies."""
    print(f"\n{Colors.BOLD}Checking Optional Dependencies...{Colors.END}")
    
    packages = [
        ("CREPE", "crepe"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("TensorBoard", "tensorboard"),
        ("ONNX", "onnx"),
        ("ONNX Runtime", "onnxruntime"),
        ("Jupyter", "jupyter"),
        ("TQDM", "tqdm"),
        ("PyYAML", "yaml"),
    ]
    
    installed_count = 0
    for package_name, import_name in packages:
        installed, version = check_package(package_name, import_name)
        if installed:
            installed_count += 1
            
    print_info(f"Optional packages installed: {installed_count}/{len(packages)}")
    return installed_count

def test_torch_functionality():
    """Test PyTorch functionality."""
    print(f"\n{Colors.BOLD}Testing PyTorch Functionality...{Colors.END}")
    
    try:
        import torch
        
        # Test basic tensor operations
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)
        z = torch.mm(x, y)
        print_success("Basic tensor operations work")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print_success(f"CUDA available - {torch.cuda.device_count()} device(s)")
            print_info(f"CUDA version: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                print_info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print_warning("CUDA not available - will use CPU")
            
        # Test neural network creation
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        print_success("Neural network creation works")
        
        return True
        
    except Exception as e:
        print_error(f"PyTorch functionality test failed: {e}")
        return False

def test_audio_functionality():
    """Test audio processing functionality."""
    print(f"\n{Colors.BOLD}Testing Audio Processing Functionality...{Colors.END}")
    
    try:
        import numpy as np
        import librosa
        
        # Generate test audio
        duration = 1.0  # seconds
        sample_rate = 44100
        frequency = 440.0  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        print_success("Test audio generation works")
        
        # Test audio analysis
        stft = librosa.stft(audio)
        print_success("STFT computation works")
        
        # Test pitch detection
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
        print_success("Basic pitch detection works")
        
        # Test feature extraction
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate)
        print_success("MFCC feature extraction works")
        
        return True
        
    except Exception as e:
        print_error(f"Audio processing test failed: {e}")
        return False

def test_crepe_functionality():
    """Test CREPE pitch detection if available."""
    print(f"\n{Colors.BOLD}Testing CREPE Functionality...{Colors.END}")
    
    try:
        import crepe
        import numpy as np
        
        # Generate test audio (16kHz for CREPE)
        duration = 1.0
        sample_rate = 16000
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Test CREPE pitch detection
        time, frequency, confidence, activation = crepe.predict(
            audio, 
            sr=sample_rate, 
            model_capacity='tiny'  # Use smallest model for testing
        )
        
        print_success("CREPE pitch detection works")
        print_info(f"Detected frequency: {np.median(frequency[confidence > 0.5]):.1f} Hz")
        
        return True
        
    except ImportError:
        print_warning("CREPE not available - install with: pip install crepe")
        return False
    except Exception as e:
        print_error(f"CREPE test failed: {e}")
        return False

def test_model_export():
    """Test model export functionality."""
    print(f"\n{Colors.BOLD}Testing Model Export Functionality...{Colors.END}")
    
    try:
        import torch
        import onnx
        import onnxruntime
        
        # Create simple test model
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
                
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        dummy_input = torch.randn(1, 10)
        
        # Test ONNX export
        torch.onnx.export(
            model,
            dummy_input,
            "test_model.onnx",
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output']
        )
        
        print_success("ONNX export works")
        
        # Test ONNX runtime
        ort_session = onnxruntime.InferenceSession("test_model.onnx")
        print_success("ONNX runtime works")
        
        # Cleanup
        Path("test_model.onnx").unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print_error(f"Model export test failed: {e}")
        return False

def check_directory_structure():
    """Check if project directory structure exists."""
    print(f"\n{Colors.BOLD}Checking Directory Structure...{Colors.END}")
    
    required_dirs = [
        "src",
        "src/data",
        "src/models", 
        "src/training",
        "src/export",
        "scripts",
        "notebooks",
        "data",
        "data/raw",
        "data/processed",
        "data/datasets",
        "models",
        "models/checkpoints",
        "models/exported"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print_success(f"Directory exists: {dir_path}")
        else:
            print_warning(f"Directory missing: {dir_path}")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print_info("Missing directories can be created automatically during usage")
        
    return len(missing_dirs) == 0

def print_summary(results):
    """Print test summary."""
    print(f"\n{Colors.BOLD}Test Summary{Colors.END}")
    print("=" * 40)
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("All tests passed! Environment is ready for ML training.")
    else:
        print_warning("Some tests failed. Check the error messages above.")
        print_info("You may still be able to use parts of the framework.")

def print_next_steps():
    """Print next steps for the user."""
    print(f"\n{Colors.BOLD}Next Steps{Colors.END}")
    print("=" * 40)
    
    print("1. Place audio files in data/raw/")
    print("2. Create training dataset:")
    print("   python scripts/create_dataset.py --input data/raw --output data/datasets/my_dataset")
    print("3. Start Jupyter Lab for interactive development:")
    print("   jupyter lab")
    print("4. Train your first model:")
    print("   python scripts/train_pitch_model.py")
    print(f"\n{Colors.BLUE}Happy training! 🎵{Colors.END}")

def main():
    """Main test function."""
    print_header()
    
    # Run all tests
    results = {}
    
    results["Python Version"] = check_python_version()
    results["Core Dependencies"] = check_core_dependencies()
    results["Audio Dependencies"] = check_audio_dependencies()
    results["PyTorch Functionality"] = test_torch_functionality()
    results["Audio Processing"] = test_audio_functionality()
    results["CREPE (Optional)"] = test_crepe_functionality()
    results["Model Export"] = test_model_export()
    results["Directory Structure"] = check_directory_structure()
    
    # Check optional dependencies (don't include in pass/fail)
    check_optional_dependencies()
    
    # Print summary
    print_summary(results)
    print_next_steps()
    
    # Return exit code
    if all(results.values()):
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
