# AutoTune Audio ML Trainer 🎵

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive Python framework for training neural networks that perform intelligent pitch correction and natural timing adjustments. This project creates AI models that seamlessly integrate with real-time audio processing engines.

**Created by Sergie Code** - AI Tools for Musicians 🎸

## 🎯 Project Overview

This ML training framework is designed to work in conjunction with the [autotune-real-time-audio-tuner](../autotune-real-time-audio-tuner) C++ engine, providing:

- **Intelligent Pitch Correction**: Neural networks that learn natural pitch correction patterns
- **Timing Adjustment**: AI models for musical timing and rhythm correction
- **Enhanced Pitch Detection**: ML-improved pitch detection beyond traditional autocorrelation
- **Real-time Integration**: Models optimized for low-latency real-time audio processing

### 🔗 Integration with C++ Engine

This Python framework trains models that are exported and deployed in the real-time C++ AutoTune engine:

```
Python ML Trainer → Model Export (ONNX) → C++ Real-time Engine → Live Audio Processing
```

## 🚀 Quick Start

### Prerequisites

- **Windows 10/11** with PowerShell
- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **Git** for version control
- **Audio files** for training data (WAV, FLAC, MP3)

### 1. Clone and Setup

```powershell
# Clone the repository
git clone https://github.com/SergieCodes/autotune-audio-ml-trainer.git
cd autotune-audio-ml-trainer

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If execution policy prevents activation, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```powershell
# Test the installation
python scripts/test_installation.py

# Start Jupyter Lab for interactive development
jupyter lab
```

### 3. Prepare Your Data

```powershell
# Place your audio files in the data/raw directory
# Supported formats: WAV, FLAC, MP3, M4A
# Recommended: 44.1kHz sample rate, mono or stereo

# Create training dataset
python scripts/create_dataset.py --input data/raw --output data/datasets/my_dataset
```

## 📁 Project Structure

```
autotune-audio-ml-trainer/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                 # Package installation
├── .gitignore               # Git ignore rules
│
├── data/                    # Training data
│   ├── raw/                 # Original audio files
│   ├── processed/           # Preprocessed audio data
│   └── datasets/            # Prepared training datasets
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── data/                # Data processing modules
│   │   ├── dataset_creator.py
│   │   ├── audio_preprocessor.py
│   │   ├── pitch_extractor.py
│   │   └── augmentation.py
│   ├── models/              # Neural network architectures
│   │   ├── pitch_correction_net.py
│   │   ├── timing_adjustment_net.py
│   │   ├── enhanced_pitch_detector.py
│   │   └── model_utils.py
│   ├── training/            # Training pipeline
│   │   ├── trainer.py
│   │   ├── loss_functions.py
│   │   ├── metrics.py
│   │   └── callbacks.py
│   └── export/              # Model export utilities
│       ├── onnx_exporter.py
│       ├── cpp_bridge.py
│       └── model_optimizer.py
│
├── scripts/                 # Utility scripts
│   ├── test_installation.py
│   ├── create_dataset.py
│   ├── train_pitch_model.py
│   ├── train_timing_model.py
│   ├── evaluate_model.py
│   └── export_for_cpp.py
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_pitch_detection_analysis.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_export_integration.ipynb
│
└── models/                  # Trained models
    ├── checkpoints/         # Training checkpoints
    └── exported/            # Models ready for C++ integration
```

## 🧠 Neural Network Models

### 1. Pitch Correction Network

Corrects pitch in real-time while maintaining natural audio characteristics:

```python
# Train pitch correction model
python scripts/train_pitch_model.py \
    --dataset data/datasets/vocal_dataset \
    --epochs 100 \
    --batch_size 32
```

**Features:**
- Real-time processing (< 5ms latency)
- Maintains formant characteristics
- Musical scale awareness
- Adjustable correction strength

### 2. Timing Adjustment Network

Intelligently adjusts timing and rhythm:

```python
# Train timing adjustment model
python scripts/train_timing_model.py \
    --dataset data/datasets/rhythm_dataset \
    --epochs 80 \
    --context_window 2048
```

**Features:**
- Musical context awareness
- BPM detection and correction
- Natural rhythm preservation
- Multi-instrument support

### 3. Enhanced Pitch Detector

ML-improved pitch detection with higher accuracy:

```python
# Train enhanced pitch detector
python scripts/train_pitch_detector.py \
    --dataset data/datasets/pitch_dataset \
    --target_accuracy 0.95
```

**Features:**
- 95%+ pitch detection accuracy
- Robust to noise and harmonics
- Real-time compatible
- Confidence scoring

## 📊 Training Pipeline

### Basic Training Workflow

1. **Data Preparation**
   ```powershell
   python scripts/create_dataset.py --input data/raw --output data/datasets/my_dataset
   ```

2. **Model Training**
   ```powershell
   python scripts/train_pitch_model.py --config configs/pitch_correction.yaml
   ```

3. **Model Evaluation**
   ```powershell
   python scripts/evaluate_model.py --model models/checkpoints/pitch_model_best.pth
   ```

4. **Export for C++**
   ```powershell
   python scripts/export_for_cpp.py --model models/checkpoints/pitch_model_best.pth --output models/exported/
   ```

### Advanced Configuration

Training parameters can be customized using YAML configuration files:

```yaml
# configs/pitch_correction.yaml
model:
  name: "PitchCorrectionNet"
  input_size: 512
  hidden_size: 256
  num_layers: 4

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 10

data:
  sample_rate: 44100
  buffer_size: 512
  augmentation: true
```

## 🔄 Integration with C++ Engine

### Model Export Process

The trained models are exported in formats compatible with the C++ real-time engine:

```python
# Export to ONNX format
from src.export.onnx_exporter import export_to_onnx

model = torch.load('models/checkpoints/pitch_model_best.pth')
export_to_onnx(model, 'models/exported/pitch_model.onnx')
```

### C++ Integration

The exported models integrate with the C++ engine through these interfaces:

```cpp
// C++ engine integration example
#include "autotune_engine.h"
#include "ml_pitch_corrector.h"

// Load ML model
MLPitchCorrector ml_corrector;
ml_corrector.load_model("models/exported/pitch_model.onnx");

// Use in real-time processing
ProcessingResult result = ml_corrector.process_with_ml(input_frame, params);
```

### Performance Requirements

Models are optimized for real-time constraints:
- **Latency**: < 5ms processing time
- **CPU Usage**: < 25% of single core
- **Memory**: < 100MB footprint
- **Accuracy**: > 95% pitch detection accuracy

## 📓 Jupyter Notebooks

Interactive notebooks for experimentation and learning:

1. **Data Exploration** (`01_data_exploration.ipynb`)
   - Audio data analysis
   - Pitch distribution visualization
   - Dataset statistics

2. **Pitch Detection Analysis** (`02_pitch_detection_analysis.ipynb`)
   - Compare pitch detection methods
   - CREPE vs. traditional algorithms
   - Accuracy benchmarking

3. **Model Training** (`03_model_training.ipynb`)
   - Interactive model training
   - Hyperparameter experimentation
   - Training visualization

4. **Model Evaluation** (`04_model_evaluation.ipynb`)
   - Performance metrics
   - Audio quality assessment
   - A/B testing

5. **Export Integration** (`05_export_integration.ipynb`)
   - Model export workflow
   - C++ integration testing
   - Performance optimization

### Starting Jupyter Lab

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start Jupyter Lab
jupyter lab

# Open your browser to: http://localhost:8888
```

## 🛠️ Development Tools

### Testing Installation

```powershell
python scripts/test_installation.py
```

This script verifies:
- ✅ Python version compatibility
- ✅ All dependencies installed
- ✅ Audio processing capabilities
- ✅ GPU availability (if applicable)
- ✅ ONNX export functionality

### Dataset Creation

```powershell
# Create dataset from audio files
python scripts/create_dataset.py \
    --input data/raw \
    --output data/datasets/my_dataset \
    --sample_rate 44100 \
    --segment_length 4.0 \
    --overlap 0.5
```

### Model Training Scripts

```powershell
# Train pitch correction model
python scripts/train_pitch_model.py \
    --dataset data/datasets/my_dataset \
    --config configs/pitch_correction.yaml \
    --output models/checkpoints/

# Train with GPU (if available)
python scripts/train_pitch_model.py \
    --dataset data/datasets/my_dataset \
    --device cuda \
    --batch_size 64
```

## 🎵 Musical Context Integration

### Supported Musical Scales

The framework includes built-in support for various musical scales:

```python
SCALES = {
    'MAJOR': [0, 2, 4, 5, 7, 9, 11],
    'MINOR': [0, 2, 3, 5, 7, 8, 10],
    'BLUES': [0, 3, 5, 6, 7, 10],
    'PENTATONIC': [0, 2, 4, 7, 9],
    'DORIAN': [0, 2, 3, 5, 7, 9, 10],
    'MIXOLYDIAN': [0, 2, 4, 5, 7, 9, 10],
    'CHROMATIC': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}
```

### Instrument-Specific Training

Train models for specific instruments:

```powershell
# Train for vocals
python scripts/train_pitch_model.py --instrument vocal --dataset data/datasets/vocal_dataset

# Train for guitar
python scripts/train_pitch_model.py --instrument guitar --dataset data/datasets/guitar_dataset

# Train for violin
python scripts/train_pitch_model.py --instrument violin --dataset data/datasets/violin_dataset
```

## 📈 Performance Monitoring

### Training Monitoring with TensorBoard

```powershell
# Start TensorBoard
tensorboard --logdir runs/

# Open browser to: http://localhost:6006
```

### Weights & Biases Integration

```python
# Initialize wandb in training script
import wandb

wandb.init(project="autotune-ml-trainer", name="pitch-correction-v1")
```

## 🔧 Advanced Features

### Data Augmentation

Automatic data augmentation for robust training:

```python
AUGMENTATIONS = [
    'pitch_shift',        # ±50 cents variations
    'time_stretch',       # ±10% tempo changes
    'noise_addition',     # Realistic background noise
    'reverb_simulation',  # Room acoustics
    'formant_shifting',   # Voice characteristics
    'dynamics_variation'  # Volume/compression changes
]
```

### Custom Loss Functions

Multi-component loss functions for audio quality:

```python
def combined_loss(predicted_audio, target_audio, predicted_pitch, target_pitch):
    # Audio reconstruction + pitch accuracy + perceptual quality + phase coherence
    return 0.4 * audio_loss + 0.3 * pitch_loss + 0.2 * perceptual_loss + 0.1 * phase_loss
```

### Model Optimization

Optimize models for real-time deployment:

```powershell
# Optimize model for inference
python scripts/optimize_model.py \
    --input models/checkpoints/pitch_model_best.pth \
    --output models/exported/pitch_model_optimized.onnx \
    --target_latency 5.0
```

## 🚀 Extending the Framework

### Adding New Model Architectures

1. Create new model in `src/models/`:
   ```python
   class CustomPitchModel(nn.Module):
       def __init__(self):
           super().__init__()
           # Your architecture here
   ```

2. Add training script in `scripts/`:
   ```python
   # train_custom_model.py
   from src.models.custom_pitch_model import CustomPitchModel
   ```

3. Update configuration files in `configs/`

### Custom Audio Preprocessing

Add custom preprocessing in `src/data/audio_preprocessor.py`:

```python
class CustomPreprocessor:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
    
    def process(self, audio):
        # Your custom preprocessing
        return processed_audio
```

### Integration with External Tools

The framework can integrate with:
- **DAWs**: Export audio from Logic Pro, Pro Tools, Ableton Live
- **Audio Libraries**: Import from Freesound, Musicnet, NSynth
- **Cloud Storage**: Train on Google Drive, Dropbox datasets
- **Version Control**: Track model versions with DVC

## 🎓 Learning Resources

### Tutorials for Beginners

1. **Audio Processing Basics**
   - Understanding sample rates and bit depth
   - Working with audio files in Python
   - Basic pitch detection concepts

2. **Machine Learning for Audio**
   - Neural network fundamentals
   - Audio-specific architectures
   - Training best practices

3. **Real-time Audio Processing**
   - Latency considerations
   - Buffer management
   - Optimization techniques

### YouTube Channel Integration

Perfect for **Sergie Code's YouTube tutorials**:
- Step-by-step model training videos
- Real-time integration demonstrations
- Advanced AI music production techniques
- C++ and Python integration tutorials

## 🤝 Contributing

This project welcomes contributions from the music and AI community:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Areas for Contribution

- New neural network architectures
- Additional audio augmentation techniques
- Performance optimizations
- Mobile/embedded device support
- New musical instrument support
- Better evaluation metrics

## 🐛 Troubleshooting

### Common Issues

**1. Virtual Environment Activation Issues**
```powershell
# If PowerShell execution policy prevents activation:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**2. CUDA/GPU Issues**
```powershell
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

**3. Audio Library Issues**
```powershell
# Install system audio libraries if needed
# For Windows: Usually handled by pip packages
pip install soundfile librosa --upgrade
```

**4. Memory Issues During Training**
```powershell
# Reduce batch size in training
python scripts/train_pitch_model.py --batch_size 16

# Use gradient accumulation for larger effective batch size
python scripts/train_pitch_model.py --batch_size 8 --accumulate_grad_batches 4
```

### Getting Help

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community support and questions
- **YouTube**: Video tutorials and demonstrations
- **Documentation**: Detailed API reference

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CREPE**: Deep learning pitch detection
- **Librosa**: Audio analysis in Python
- **PyTorch**: Deep learning framework
- **Music Information Retrieval Community**: Research and datasets
- **Open Source Contributors**: Making AI accessible to musicians

## 🔗 Related Projects

- [autotune-real-time-audio-tuner](../autotune-real-time-audio-tuner): C++ real-time engine

## 👨‍💻 About the Creator

**Sergie Code** is a software engineer passionate about creating AI tools for musicians. Through YouTube tutorials and open-source projects, Sergie helps developers build innovative music technology solutions.

- 📸 Instagram: https://www.instagram.com/sergiecode

- 🧑🏼‍💻 LinkedIn: https://www.linkedin.com/in/sergiecode/

- 📽️Youtube: https://www.youtube.com/@SergieCode

- 😺 Github: https://github.com/sergiecode

- 👤 Facebook: https://www.facebook.com/sergiecodeok

- 🎞️ Tiktok: https://www.tiktok.com/@sergiecode

- 🕊️Twitter: https://twitter.com/sergiecode

- 🧵Threads: https://www.threads.net/@sergiecode

---

**Built with ❤️ by [Sergie Code](https://github.com/SergieCode) for the music community**

*Making AI-powered music tools accessible to everyone* 🎵
