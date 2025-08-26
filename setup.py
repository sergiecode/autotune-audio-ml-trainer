"""
Setup configuration for AutoTune Audio ML Trainer
"""

from setuptools import setup, find_packages

setup(
    name="autotune-audio-ml-trainer",
    version="1.0.0",
    author="Sergie Code",
    author_email="sergiecode@example.com",
    description="ML framework for intelligent audio pitch correction and timing adjustment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SergieCodes/autotune-audio-ml-trainer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchaudio>=0.12.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0",
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
        ],
        "export": [
            "onnx>=1.12.0",
            "onnxruntime>=1.12.0",
        ],
        "pitch": [
            "crepe>=0.0.12",
        ],
        "audio": [
            "resampy>=0.3.0",
            "pydub>=0.25.0",
        ],
        "quality": [
            "pesq>=0.0.3",
            "pystoi>=0.3.3",
        ]
    },
    entry_points={
        "console_scripts": [
            "autotune-create-dataset=scripts.create_dataset:main",
            "autotune-train=scripts.train_pitch_model:main",
            "autotune-export=scripts.export_for_cpp:main",
            "autotune-test=scripts.test_installation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
