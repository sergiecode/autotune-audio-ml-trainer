"""
AutoTune Audio ML Trainer

A comprehensive Python framework for training neural networks that perform 
intelligent pitch correction and natural timing adjustments.

Created by Sergie Code - AI Tools for Musicians
"""

__version__ = "1.0.0"
__author__ = "Sergie Code"
__email__ = "sergiecode@example.com"
__description__ = "ML framework for intelligent audio pitch correction and timing adjustment"

# Core imports
from .data import *
from .models import *
from .training import *
from .export import *

# Configuration
SAMPLE_RATE = 44100
BUFFER_SIZE = 512
CHANNELS = 1
BIT_DEPTH = 32

# Musical scales (compatible with C++ engine)
SCALES = {
    'MAJOR': [0, 2, 4, 5, 7, 9, 11],
    'MINOR': [0, 2, 3, 5, 7, 8, 10],
    'BLUES': [0, 3, 5, 6, 7, 10],
    'PENTATONIC': [0, 2, 4, 7, 9],
    'DORIAN': [0, 2, 3, 5, 7, 9, 10],
    'MIXOLYDIAN': [0, 2, 4, 5, 7, 9, 10],
    'CHROMATIC': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}

# Pitch detection parameters
MIN_FREQUENCY = 80.0   # Hz (lowest guitar string)
MAX_FREQUENCY = 2000.0 # Hz (highest vocals)
CONFIDENCE_THRESHOLD = 0.7  # minimum confidence for valid detection
