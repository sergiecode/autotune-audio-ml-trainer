"""
Neural Network Models for AutoTune ML Trainer

This module contains neural network architectures for pitch correction,
timing adjustment, and enhanced pitch detection.
"""

from .pitch_correction_net import PitchCorrectionNet

__all__ = [
    'PitchCorrectionNet',
]
