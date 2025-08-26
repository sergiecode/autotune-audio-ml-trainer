"""
Pitch Correction Neural Network

This module implements a neural network for real-time pitch correction
that maintains natural audio characteristics while correcting pitch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class PitchCorrectionNet(nn.Module):
    """
    Real-time pitch correction neural network.
    
    This model takes audio input and target pitch information to produce
    pitch-corrected audio while preserving natural characteristics.
    """
    
    def __init__(self,
                 input_size: int = 512,
                 hidden_size: int = 256,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        """
        Initialize the PitchCorrectionNet.
        
        Args:
            input_size: Size of input audio buffer
            hidden_size: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super(PitchCorrectionNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Input processing layers
        self.input_norm = nn.BatchNorm1d(1)
        self.input_conv = nn.Conv1d(1, 64, kernel_size=15, padding=7)
        
        # Pitch embedding layer
        self.pitch_embedding = nn.Linear(2, 32)  # target_pitch + correction_strength
        
        # Main processing network
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_size + 32, hidden_size))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            
        self.main_network = nn.Sequential(*layers)
        
        # Output layers
        self.output_projection = nn.Linear(hidden_size, input_size)
        self.confidence_head = nn.Linear(hidden_size, 1)
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, 
                audio_buffer: torch.Tensor,
                target_pitch: torch.Tensor,
                correction_strength: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the pitch correction network.
        
        Args:
            audio_buffer: Input audio [batch_size, buffer_size]
            target_pitch: Target frequency in Hz [batch_size, 1]
            correction_strength: Correction amount 0-1 [batch_size, 1]
            
        Returns:
            Tuple of (corrected_audio, confidence_score)
        """
        batch_size = audio_buffer.size(0)
        
        # Normalize audio input
        audio_normalized = self.input_norm(audio_buffer.unsqueeze(1)).squeeze(1)
        
        # Process audio with convolution
        audio_conv = self.input_conv(audio_normalized.unsqueeze(1))  # [batch, 64, seq_len]
        audio_conv = audio_conv.mean(dim=1)  # Average across channels [batch, seq_len]
        
        # Resize to input_size if needed
        if audio_conv.size(1) != self.input_size:
            audio_features = F.adaptive_avg_pool1d(audio_conv.unsqueeze(1), self.input_size).squeeze(1)
        else:
            audio_features = audio_conv
        
        # Normalize target pitch (log scale)
        target_pitch_norm = torch.log(target_pitch + 1e-8) / 10.0  # Rough normalization
        
        # Create pitch embedding
        pitch_input = torch.cat([target_pitch_norm, correction_strength], dim=1)
        pitch_features = self.pitch_embedding(pitch_input)
        
        # Combine audio and pitch features
        combined_input = torch.cat([audio_features, pitch_features], dim=1)
        
        # Main processing
        features = self.main_network(combined_input)
        
        # Generate outputs
        pitch_correction = self.output_projection(features)
        confidence = torch.sigmoid(self.confidence_head(features))
        
        # Apply residual connection with learnable weight
        residual_weight = torch.sigmoid(self.residual_weight)
        corrected_audio = (1 - residual_weight) * audio_buffer + residual_weight * pitch_correction
        
        # Apply correction strength
        final_audio = (1 - correction_strength) * audio_buffer + \
                     correction_strength * corrected_audio
        
        return final_audio, confidence
        
    def get_model_info(self) -> Dict:
        """
        Get model architecture information.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'PitchCorrectionNet',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }
        
    def estimate_latency(self, sample_rate: int = 44100) -> float:
        """
        Estimate processing latency for real-time use.
        
        Args:
            sample_rate: Audio sample rate
            
        Returns:
            Estimated latency in milliseconds
        """
        # This is a rough estimate based on model complexity
        # Actual latency depends on hardware and optimization
        complexity_factor = self.hidden_size * self.num_layers / 1000000
        base_latency = 2.0  # Base latency in ms
        
        estimated_latency = base_latency + complexity_factor * 10
        return min(estimated_latency, 50.0)  # Cap at 50ms


class MultiScalePitchCorrectionNet(nn.Module):
    """
    Multi-scale pitch correction network that processes audio at different
    temporal resolutions for improved accuracy and naturalness.
    """
    
    def __init__(self,
                 input_size: int = 512,
                 scales: list = [1, 2, 4],
                 hidden_size: int = 256):
        """
        Initialize the multi-scale network.
        
        Args:
            input_size: Size of input audio buffer
            scales: List of temporal scales to process
            hidden_size: Hidden layer dimension
        """
        super(MultiScalePitchCorrectionNet, self).__init__()
        
        self.input_size = input_size
        self.scales = scales
        self.hidden_size = hidden_size
        
        # Create separate networks for each scale
        self.scale_networks = nn.ModuleList([
            PitchCorrectionNet(
                input_size=input_size // scale,
                hidden_size=hidden_size // len(scales),
                num_layers=3
            ) for scale in scales
        ])
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(input_size * len(scales), hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, input_size)
        )
        
        # Confidence fusion
        self.confidence_fusion = nn.Sequential(
            nn.Linear(len(scales), hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                audio_buffer: torch.Tensor,
                target_pitch: torch.Tensor,
                correction_strength: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-scale processing.
        
        Args:
            audio_buffer: Input audio [batch_size, buffer_size]
            target_pitch: Target frequency in Hz [batch_size, 1]
            correction_strength: Correction amount 0-1 [batch_size, 1]
            
        Returns:
            Tuple of (corrected_audio, confidence_score)
        """
        scale_outputs = []
        scale_confidences = []
        
        # Process at each scale
        for i, (scale, network) in enumerate(zip(self.scales, self.scale_networks)):
            # Downsample audio
            if scale > 1:
                downsampled = F.avg_pool1d(
                    audio_buffer.unsqueeze(1), 
                    kernel_size=scale, 
                    stride=scale
                ).squeeze(1)
            else:
                downsampled = audio_buffer
                
            # Process with scale-specific network
            scale_output, scale_confidence = network(
                downsampled, target_pitch, correction_strength
            )
            
            # Upsample back to original size
            if scale > 1:
                scale_output = F.interpolate(
                    scale_output.unsqueeze(1),
                    size=self.input_size,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
                
            scale_outputs.append(scale_output)
            scale_confidences.append(scale_confidence)
            
        # Fuse outputs
        combined_output = torch.cat(scale_outputs, dim=1)
        fused_audio = self.fusion_network(combined_output)
        
        # Fuse confidences
        combined_confidences = torch.cat(scale_confidences, dim=1)
        fused_confidence = self.confidence_fusion(combined_confidences)
        
        return fused_audio, fused_confidence


class ConvolutionalPitchCorrector(nn.Module):
    """
    Convolutional approach to pitch correction using 1D convolutions
    for better temporal modeling.
    """
    
    def __init__(self,
                 input_size: int = 512,
                 num_channels: int = 64,
                 kernel_sizes: list = [3, 5, 7, 15]):
        """
        Initialize the convolutional pitch corrector.
        
        Args:
            input_size: Size of input audio buffer
            num_channels: Number of convolutional channels
            kernel_sizes: List of kernel sizes for multi-scale convolution
        """
        super(ConvolutionalPitchCorrector, self).__init__()
        
        self.input_size = input_size
        self.num_channels = num_channels
        
        # Multi-scale convolution layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, num_channels, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Pitch conditioning
        self.pitch_proj = nn.Linear(2, num_channels)
        
        # Processing layers
        total_channels = num_channels * len(kernel_sizes) + num_channels
        
        self.processing = nn.Sequential(
            nn.Conv1d(total_channels, num_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_channels * 2),
            nn.GELU(),
            nn.Conv1d(num_channels * 2, num_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_channels),
            nn.GELU(),
            nn.Conv1d(num_channels, 1, kernel_size=3, padding=1)
        )
        
        # Confidence estimation
        self.confidence_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(total_channels, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                audio_buffer: torch.Tensor,
                target_pitch: torch.Tensor,
                correction_strength: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with convolutional processing.
        
        Args:
            audio_buffer: Input audio [batch_size, buffer_size]
            target_pitch: Target frequency in Hz [batch_size, 1]
            correction_strength: Correction amount 0-1 [batch_size, 1]
            
        Returns:
            Tuple of (corrected_audio, confidence_score)
        """
        batch_size = audio_buffer.size(0)
        
        # Multi-scale convolution
        audio_input = audio_buffer.unsqueeze(1)  # Add channel dimension
        conv_features = []
        
        for conv_layer in self.conv_layers:
            conv_out = F.gelu(conv_layer(audio_input))
            conv_features.append(conv_out)
            
        # Combine convolution features
        combined_conv = torch.cat(conv_features, dim=1)
        
        # Add pitch conditioning
        target_pitch_norm = torch.log(target_pitch + 1e-8) / 10.0
        pitch_input = torch.cat([target_pitch_norm, correction_strength], dim=1)
        pitch_features = self.pitch_proj(pitch_input)
        
        # Broadcast pitch features to match audio length
        pitch_broadcast = pitch_features.unsqueeze(-1).expand(-1, -1, self.input_size)
        
        # Combine all features
        all_features = torch.cat([combined_conv, pitch_broadcast], dim=1)
        
        # Process to get correction
        pitch_correction = self.processing(all_features).squeeze(1)
        
        # Estimate confidence
        confidence = self.confidence_net(all_features)
        
        # Apply correction with strength
        corrected_audio = (1 - correction_strength.unsqueeze(-1)) * audio_buffer + \
                         correction_strength.unsqueeze(-1) * pitch_correction
        
        return corrected_audio, confidence


def create_pitch_correction_model(model_type: str = 'standard', **kwargs) -> nn.Module:
    """
    Factory function to create different types of pitch correction models.
    
    Args:
        model_type: Type of model ('standard', 'multiscale', 'convolutional')
        **kwargs: Additional arguments for model creation
        
    Returns:
        Pitch correction model
    """
    if model_type == 'standard':
        return PitchCorrectionNet(**kwargs)
    elif model_type == 'multiscale':
        return MultiScalePitchCorrectionNet(**kwargs)
    elif model_type == 'convolutional':
        return ConvolutionalPitchCorrector(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    batch_size = 4
    buffer_size = 512
    
    audio_buffer = torch.randn(batch_size, buffer_size).to(device)
    target_pitch = torch.rand(batch_size, 1).to(device) * 1000 + 100  # 100-1100 Hz
    correction_strength = torch.rand(batch_size, 1).to(device)
    
    # Test standard model
    model = PitchCorrectionNet().to(device)
    corrected_audio, confidence = model(audio_buffer, target_pitch, correction_strength)
    
    print(f"Input shape: {audio_buffer.shape}")
    print(f"Output shape: {corrected_audio.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Model info: {model.get_model_info()}")
    print(f"Estimated latency: {model.estimate_latency():.2f} ms")
