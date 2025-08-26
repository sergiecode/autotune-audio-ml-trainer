"""
Basic Training Script for AutoTune ML Trainer

This script demonstrates model training with the pitch correction network.
"""

import sys
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.pitch_correction_net import PitchCorrectionNet

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleAudioDataset(Dataset):
    """Simple dataset for demonstration purposes."""
    
    def __init__(self, num_samples=1000, buffer_size=512):
        self.num_samples = num_samples
        self.buffer_size = buffer_size
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Generate synthetic training data
        input_audio = torch.randn(self.buffer_size)
        target_audio = input_audio + 0.1 * torch.randn(self.buffer_size)
        target_pitch = torch.rand(1) * 1000 + 100  # 100-1100 Hz
        correction_strength = torch.rand(1)
        confidence = torch.rand(1) * 0.5 + 0.5
        
        return {
            'input_audio': input_audio,
            'target_audio': target_audio,
            'target_pitch': target_pitch,
            'correction_strength': correction_strength,
            'confidence': confidence
        }


class CombinedLoss(nn.Module):
    """Multi-component loss function."""
    
    def __init__(self, audio_weight=0.6, spectral_weight=0.3, confidence_weight=0.1):
        super().__init__()
        self.audio_weight = audio_weight
        self.spectral_weight = spectral_weight
        self.confidence_weight = confidence_weight
        
    def forward(self, pred_audio, target_audio, pred_conf, target_conf):
        # Audio reconstruction loss
        audio_loss = nn.functional.mse_loss(pred_audio, target_audio)
        
        # Spectral loss
        pred_fft = torch.fft.fft(pred_audio, dim=-1)
        target_fft = torch.fft.fft(target_audio, dim=-1)
        spectral_loss = nn.functional.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
        
        # Confidence loss
        confidence_loss = nn.functional.binary_cross_entropy(pred_conf, target_conf)
        
        total_loss = (self.audio_weight * audio_loss + 
                     self.spectral_weight * spectral_loss + 
                     self.confidence_weight * confidence_loss)
        
        return total_loss, {
            'audio_loss': audio_loss.item(),
            'spectral_loss': spectral_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'total_loss': total_loss.item()
        }


def train_model(args):
    """Main training function."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Create model
    model = PitchCorrectionNet(
        input_size=args.buffer_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)
    
    logger.info(f"Model created: {model.get_model_info()}")
    
    # Create datasets
    train_dataset = SimpleAudioDataset(args.train_samples, args.buffer_size)
    val_dataset = SimpleAudioDataset(args.val_samples, args.buffer_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Setup training
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_audio = batch['input_audio'].to(device)
            target_audio = batch['target_audio'].to(device)
            target_pitch = batch['target_pitch'].to(device)
            correction_strength = batch['correction_strength'].to(device)
            confidence = batch['confidence'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_audio, pred_conf = model(input_audio, target_pitch, correction_strength)
            
            # Compute loss
            loss, loss_components = criterion(pred_audio, target_audio, pred_conf, confidence)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss_components['total_loss'])
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Loss: {loss_components['total_loss']:.4f}")
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_audio = batch['input_audio'].to(device)
                target_audio = batch['target_audio'].to(device)
                target_pitch = batch['target_pitch'].to(device)
                correction_strength = batch['correction_strength'].to(device)
                confidence = batch['confidence'].to(device)
                
                pred_audio, pred_conf = model(input_audio, target_pitch, correction_strength)
                loss, loss_components = criterion(pred_audio, target_audio, pred_conf, confidence)
                val_losses.append(loss_components['total_loss'])
        
        # Update history
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'model_config': {
                    'input_size': args.buffer_size,
                    'hidden_size': args.hidden_size,
                    'num_layers': args.num_layers
                }
            }
            
            checkpoint_path = Path(args.output_dir) / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
    
    # Save final model and history
    final_model_path = Path(args.output_dir) / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    
    history_path = Path(args.output_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train AutoTune ML pitch correction model")
    
    # Model architecture
    parser.add_argument('--buffer_size', type=int, default=512, help='Audio buffer size')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of hidden layers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    # Data parameters
    parser.add_argument('--train_samples', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=1000, help='Number of validation samples')
    
    # Output
    parser.add_argument('--output_dir', default='../models/checkpoints', help='Output directory')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train model
    model, history = train_model(args)
    
    print(f"\nTraining Summary:")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Best val loss: {min(history['val_loss']):.4f}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
