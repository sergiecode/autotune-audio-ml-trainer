"""
Model Export Script for C++ Integration

This script exports trained PyTorch models to ONNX format for deployment
in the real-time C++ AutoTune engine.
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.pitch_correction_net import PitchCorrectionNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(model, export_path, input_size=512):
    """Export PyTorch model to ONNX format."""
    
    model.eval()
    
    # Create dummy inputs for tracing
    dummy_audio = torch.randn(1, input_size)
    dummy_pitch = torch.tensor([[440.0]])  # A4 note
    dummy_strength = torch.tensor([[0.5]])  # 50% correction
    
    logger.info(f"Exporting model to ONNX: {export_path}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_audio, dummy_pitch, dummy_strength),
        export_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['audio_input', 'target_pitch', 'correction_strength'],
        output_names=['corrected_audio', 'confidence'],
        dynamic_axes={
            'audio_input': {0: 'batch_size'},
            'target_pitch': {0: 'batch_size'},
            'correction_strength': {0: 'batch_size'},
            'corrected_audio': {0: 'batch_size'},
            'confidence': {0: 'batch_size'}
        }
    )
    
    # Verify export
    try:
        import onnx
        onnx_model = onnx.load(export_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model validation passed")
        
        # Test with ONNX Runtime
        try:
            import onnxruntime as ort
            ort_session = ort.InferenceSession(str(export_path))
            
            # Test inference
            test_inputs = {
                'audio_input': dummy_audio.numpy(),
                'target_pitch': dummy_pitch.numpy(),
                'correction_strength': dummy_strength.numpy()
            }
            
            ort_outputs = ort_session.run(None, test_inputs)
            logger.info(f"✅ ONNX Runtime test passed")
            logger.info(f"Output shapes: {[output.shape for output in ort_outputs]}")
            
        except ImportError:
            logger.warning("ONNX Runtime not available for testing")
            
    except ImportError:
        logger.warning("ONNX not available for verification")
        
    logger.info(f"Model exported successfully to: {export_path}")


def main():
    parser = argparse.ArgumentParser(description="Export AutoTune ML model to ONNX")
    
    parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--output', help='Output ONNX file path')
    parser.add_argument('--buffer_size', type=int, default=512, help='Audio buffer size')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of hidden layers')
    
    args = parser.parse_args()
    
    # Create model with same architecture
    model = PitchCorrectionNet(
        input_size=args.buffer_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )
    
    # Load trained weights
    checkpoint = torch.load(args.model, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded model state dict")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        model_path = Path(args.model)
        output_path = model_path.parent / f"{model_path.stem}.onnx"
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export model
    export_to_onnx(model, output_path, args.buffer_size)
    
    print(f"\n🎵 Export completed!")
    print(f"ONNX model: {output_path}")
    print(f"Ready for C++ integration!")


if __name__ == "__main__":
    main()
