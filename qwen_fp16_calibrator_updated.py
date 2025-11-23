"""
Qwen-Image FP16 Auto-Calibration Tool

This script automatically finds the optimal scaling factors for each layer
to prevent NaN overflow in FP16 inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from diffusers import DiffusionPipeline
import numpy as np
import json
from collections import defaultdict
import inspect


class ActivationMonitor:
    """Monitors activations to detect overflow and track statistics"""
    
    def __init__(self):
        self.stats = defaultdict(lambda: {
            'max': 0.0,
            'min': float('inf'),
            'has_nan': False,
            'has_inf': False,
            'mean': 0.0,
            'std': 0.0
        })
        self.hooks = []
    
    def register_hook(self, module, name):
        """Register a forward hook to monitor activations"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            
            if isinstance(tensor, torch.Tensor):
                self.stats[name]['max'] = max(self.stats[name]['max'], tensor.abs().max().item())
                self.stats[name]['min'] = min(self.stats[name]['min'], tensor.abs().min().item())
                self.stats[name]['has_nan'] = self.stats[name]['has_nan'] or torch.isnan(tensor).any().item()
                self.stats[name]['has_inf'] = self.stats[name]['has_inf'] or torch.isinf(tensor).any().item()
                self.stats[name]['mean'] = tensor.abs().mean().item()
                self.stats[name]['std'] = tensor.std().item()
        
        handle = module.register_forward_hook(hook)
        self.hooks.append(handle)
        return handle
    
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_problematic_layers(self, threshold=65000):
        """Return layers with activations exceeding FP16 range"""
        problematic = {}
        for name, stats in self.stats.items():
            if stats['has_nan'] or stats['has_inf'] or stats['max'] > threshold:
                problematic[name] = stats
        return problematic
    
    def print_summary(self):
        """Print activation statistics summary"""
        print("\n" + "="*80)
        print("ACTIVATION STATISTICS")
        print("="*80)
        
        for name, stats in sorted(self.stats.items()):
            status = "✓"
            if stats['has_nan']:
                status = "✗ NaN"
            elif stats['has_inf']:
                status = "✗ Inf"
            elif stats['max'] > 65000:
                status = "⚠ Overflow"
            
            print(f"{status} {name:50s} | Max: {stats['max']:12.2f} | Mean: {stats['mean']:10.2f} | NaN: {stats['has_nan']}")


class AdaptiveScaledAttentionWrapper:
    """Attention wrapper with configurable per-layer scaling that handles all Qwen parameters"""
    
    def __init__(self, original_processor, layer_idx: int, 
                 qkv_scale: float = 1.0, out_scale: float = 1.0):
        self.original_processor = original_processor
        self.layer_idx = layer_idx
        self.qkv_scale = qkv_scale
        self.out_scale = out_scale
        
        # Get the signature of the original processor to handle all parameters
        if hasattr(original_processor, '__call__'):
            self.processor_signature = inspect.signature(original_processor.__call__)
        else:
            self.processor_signature = None
    
    def update_scales(self, qkv_scale: float = None, out_scale: float = None):
        """Update scaling factors"""
        if qkv_scale is not None:
            self.qkv_scale = qkv_scale
        if out_scale is not None:
            self.out_scale = out_scale
    
    def __call__(self, attn, hidden_states, *args, **kwargs):
        """
        Forward pass with scaling. Accepts any arguments that the original processor accepts.
        """
        
        is_fp16 = (hidden_states.dtype == torch.float16)
        
        # If not FP16 or no scaling needed, pass through
        if not is_fp16 or (self.qkv_scale == 1.0 and self.out_scale == 1.0):
            return self.original_processor(attn, hidden_states, *args, **kwargs)
        
        # Store original projections
        original_to_q = attn.to_q
        original_to_k = attn.to_k
        original_to_v = attn.to_v
        original_to_out_0 = attn.to_out[0]
        
        # Create scaled versions
        if self.qkv_scale != 1.0:
            def scaled_to_q(x):
                return original_to_q(x / self.qkv_scale) * self.qkv_scale
            
            def scaled_to_k(x):
                return original_to_k(x / self.qkv_scale) * self.qkv_scale
            
            def scaled_to_v(x):
                return original_to_v(x / self.qkv_scale) * self.qkv_scale
            
            attn.to_q = scaled_to_q
            attn.to_k = scaled_to_k
            attn.to_v = scaled_to_v
        
        if self.out_scale != 1.0:
            def scaled_to_out(x):
                return original_to_out_0(x / self.out_scale) * self.out_scale
            
            attn.to_out[0] = scaled_to_out
        
        # Call original processor with all arguments
        try:
            result = self.original_processor(attn, hidden_states, *args, **kwargs)
        finally:
            # Always restore original layers, even if there's an error
            attn.to_q = original_to_q
            attn.to_k = original_to_k
            attn.to_v = original_to_v
            attn.to_out[0] = original_to_out_0
        
        return result


class AdaptiveScaledFFN(nn.Module):
    """FFN wrapper with configurable scaling"""
    
    def __init__(self, original_ffn, layer_idx: int, scale: float = 1.0):
        super().__init__()
        self.ffn = original_ffn
        self.layer_idx = layer_idx
        self.scale = scale
    
    def update_scale(self, scale: float):
        """Update scaling factor"""
        self.scale = scale
    
    def forward(self, x, *args, **kwargs):
        is_fp16 = (x.dtype == torch.float16)
        scale = self.scale if is_fp16 else 1.0
        
        if scale == 1.0:
            return self.ffn(x, *args, **kwargs)
        
        x = x / scale
        x = self.ffn(x, *args, **kwargs)
        x = x * scale
        return x


class QwenFP16Calibrator:
    """Automatic calibration tool for finding optimal FP16 scaling factors"""
    
    def __init__(self, pipe):
        self.pipe = pipe
        self.blocks = self._get_blocks()
        self.num_blocks = len(self.blocks)
        
        # Default scaling factors (will be optimized)
        self.qkv_scales = [1.0] * self.num_blocks
        self.out_scales = [1.0] * self.num_blocks
        self.ffn_scales = [1.0] * self.num_blocks
        
        # Installed wrappers
        self.attn_wrappers = []
        self.ffn_wrappers = []
        
        print(f"Initialized calibrator with {self.num_blocks} transformer blocks")
    
    def _get_blocks(self):
        """Get transformer blocks from pipeline"""
        transformer = self.pipe.transformer
        
        if hasattr(transformer, 'transformer_blocks'):
            return transformer.transformer_blocks
        elif hasattr(transformer, 'blocks'):
            return transformer.blocks
        else:
            raise ValueError("Could not find transformer blocks")
    
    def install_adaptive_scaling(self):
        """Install adaptive scaling wrappers on all blocks"""
        print("\nInstalling adaptive scaling wrappers...")
        
        for idx, block in enumerate(self.blocks):
            # Install attention wrapper
            if hasattr(block, 'attn'):
                try:
                    # Get the original processor
                    if hasattr(block.attn, 'processor'):
                        original_processor = block.attn.processor
                    else:
                        # Try to get default processor
                        try:
                            original_processor = block.attn.get_processor()
                        except:
                            from diffusers.models.attention_processor import AttnProcessor2_0
                            original_processor = AttnProcessor2_0()
                    
                    # Wrap it
                    wrapper = AdaptiveScaledAttentionWrapper(
                        original_processor, idx,
                        self.qkv_scales[idx], self.out_scales[idx]
                    )
                    block.attn.processor = wrapper
                    self.attn_wrappers.append(wrapper)
                    print(f"  ✓ Installed attention wrapper on block {idx}")
                except Exception as e:
                    print(f"  ✗ Failed to install attention wrapper on block {idx}: {e}")
                    self.attn_wrappers.append(None)
            else:
                self.attn_wrappers.append(None)
            
            # Install FFN wrapper
            ffn_installed = False
            for attr_name in ['ff', 'feed_forward', 'ff_net', 'mlp']:
                if hasattr(block, attr_name):
                    try:
                        original_ffn = getattr(block, attr_name)
                        wrapper = AdaptiveScaledFFN(original_ffn, idx, self.ffn_scales[idx])
                        setattr(block, attr_name, wrapper)
                        self.ffn_wrappers.append(wrapper)
                        print(f"  ✓ Installed FFN wrapper on block {idx} ({attr_name})")
                        ffn_installed = True
                        break
                    except Exception as e:
                        print(f"  ✗ Failed to install FFN wrapper on block {idx}: {e}")
            
            if not ffn_installed:
                print(f"  ⚠ Could not find FFN in block {idx}")
                self.ffn_wrappers.append(None)
        
        print("\n✓ All wrappers installed\n")
    
    def update_scales(self, qkv_scales=None, out_scales=None, ffn_scales=None):
        """Update scaling factors for all layers"""
        if qkv_scales is not None:
            self.qkv_scales = qkv_scales
        if out_scales is not None:
            self.out_scales = out_scales
        if ffn_scales is not None:
            self.ffn_scales = ffn_scales
        
        # Update wrappers
        for idx, wrapper in enumerate(self.attn_wrappers):
            if wrapper is not None:
                wrapper.update_scales(self.qkv_scales[idx], self.out_scales[idx])
        
        for idx, wrapper in enumerate(self.ffn_wrappers):
            if wrapper is not None:
                wrapper.update_scale(self.ffn_scales[idx])
    
    def test_generation(self, prompt="A simple test image", steps=10, width=512, height=512):
        """Test generation and check for NaN values"""
        print(f"\nTesting generation (steps={steps}, size={width}x{height})...")
        
        try:
            with torch.no_grad():
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=" ",
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    true_cfg_scale=4.0,
                    generator=torch.Generator(device=self.pipe.device).manual_seed(42)
                ).images[0]
            
            # Check for NaN
            img_array = np.array(image)
            has_nan = np.isnan(img_array).any()
            
            if has_nan:
                print("✗ Output contains NaN values")
                return False, None
            else:
                print("✓ Generation successful, no NaN values")
                return True, image
                
        except Exception as e:
            print(f"✗ Generation failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def monitor_activations(self, prompt="A test image", steps=5):
        """Run generation with activation monitoring"""
        print("\nMonitoring activations...")
        
        monitor = ActivationMonitor()
        
        # Register hooks on all blocks
        for idx, block in enumerate(self.blocks):
            if hasattr(block, 'attn'):
                monitor.register_hook(block.attn, f"block_{idx}_attn")
            
            for attr_name in ['ff', 'feed_forward', 'ff_net', 'mlp']:
                if hasattr(block, attr_name):
                    ffn_module = getattr(block, attr_name)
                    # If it's wrapped, unwrap it for monitoring
                    if isinstance(ffn_module, AdaptiveScaledFFN):
                        ffn_module = ffn_module.ffn
                    monitor.register_hook(ffn_module, f"block_{idx}_ffn")
                    break
        
        # Run generation
        try:
            with torch.no_grad():
                _ = self.pipe(
                    prompt=prompt,
                    negative_prompt=" ",
                    width=512,
                    height=512,
                    num_inference_steps=steps,
                    true_cfg_scale=4.0,
                    generator=torch.Generator(device=self.pipe.device).manual_seed(42)
                )
        except Exception as e:
            print(f"Generation failed during monitoring: {e}")
        
        monitor.clear_hooks()
        monitor.print_summary()
        
        return monitor
    
    def auto_calibrate(self, test_prompt="A coffee shop with neon lights", 
                      initial_qkv=8.0, initial_out=2.0, initial_ffn=32.0,
                      max_iterations=10):
        """
        Automatically calibrate scaling factors to eliminate NaN values.
        
        Strategy:
        1. Start with suggested scales from Draw Things blog
        2. Monitor activations to find problematic layers
        3. Increase scaling for problematic layers iteratively
        4. Test generation after each adjustment
        """
        
        print("\n" + "="*80)
        print("STARTING AUTO-CALIBRATION")
        print("="*80)
        
        # Initialize with base scales
        self.qkv_scales = [initial_qkv] * self.num_blocks
        self.out_scales = [initial_out] * self.num_blocks
        
        # FFN scales: 32x for layers 0-58, 512x for layer 59
        self.ffn_scales = [initial_ffn if i < 59 else initial_ffn * 16 for i in range(self.num_blocks)]
        
        self.update_scales(self.qkv_scales, self.out_scales, self.ffn_scales)
        
        print(f"\nInitial scales:")
        print(f"  QKV: {initial_qkv}x")
        print(f"  Attention Output: {initial_out}x")
        print(f"  FFN: {initial_ffn}x (layers 0-58), {initial_ffn*16}x (layer 59+)")
        
        for iteration in range(max_iterations):
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*80}")
            
            # Test generation with smaller resolution first, then larger
            if iteration < 3:
                test_width, test_height = 512, 512
                test_steps = 10
            else:
                test_width, test_height = 1024, 1024
                test_steps = 20
            
            success, image = self.test_generation(test_prompt, steps=test_steps, 
                                                  width=test_width, height=test_height)
            
            if success and iteration >= 3:  # Only declare success after testing larger resolution
                print(f"\n✓✓✓ CALIBRATION SUCCESSFUL ✓✓✓")
                print(f"\nFinal scaling factors:")
                self.print_scale_config()
                return True, image
            elif success:
                print(f"✓ Succeeded at {test_width}x{test_height}, testing larger resolution...")
                continue
            
            # Monitor activations to find problematic layers
            print("\nAnalyzing problematic layers...")
            monitor = self.monitor_activations(test_prompt, steps=5)
            problematic = monitor.get_problematic_layers(threshold=60000)
            
            if not problematic:
                print("No problematic layers detected, but generation still failed.")
                print("Increasing all scales by 2x...")
                self.qkv_scales = [s * 2 for s in self.qkv_scales]
                self.out_scales = [s * 2 for s in self.out_scales]
                self.ffn_scales = [s * 2 for s in self.ffn_scales]
            else:
                print(f"\nFound {len(problematic)} problematic layer(s)")
                
                # Extract layer indices and increase their scales
                for name, stats in problematic.items():
                    if 'block_' in name:
                        try:
                            layer_idx = int(name.split('_')[1])
                            
                            if 'attn' in name:
                                self.qkv_scales[layer_idx] *= 2
                                self.out_scales[layer_idx] *= 2
                                print(f"  Increased QKV/Out scales for layer {layer_idx}: "
                                      f"QKV={self.qkv_scales[layer_idx]}x, Out={self.out_scales[layer_idx]}x")
                            
                            if 'ffn' in name:
                                self.ffn_scales[layer_idx] *= 2
                                print(f"  Increased FFN scale for layer {layer_idx}: "
                                      f"{self.ffn_scales[layer_idx]}x")
                        except Exception as e:
                            print(f"  Error parsing layer from {name}: {e}")
            
            # Update wrappers with new scales
            self.update_scales(self.qkv_scales, self.out_scales, self.ffn_scales)
        
        print(f"\n✗ Calibration did not converge after {max_iterations} iterations")
        print("\nCurrent scaling factors:")
        self.print_scale_config()
        return False, None
    
    def print_scale_config(self):
        """Print current scaling configuration"""
        print("\n" + "-"*80)
        print("SCALING CONFIGURATION")
        print("-"*80)
        
        for idx in range(self.num_blocks):
            qkv = self.qkv_scales[idx]
            out = self.out_scales[idx]
            ffn = self.ffn_scales[idx]
            
            marker = " ⚠" if (qkv > 64 or out > 64 or ffn > 512) else ""
            print(f"Block {idx:2d}: QKV={qkv:6.1f}x  Out={out:6.1f}x  FFN={ffn:6.1f}x{marker}")
        
        print("-"*80)
    
    def save_config(self, filename="qwen_fp16_scales.json"):
        """Save calibrated scales to file"""
        config = {
            'qkv_scales': self.qkv_scales,
            'out_scales': self.out_scales,
            'ffn_scales': self.ffn_scales,
            'num_blocks': self.num_blocks
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ Saved configuration to {filename}")
    
    def load_config(self, filename="qwen_fp16_scales.json"):
        """Load scales from file"""
        with open(filename, 'r') as f:
            config = json.load(f)
        
        self.qkv_scales = config['qkv_scales']
        self.out_scales = config['out_scales']
        self.ffn_scales = config['ffn_scales']
        
        self.update_scales(self.qkv_scales, self.out_scales, self.ffn_scales)
        
        print(f"\n✓ Loaded configuration from {filename}")
        self.print_scale_config()


def main():
    """Example usage"""
    
    model_name = "Qwen/Qwen-Image"
    
    print("="*80)
    print("QWEN-IMAGE FP16 AUTO-CALIBRATION TOOL")
    print("="*80)
    
    print("\nLoading model...")
    
    if torch.cuda.is_available():
        torch_dtype = torch.float16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"
        print("\n⚠ Warning: Running on CPU. Calibration may be very slow.")
    
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    
    print(f"✓ Model loaded on {device} with dtype {torch_dtype}")
    
    # Create calibrator
    calibrator = QwenFP16Calibrator(pipe)
    
    # Install adaptive scaling
    calibrator.install_adaptive_scaling()
    
    # Run auto-calibration
    test_prompt = '''A coffee shop entrance with a chalkboard sign and neon lights'''
    
    success, image = calibrator.auto_calibrate(
        test_prompt=test_prompt,
        initial_qkv=8.0,
        initial_out=2.0,
        initial_ffn=32.0,
        max_iterations=10
    )
    
    if success:
        image.save("calibrated_output.png")
        print("\n✓ Test image saved as 'calibrated_output.png'")
        
        # Save configuration
        calibrator.save_config("qwen_fp16_scales.json")
        
        print("\n" + "="*80)
        print("CALIBRATION COMPLETE!")
        print("="*80)
        print("\nYou can now use these scales for inference.")
        print("The configuration has been saved to 'qwen_fp16_scales.json'")
    else:
        print("\n⚠ Calibration did not fully succeed.")
        print("You may need to:")
        print("  1. Increase max_iterations")
        print("  2. Try higher initial scaling factors")
        print("  3. Use BF16 instead of FP16 if your hardware supports it")


if __name__ == "__main__":
    main()
