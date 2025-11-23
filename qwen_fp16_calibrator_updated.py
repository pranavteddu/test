"""
Qwen-Image FP16 Direct Layer Scaling

This approach directly modifies the Linear layers instead of wrapping processors,
avoiding parameter signature issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from diffusers import DiffusionPipeline
import numpy as np
import json
from collections import defaultdict


class ScaledLinear(nn.Module):
    """
    A Linear layer wrapper that applies input/output scaling for FP16 stability.
    This directly replaces nn.Linear modules.
    """
    
    def __init__(self, original_linear: nn.Linear, input_scale: float = 1.0, output_scale: float = 1.0):
        super().__init__()
        self.linear = original_linear
        self.input_scale = input_scale
        self.output_scale = output_scale
        
        # Copy attributes from original linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
    
    def forward(self, x):
        is_fp16 = (x.dtype == torch.float16)
        
        if not is_fp16 or (self.input_scale == 1.0 and self.output_scale == 1.0):
            return self.linear(x)
        
        # Scale down input
        if self.input_scale != 1.0:
            x = x / self.input_scale
        
        # Forward through linear
        x = self.linear(x)
        
        # Scale output
        if self.output_scale != 1.0:
            x = x * self.output_scale
        
        # If we scaled input, scale output to compensate
        if self.input_scale != 1.0:
            x = x * self.input_scale
        
        if self.output_scale != 1.0:
            x = x / self.output_scale
        
        return x
    
    def update_scales(self, input_scale: float = None, output_scale: float = None):
        """Update scaling factors"""
        if input_scale is not None:
            self.input_scale = input_scale
        if output_scale is not None:
            self.output_scale = output_scale


class ActivationMonitor:
    """Monitors activations to detect overflow"""
    
    def __init__(self):
        self.stats = defaultdict(lambda: {
            'max': 0.0,
            'has_nan': False,
            'has_inf': False,
        })
        self.hooks = []
    
    def register_hook(self, module, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            
            if isinstance(tensor, torch.Tensor):
                self.stats[name]['max'] = max(self.stats[name]['max'], tensor.abs().max().item())
                self.stats[name]['has_nan'] = self.stats[name]['has_nan'] or torch.isnan(tensor).any().item()
                self.stats[name]['has_inf'] = self.stats[name]['has_inf'] or torch.isinf(tensor).any().item()
        
        handle = module.register_forward_hook(hook)
        self.hooks.append(handle)
        return handle
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_problematic_layers(self, threshold=65000):
        problematic = {}
        for name, stats in self.stats.items():
            if stats['has_nan'] or stats['has_inf'] or stats['max'] > threshold:
                problematic[name] = stats
        return problematic
    
    def print_summary(self):
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
            
            print(f"{status} {name:60s} | Max: {stats['max']:12.2f}")


class QwenFP16Calibrator:
    """Calibrator that directly modifies Linear layers"""
    
    def __init__(self, pipe):
        self.pipe = pipe
        self.blocks = self._get_blocks()
        self.num_blocks = len(self.blocks)
        
        # Track which layers have been modified
        self.modified_layers = []
        
        # Scaling factors per block
        self.qkv_scales = [1.0] * self.num_blocks
        self.out_scales = [1.0] * self.num_blocks
        self.ffn_scales = [1.0] * self.num_blocks
        
        print(f"Initialized calibrator with {self.num_blocks} transformer blocks")
    
    def _get_blocks(self):
        transformer = self.pipe.transformer
        
        if hasattr(transformer, 'transformer_blocks'):
            return transformer.transformer_blocks
        elif hasattr(transformer, 'blocks'):
            return transformer.blocks
        else:
            raise ValueError("Could not find transformer blocks")
    
    def install_scaling_layers(self):
        """Replace Linear layers with ScaledLinear in attention and FFN"""
        print("\nInstalling scaled linear layers...")
        
        for idx, block in enumerate(self.blocks):
            print(f"\nBlock {idx}:")
            
            # Find and replace attention projection layers
            if hasattr(block, 'attn'):
                attn = block.attn
                
                # Replace to_q, to_k, to_v
                for proj_name in ['to_q', 'to_k', 'to_v']:
                    if hasattr(attn, proj_name):
                        original = getattr(attn, proj_name)
                        if isinstance(original, nn.Linear):
                            scaled = ScaledLinear(original, self.qkv_scales[idx], 1.0)
                            setattr(attn, proj_name, scaled)
                            self.modified_layers.append((idx, f'attn.{proj_name}', scaled))
                            print(f"  ✓ Replaced {proj_name}")
                
                # Replace to_out
                if hasattr(attn, 'to_out'):
                    if isinstance(attn.to_out, nn.ModuleList) or isinstance(attn.to_out, nn.Sequential):
                        if len(attn.to_out) > 0 and isinstance(attn.to_out[0], nn.Linear):
                            original = attn.to_out[0]
                            scaled = ScaledLinear(original, self.out_scales[idx], 1.0)
                            attn.to_out[0] = scaled
                            self.modified_layers.append((idx, 'attn.to_out', scaled))
                            print(f"  ✓ Replaced to_out")
                    elif isinstance(attn.to_out, nn.Linear):
                        original = attn.to_out
                        scaled = ScaledLinear(original, self.out_scales[idx], 1.0)
                        attn.to_out = scaled
                        self.modified_layers.append((idx, 'attn.to_out', scaled))
                        print(f"  ✓ Replaced to_out")
            
            # Find and replace FFN layers
            ffn_found = False
            for attr_name in ['ff', 'feed_forward', 'ff_net', 'mlp']:
                if hasattr(block, attr_name):
                    ffn = getattr(block, attr_name)
                    
                    # Look for linear layers in FFN
                    if hasattr(ffn, 'net') and isinstance(ffn.net, nn.Sequential):
                        for i, layer in enumerate(ffn.net):
                            if isinstance(layer, nn.Linear) and i == 0:  # First linear layer
                                original = layer
                                scaled = ScaledLinear(original, self.ffn_scales[idx], 1.0)
                                ffn.net[i] = scaled
                                self.modified_layers.append((idx, f'{attr_name}.net[{i}]', scaled))
                                print(f"  ✓ Replaced {attr_name}.net[{i}]")
                                ffn_found = True
                                break
                    
                    # Try direct linear layers
                    if not ffn_found:
                        for sub_attr in ['linear1', 'fc1', 'w1', 'c_fc']:
                            if hasattr(ffn, sub_attr):
                                original = getattr(ffn, sub_attr)
                                if isinstance(original, nn.Linear):
                                    scaled = ScaledLinear(original, self.ffn_scales[idx], 1.0)
                                    setattr(ffn, sub_attr, scaled)
                                    self.modified_layers.append((idx, f'{attr_name}.{sub_attr}', scaled))
                                    print(f"  ✓ Replaced {attr_name}.{sub_attr}")
                                    ffn_found = True
                                    break
                    
                    if ffn_found:
                        break
            
            if not ffn_found:
                print(f"  ⚠ Could not find FFN linear layers")
        
        print(f"\n✓ Installed {len(self.modified_layers)} scaled linear layers")
    
    def update_scales(self, qkv_scales=None, out_scales=None, ffn_scales=None):
        """Update scaling factors"""
        if qkv_scales is not None:
            self.qkv_scales = qkv_scales
        if out_scales is not None:
            self.out_scales = out_scales
        if ffn_scales is not None:
            self.ffn_scales = ffn_scales
        
        # Update all modified layers
        for block_idx, layer_name, scaled_layer in self.modified_layers:
            if 'to_q' in layer_name or 'to_k' in layer_name or 'to_v' in layer_name:
                scaled_layer.update_scales(input_scale=self.qkv_scales[block_idx])
            elif 'to_out' in layer_name:
                scaled_layer.update_scales(input_scale=self.out_scales[block_idx])
            elif any(x in layer_name for x in ['ff', 'feed_forward', 'mlp']):
                scaled_layer.update_scales(input_scale=self.ffn_scales[block_idx])
    
    def test_generation(self, prompt="A simple test image", steps=10, width=512, height=512):
        """Test generation"""
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
            
            img_array = np.array(image)
            has_nan = np.isnan(img_array).any()
            
            if has_nan:
                print("✗ Output contains NaN values")
                return False, None
            else:
                print("✓ Generation successful!")
                return True, image
                
        except Exception as e:
            print(f"✗ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def monitor_activations(self, prompt="A test image", steps=5):
        """Monitor activations"""
        print("\nMonitoring activations...")
        
        monitor = ActivationMonitor()
        
        for idx, block in enumerate(self.blocks):
            # Monitor attention output
            if hasattr(block, 'attn'):
                monitor.register_hook(block.attn, f"block_{idx}_attn")
            
            # Monitor FFN output
            for attr_name in ['ff', 'feed_forward', 'ff_net', 'mlp']:
                if hasattr(block, attr_name):
                    monitor.register_hook(getattr(block, attr_name), f"block_{idx}_ffn")
                    break
        
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
            print(f"Monitoring failed: {e}")
        
        monitor.clear_hooks()
        monitor.print_summary()
        
        return monitor
    
    def auto_calibrate(self, test_prompt="A coffee shop", 
                      initial_qkv=8.0, initial_out=2.0, initial_ffn=32.0,
                      max_iterations=10):
        """Auto-calibrate scaling factors"""
        
        print("\n" + "="*80)
        print("AUTO-CALIBRATION")
        print("="*80)
        
        # Initialize scales
        self.qkv_scales = [initial_qkv] * self.num_blocks
        self.out_scales = [initial_out] * self.num_blocks
        self.ffn_scales = [initial_ffn if i < 59 else initial_ffn * 16 for i in range(self.num_blocks)]
        
        self.update_scales(self.qkv_scales, self.out_scales, self.ffn_scales)
        
        print(f"\nInitial scales: QKV={initial_qkv}x, Out={initial_out}x, FFN={initial_ffn}x")
        
        for iteration in range(max_iterations):
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*80}")
            
            # Progressive resolution testing
            if iteration < 3:
                w, h, s = 512, 512, 10
            else:
                w, h, s = 1024, 1024, 20
            
            success, image = self.test_generation(test_prompt, steps=s, width=w, height=h)
            
            if success and iteration >= 3:
                print(f"\n✓✓✓ SUCCESS ✓✓✓")
                self.print_scale_config()
                return True, image
            elif success:
                print(f"✓ Success at {w}x{h}, testing larger...")
                continue
            
            # Analyze failures
            print("\nAnalyzing problematic layers...")
            monitor = self.monitor_activations(test_prompt, steps=5)
            problematic = monitor.get_problematic_layers(threshold=60000)
            
            if not problematic:
                print("No specific problematic layers, increasing all scales by 2x")
                self.qkv_scales = [s * 2 for s in self.qkv_scales]
                self.out_scales = [s * 2 for s in self.out_scales]
                self.ffn_scales = [s * 2 for s in self.ffn_scales]
            else:
                print(f"Found {len(problematic)} problematic layers")
                for name, stats in problematic.items():
                    if 'block_' in name:
                        try:
                            layer_idx = int(name.split('_')[1])
                            if 'attn' in name:
                                self.qkv_scales[layer_idx] *= 2
                                self.out_scales[layer_idx] *= 2
                                print(f"  Increased scales for block {layer_idx} attention")
                            if 'ffn' in name:
                                self.ffn_scales[layer_idx] *= 2
                                print(f"  Increased scales for block {layer_idx} FFN")
                        except:
                            pass
            
            self.update_scales(self.qkv_scales, self.out_scales, self.ffn_scales)
        
        print(f"\n✗ Did not converge after {max_iterations} iterations")
        self.print_scale_config()
        return False, None
    
    def print_scale_config(self):
        print("\n" + "-"*80)
        print("SCALING CONFIGURATION")
        print("-"*80)
        for idx in range(self.num_blocks):
            print(f"Block {idx:2d}: QKV={self.qkv_scales[idx]:6.1f}x  "
                  f"Out={self.out_scales[idx]:6.1f}x  FFN={self.ffn_scales[idx]:6.1f}x")
        print("-"*80)
    
    def save_config(self, filename="qwen_fp16_scales.json"):
        config = {
            'qkv_scales': self.qkv_scales,
            'out_scales': self.out_scales,
            'ffn_scales': self.ffn_scales,
        }
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n✓ Saved to {filename}")


def main():
    model_name = "Qwen/Qwen-Image"
    
    print("="*80)
    print("QWEN-IMAGE FP16 CALIBRATION (Direct Layer Scaling)")
    print("="*80)
    
    print("\nLoading model...")
    
    if torch.cuda.is_available():
        torch_dtype = torch.float16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"
    
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    
    print(f"✓ Loaded on {device}")
    
    calibrator = QwenFP16Calibrator(pipe)
    calibrator.install_scaling_layers()
    
    success, image = calibrator.auto_calibrate(
        test_prompt="A coffee shop with neon lights",
        initial_qkv=8.0,
        initial_out=2.0,
        initial_ffn=32.0,
        max_iterations=10
    )
    
    if success:
        image.save("calibrated_output.png")
        calibrator.save_config("qwen_fp16_scales.json")
        print("\n✓ COMPLETE! Image saved as 'calibrated_output.png'")


if __name__ == "__main__":
    main()
