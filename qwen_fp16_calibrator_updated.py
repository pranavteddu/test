"""
Qwen-Image FP16 Deep Layer Scanner

Comprehensively scans and scales ALL linear layers in the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from diffusers import DiffusionPipeline
import numpy as np
import json
from collections import defaultdict


class ScaledLinear(nn.Module):
    """Linear layer with FP16 scaling"""
    
    def __init__(self, original_linear: nn.Linear, input_scale: float = 1.0):
        super().__init__()
        self.linear = original_linear
        self.input_scale = input_scale
        
        # Copy attributes
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
    
    def forward(self, x):
        is_fp16 = (x.dtype == torch.float16)
        
        if not is_fp16 or self.input_scale == 1.0:
            return self.linear(x)
        
        # Scale down, compute, scale back up
        x = x / self.input_scale
        x = self.linear(x)
        x = x * self.input_scale
        
        return x
    
    def update_scale(self, input_scale: float):
        """Update scaling factor"""
        self.input_scale = input_scale


def explore_module_structure(module, prefix="", max_depth=10):
    """Recursively explore module structure to find all submodules"""
    if max_depth == 0:
        return
    
    print(f"{prefix}{module.__class__.__name__}")
    
    for name, child in module.named_children():
        print(f"{prefix}  .{name}: {child.__class__.__name__}", end="")
        if isinstance(child, nn.Linear):
            print(f" [Linear: {child.in_features} -> {child.out_features}]")
        else:
            print()
        
        if len(list(child.children())) > 0:
            explore_module_structure(child, prefix + "    ", max_depth - 1)


class QwenFP16Calibrator:
    """Calibrator with deep module exploration"""
    
    def __init__(self, pipe):
        self.pipe = pipe
        self.blocks = self._get_blocks()
        self.num_blocks = len(self.blocks)
        
        # Track all scaled layers
        self.scaled_layers = []
        
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
    
    def explore_block_structure(self, block_idx=0):
        """Explore and print the structure of a transformer block"""
        print(f"\n{'='*80}")
        print(f"EXPLORING BLOCK {block_idx} STRUCTURE")
        print(f"{'='*80}\n")
        
        if block_idx < len(self.blocks):
            explore_module_structure(self.blocks[block_idx])
        else:
            print(f"Block {block_idx} doesn't exist")
    
    def find_all_linear_layers(self, module, parent_name=""):
        """Recursively find all Linear layers in a module"""
        linear_layers = []
        
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            if isinstance(child, nn.Linear):
                linear_layers.append((full_name, child))
            else:
                # Recurse into submodules
                linear_layers.extend(self.find_all_linear_layers(child, full_name))
        
        return linear_layers
    
    def install_scaling_layers(self):
        """Find and replace ALL linear layers in attention and FFN"""
        print("\nScanning and replacing linear layers...")
        
        for idx, block in enumerate(self.blocks):
            print(f"\n{'='*60}")
            print(f"Block {idx}")
            print(f"{'='*60}")
            
            # Find all linear layers in this block
            all_linears = self.find_all_linear_layers(block, f"block_{idx}")
            
            print(f"Found {len(all_linears)} linear layers:")
            for name, layer in all_linears:
                print(f"  {name}: {layer.in_features} -> {layer.out_features}")
            
            # Categorize and replace layers
            for full_name, original_layer in all_linears:
                # Determine which type of layer this is
                scale = 1.0
                layer_type = "unknown"
                
                # Check if it's in attention
                if 'attn' in full_name.lower():
                    if any(x in full_name for x in ['to_q', 'to_k', 'to_v', 'q_proj', 'k_proj', 'v_proj']):
                        scale = self.qkv_scales[idx]
                        layer_type = "attn_qkv"
                    elif any(x in full_name for x in ['to_out', 'out_proj', 'o_proj']):
                        scale = self.out_scales[idx]
                        layer_type = "attn_out"
                    else:
                        scale = self.qkv_scales[idx]  # Default for attention
                        layer_type = "attn_other"
                
                # Check if it's in FFN/MLP
                elif any(x in full_name.lower() for x in ['ff', 'ffn', 'mlp', 'feed_forward']):
                    scale = self.ffn_scales[idx]
                    layer_type = "ffn"
                
                # Replace the layer if it needs scaling
                if scale != 1.0:
                    scaled_layer = ScaledLinear(original_layer, scale)
                    
                    # Navigate to parent and replace
                    parts = full_name.split('.')
                    parent = block
                    for part in parts[1:-1]:  # Skip 'block_X' and last part
                        if part.isdigit():
                            parent = parent[int(part)]
                        else:
                            parent = getattr(parent, part)
                    
                    # Set the scaled layer
                    last_part = parts[-1]
                    if last_part.isdigit():
                        parent[int(last_part)] = scaled_layer
                    else:
                        setattr(parent, last_part, scaled_layer)
                    
                    self.scaled_layers.append((idx, full_name, layer_type, scaled_layer))
                    print(f"  ✓ Replaced {full_name} ({layer_type}, scale={scale}x)")
        
        print(f"\n{'='*60}")
        print(f"✓ Replaced {len(self.scaled_layers)} linear layers total")
        print(f"{'='*60}")
        
        # Print summary
        attn_qkv = sum(1 for _, _, t, _ in self.scaled_layers if t == "attn_qkv")
        attn_out = sum(1 for _, _, t, _ in self.scaled_layers if t == "attn_out")
        ffn = sum(1 for _, _, t, _ in self.scaled_layers if t == "ffn")
        
        print(f"\nSummary:")
        print(f"  Attention QKV layers: {attn_qkv}")
        print(f"  Attention Output layers: {attn_out}")
        print(f"  FFN layers: {ffn}")
        print(f"  Other: {len(self.scaled_layers) - attn_qkv - attn_out - ffn}")
    
    def update_scales(self, qkv_scales=None, out_scales=None, ffn_scales=None):
        """Update scaling factors"""
        if qkv_scales is not None:
            self.qkv_scales = qkv_scales
        if out_scales is not None:
            self.out_scales = out_scales
        if ffn_scales is not None:
            self.ffn_scales = ffn_scales
        
        # Update all scaled layers
        for block_idx, name, layer_type, scaled_layer in self.scaled_layers:
            if layer_type in ["attn_qkv", "attn_other"]:
                scaled_layer.update_scale(self.qkv_scales[block_idx])
            elif layer_type == "attn_out":
                scaled_layer.update_scale(self.out_scales[block_idx])
            elif layer_type == "ffn":
                scaled_layer.update_scale(self.ffn_scales[block_idx])
    
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
            
            # Also check if image is blank (all same value)
            img_std = img_array.std()
            is_blank = img_std < 1.0
            
            if has_nan:
                print("✗ Output contains NaN values")
                return False, None
            elif is_blank:
                print(f"✗ Output is blank (std={img_std:.4f})")
                return False, None
            else:
                print(f"✓ Generation successful! (std={img_std:.2f})")
                return True, image
                
        except Exception as e:
            print(f"✗ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
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
            
            # Test with increasing resolution
            if iteration < 2:
                w, h, s = 512, 512, 10
            elif iteration < 5:
                w, h, s = 768, 768, 15
            else:
                w, h, s = 1024, 1024, 20
            
            success, image = self.test_generation(test_prompt, steps=s, width=w, height=h)
            
            if success and iteration >= 5:
                print(f"\n✓✓✓ CALIBRATION SUCCESSFUL ✓✓✓")
                self.print_scale_config()
                return True, image
            elif success:
                print(f"✓ Success at {w}x{h}, testing larger resolution...")
                continue
            
            # If failed, increase scales
            print("\nIncreasing scales...")
            
            # Gradually increase all scales
            self.qkv_scales = [s * 1.5 for s in self.qkv_scales]
            self.out_scales = [s * 1.5 for s in self.out_scales]
            self.ffn_scales = [s * 1.5 for s in self.ffn_scales]
            
            self.update_scales(self.qkv_scales, self.out_scales, self.ffn_scales)
            
            print(f"New scales: QKV={self.qkv_scales[0]:.1f}x, Out={self.out_scales[0]:.1f}x, FFN={self.ffn_scales[0]:.1f}x")
        
        print(f"\n✗ Did not converge after {max_iterations} iterations")
        self.print_scale_config()
        return False, None
    
    def print_scale_config(self):
        print("\n" + "-"*80)
        print("SCALING CONFIGURATION")
        print("-"*80)
        for idx in range(min(self.num_blocks, 10)):  # Show first 10
            print(f"Block {idx:2d}: QKV={self.qkv_scales[idx]:6.1f}x  "
                  f"Out={self.out_scales[idx]:6.1f}x  FFN={self.ffn_scales[idx]:6.1f}x")
        if self.num_blocks > 10:
            print("...")
            idx = self.num_blocks - 1
            print(f"Block {idx:2d}: QKV={self.qkv_scales[idx]:6.1f}x  "
                  f"Out={self.out_scales[idx]:6.1f}x  FFN={self.ffn_scales[idx]:6.1f}x")
        print("-"*80)
    
    def save_config(self, filename="qwen_fp16_scales.json"):
        config = {
            'qkv_scales': self.qkv_scales,
            'out_scales': self.out_scales,
            'ffn_scales': self.ffn_scales,
            'num_scaled_layers': len(self.scaled_layers)
        }
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n✓ Saved to {filename}")


def main():
    model_name = "Qwen/Qwen-Image"
    
    print("="*80)
    print("QWEN-IMAGE FP16 DEEP LAYER SCANNER")
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
    
    # First, explore one block to see structure
    print("\n" + "="*80)
    print("STEP 1: Exploring model structure...")
    print("="*80)
    calibrator.explore_block_structure(0)
    
    # Ask user if they want to continue
    print("\n" + "="*80)
    print("STEP 2: Installing scaled layers...")
    print("="*80)
    
    calibrator.install_scaling_layers()
    
    # Run calibration
    print("\n" + "="*80)
    print("STEP 3: Running auto-calibration...")
    print("="*80)
    
    success, image = calibrator.auto_calibrate(
        test_prompt="A beautiful coffee shop with warm lighting and neon signs",
        initial_qkv=8.0,
        initial_out=2.0,
        initial_ffn=32.0,
        max_iterations=15
    )
    
    if success:
        image.save("calibrated_output.png")
        calibrator.save_config("qwen_fp16_scales.json")
        print("\n✓ COMPLETE! Image saved as 'calibrated_output.png'")
    else:
        print("\n✗ Calibration did not succeed")
        print("Suggestions:")
        print("  1. Try higher initial scales (qkv=16, out=4, ffn=64)")
        print("  2. Check if your GPU has enough VRAM")
        print("  3. Try reducing resolution further")


if __name__ == "__main__":
    main()
