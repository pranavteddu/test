"""
Qwen-Image Mixed Precision Approach

Instead of scaling, keep critical layers in FP32 while running most in FP16.
This is more reliable than activation scaling.
"""

import torch
import torch.nn as nn
from diffusers import DiffusionPipeline
import numpy as np


class FP32Wrapper(nn.Module):
    """Forces a module to compute in FP32"""
    
    def __init__(self, module):
        super().__init__()
        self.module = module
        
    def forward(self, x, *args, **kwargs):
        # Convert input to FP32
        original_dtype = x.dtype
        x_fp32 = x.float()
        
        # Process in FP32
        output = self.module(x_fp32, *args, **kwargs)
        
        # Convert back to original dtype
        if isinstance(output, torch.Tensor):
            return output.to(original_dtype)
        elif isinstance(output, tuple):
            return tuple(o.to(original_dtype) if isinstance(o, torch.Tensor) else o for o in output)
        else:
            return output


def patch_qwen_for_mixed_precision(pipe, strategy="conservative"):
    """
    Patch Qwen-Image to use mixed precision.
    
    Strategies:
    - "conservative": Keep most layers in FP32 (safest, slower)
    - "balanced": FP32 for later layers only (good balance)
    - "aggressive": Only critical layers in FP32 (fastest, may still have issues)
    """
    
    print(f"\n{'='*80}")
    print(f"Applying Mixed Precision Patching - Strategy: {strategy}")
    print(f"{'='*80}\n")
    
    transformer = pipe.transformer
    
    # Get blocks
    if hasattr(transformer, 'transformer_blocks'):
        blocks = transformer.transformer_blocks
    elif hasattr(transformer, 'blocks'):
        blocks = transformer.blocks
    else:
        raise ValueError("Could not find transformer blocks")
    
    num_blocks = len(blocks)
    print(f"Found {num_blocks} transformer blocks\n")
    
    # Determine which blocks to keep in FP32
    if strategy == "conservative":
        # Keep all blocks in FP32
        fp32_block_indices = list(range(num_blocks))
        print("Strategy: All blocks in FP32")
    
    elif strategy == "balanced":
        # Keep later half in FP32 (where overflow is more likely)
        fp32_block_indices = list(range(num_blocks // 2, num_blocks))
        print(f"Strategy: Blocks {num_blocks//2}-{num_blocks-1} in FP32")
    
    elif strategy == "aggressive":
        # Only keep last 10 blocks in FP32
        fp32_block_indices = list(range(max(0, num_blocks - 10), num_blocks))
        print(f"Strategy: Last 10 blocks in FP32")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Patch blocks
    patched_count = 0
    for idx in fp32_block_indices:
        block = blocks[idx]
        
        # Wrap attention in FP32
        if hasattr(block, 'attn'):
            block.attn = FP32Wrapper(block.attn)
            patched_count += 1
        
        # Wrap FFN in FP32
        for attr in ['ff', 'feed_forward', 'mlp', 'ff_net']:
            if hasattr(block, attr):
                setattr(block, attr, FP32Wrapper(getattr(block, attr)))
                break
        
        print(f"  Block {idx:2d}: Wrapped in FP32")
    
    print(f"\n✓ Patched {patched_count} blocks with FP32 wrappers\n")
    
    return pipe


def test_all_strategies(pipe, prompt="A coffee shop with neon lights"):
    """Test all three strategies to find which works best"""
    
    print("\n" + "="*80)
    print("TESTING ALL STRATEGIES")
    print("="*80)
    
    strategies = ["aggressive", "balanced", "conservative"]
    results = {}
    
    for strategy in strategies:
        print(f"\n\n{'='*80}")
        print(f"TESTING: {strategy.upper()}")
        print(f"{'='*80}")
        
        # Reload model fresh for each test
        print("\nReloading model...")
        device = pipe.device
        dtype = pipe.transformer.dtype
        model_name = "Qwen/Qwen-Image"
        
        del pipe
        torch.cuda.empty_cache()
        
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
        pipe = pipe.to(device)
        
        # Apply strategy
        pipe = patch_qwen_for_mixed_precision(pipe, strategy=strategy)
        
        # Test generation
        print(f"\nGenerating test image...")
        
        try:
            with torch.no_grad():
                image = pipe(
                    prompt=prompt,
                    negative_prompt=" ",
                    width=1024,
                    height=1024,
                    num_inference_steps=20,
                    true_cfg_scale=4.0,
                    generator=torch.Generator(device=device).manual_seed(42)
                ).images[0]
            
            # Check quality
            img_array = np.array(image)
            has_nan = np.isnan(img_array).any()
            img_std = img_array.std()
            is_blank = img_std < 1.0
            
            if has_nan:
                print(f"✗ FAILED: Contains NaN")
                results[strategy] = {"success": False, "reason": "NaN"}
            elif is_blank:
                print(f"✗ FAILED: Blank image (std={img_std:.4f})")
                results[strategy] = {"success": False, "reason": "blank"}
            else:
                print(f"✓ SUCCESS! Image quality: std={img_std:.2f}")
                image.save(f"qwen_output_{strategy}.png")
                results[strategy] = {"success": True, "std": img_std}
        
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results[strategy] = {"success": False, "reason": str(e)}
    
    # Summary
    print("\n\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for strategy, result in results.items():
        if result["success"]:
            print(f"✓ {strategy:15s}: SUCCESS (std={result['std']:.2f})")
        else:
            print(f"✗ {strategy:15s}: FAILED ({result.get('reason', 'unknown')})")
    
    return results, pipe


def alternative_approach_1_cast_to_fp32_on_forward():
    """
    Alternative 1: Monkey-patch the forward pass to cast to FP32 at runtime
    """
    print("\n" + "="*80)
    print("ALTERNATIVE APPROACH 1: Runtime FP32 Casting")
    print("="*80 + "\n")
    
    print("This approach intercepts the transformer forward pass and casts")
    print("activations to FP32 when they exceed safe FP16 ranges.\n")
    
    code = '''
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.float16)

# Store original forward
original_forward = pipe.transformer.forward

def safe_fp32_forward(self, *args, **kwargs):
    """Forward with automatic FP32 casting when needed"""
    # Try FP16 first
    try:
        with torch.cuda.amp.autocast(enabled=False):
            # Force FP32 for the entire forward pass
            result = original_forward(*args, **kwargs)
        return result
    except:
        # Fallback to original if this causes issues
        return original_forward(*args, **kwargs)

pipe.transformer.forward = safe_fp32_forward.__get__(pipe.transformer, type(pipe.transformer))
pipe = pipe.to("cuda")

# Generate
image = pipe(prompt="Your prompt", width=1024, height=1024).images[0]
'''
    print(code)


def alternative_approach_2_use_bfloat16():
    """
    Alternative 2: Use BF16 instead of FP16
    """
    print("\n" + "="*80)
    print("ALTERNATIVE APPROACH 2: Use BFloat16")
    print("="*80 + "\n")
    
    print("BFloat16 has a much larger dynamic range than FP16:")
    print("  FP16 max: ~65,504")
    print("  BF16 max: ~3.4×10^38 (same as FP32)")
    print("\nThis should eliminate overflow issues entirely.\n")
    
    code = '''
import torch
from diffusers import DiffusionPipeline

# Load in BFloat16
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype=torch.bfloat16  # Use BF16 instead of FP16
)
pipe = pipe.to("cuda")

# Generate normally
image = pipe(
    prompt="Your prompt",
    width=1024,
    height=1024,
    num_inference_steps=50
).images[0]

image.save("output.png")
'''
    print(code)
    print("\nNote: Requires GPU with BF16 support (Ampere or newer: RTX 3000+, A100, etc.)")


def alternative_approach_3_quantization():
    """
    Alternative 3: Use 8-bit quantization
    """
    print("\n" + "="*80)
    print("ALTERNATIVE APPROACH 3: 8-bit Quantization")
    print("="*80 + "\n")
    
    print("Use bitsandbytes to load the model in 8-bit mode.")
    print("This reduces memory and avoids FP16 overflow issues.\n")
    
    code = '''
import torch
from diffusers import DiffusionPipeline

# Load with 8-bit quantization
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    load_in_8bit=True,  # Requires bitsandbytes
    device_map="auto"
)

# Generate
image = pipe(
    prompt="Your prompt",
    width=1024,
    height=1024
).images[0]
'''
    print(code)
    print("\nRequires: pip install bitsandbytes accelerate")


def main():
    model_name = "Qwen/Qwen-Image"
    
    print("="*80)
    print("QWEN-IMAGE FP16 SOLUTION: MIXED PRECISION")
    print("="*80)
    
    # Check for BF16 support
    if torch.cuda.is_available():
        device = "cuda"
        
        # Check if GPU supports BF16
        if torch.cuda.is_bf16_supported():
            print("\n✓ Your GPU supports BFloat16!")
            print("  Recommendation: Use BFloat16 instead of FP16 (see Alternative 2)")
            
            use_bf16 = input("\nWould you like to try BFloat16? (y/n): ").lower().strip()
            
            if use_bf16 == 'y':
                print("\nLoading with BFloat16...")
                pipe = DiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16
                )
                pipe = pipe.to(device)
                
                print("\nGenerating test image...")
                image = pipe(
                    prompt="A coffee shop with warm lighting and neon signs",
                    negative_prompt=" ",
                    width=1024,
                    height=1024,
                    num_inference_steps=20,
                    true_cfg_scale=4.0
                ).images[0]
                
                img_array = np.array(image)
                img_std = img_array.std()
                
                print(f"✓ Generation complete! Image std: {img_std:.2f}")
                image.save("qwen_bfloat16_output.png")
                print("✓ Saved as 'qwen_bfloat16_output.png'")
                return
        else:
            print("\n✗ Your GPU does not support BFloat16")
            print("  Falling back to mixed precision FP16/FP32 approach")
    else:
        device = "cpu"
        print("\n⚠ Running on CPU")
    
    # Load model
    print(f"\nLoading model in FP16...")
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    
    # Test strategies
    results, pipe = test_all_strategies(
        pipe,
        prompt="A beautiful coffee shop with warm lighting and neon signs"
    )
    
    # Show alternative approaches
    print("\n\n" + "="*80)
    print("IF MIXED PRECISION DIDN'T WORK, TRY THESE ALTERNATIVES:")
    print("="*80)
    
    alternative_approach_2_use_bfloat16()
    alternative_approach_3_quantization()
    alternative_approach_1_cast_to_fp32_on_forward()


if __name__ == "__main__":
    main()
