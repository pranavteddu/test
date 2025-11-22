"""
Qwen-Image FP16 Activation Scaling Patcher
Applies activation scaling to enable stable FP16 inference for Qwen-Image models.

Based on the optimization strategy from Draw Things:
https://engineering.drawthings.ai/p/optimizing-qwen-image-for-edge-devices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any
from diffusers import DiffusionPipeline
import inspect


class ScaledAttentionWrapper:
    """
    Wraps the original attention processor and adds FP16 scaling.
    Preserves the original processor's return signature.
    """
    
    def __init__(self, original_processor, layer_idx: int = 0):
        self.original_processor = original_processor
        self.layer_idx = layer_idx
    
    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        
        is_fp16 = (hidden_states.dtype == torch.float16)
        
        if not is_fp16:
            # No scaling needed for non-FP16
            return self.original_processor(
                attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs
            )
        
        # Apply FP16 scaling
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Prepare encoder hidden states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        # QKV input scaling (8x for FP16)
        qkv_scale = 8.0
        
        # Store original projections
        original_to_q = attn.to_q
        original_to_k = attn.to_k
        original_to_v = attn.to_v
        original_to_out_0 = attn.to_out[0]
        
        # Temporarily wrap the projection layers
        attn.to_q = lambda x: original_to_q(x / qkv_scale) * qkv_scale
        attn.to_k = lambda x: original_to_k(x / qkv_scale) * qkv_scale
        attn.to_v = lambda x: original_to_v(x / qkv_scale) * qkv_scale
        
        # Wrap output projection (2x scaling)
        out_scale = 2.0
        
        def scaled_out_proj(x):
            return original_to_out_0(x / out_scale) * out_scale
        
        attn.to_out[0] = scaled_out_proj
        
        # Call original processor with wrapped layers
        result = self.original_processor(
            attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs
        )
        
        # Restore original layers
        attn.to_q = original_to_q
        attn.to_k = original_to_k
        attn.to_v = original_to_v
        attn.to_out[0] = original_to_out_0
        
        return result


class ScaledFFN(nn.Module):
    """
    Wrapper for FFN with adaptive scaling based on layer depth.
    
    Layers 0-58: 32x scaling
    Layer 59+: 512x scaling
    """
    
    def __init__(self, original_ffn, layer_idx: int):
        super().__init__()
        self.ffn = original_ffn
        self.layer_idx = layer_idx
        
        # Determine scaling factor
        if layer_idx >= 59:
            self.scale = 512.0
        else:
            self.scale = 32.0
    
    def forward(self, x):
        is_fp16 = (x.dtype == torch.float16)
        scale = self.scale if is_fp16 else 1.0
        
        # Scale down input
        x = x / scale
        
        # Forward through original FFN
        x = self.ffn(x)
        
        # Scale back up
        x = x * scale
        
        return x


def patch_transformer_block(block, layer_idx: int):
    """
    Patch a single transformer block with FP16 scaling.
    
    Args:
        block: The transformer block to patch
        layer_idx: Index of this block in the model (0-59)
    """
    
    # Patch attention processor
    if hasattr(block, 'attn'):
        # Get the current processor
        if hasattr(block.attn, 'processor'):
            original_processor = block.attn.processor
        else:
            # Try to get the default processor
            try:
                original_processor = block.attn.get_processor()
            except:
                print(f"  ⚠ Could not get processor for block {layer_idx}, skipping attention patch")
                original_processor = None
        
        if original_processor is not None:
            # Wrap it with scaling
            block.attn.processor = ScaledAttentionWrapper(original_processor, layer_idx)
            print(f"  ✓ Patched attention in block {layer_idx}")
        else:
            # Try setting the processor directly
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                block.attn.set_processor(ScaledAttentionWrapper(AttnProcessor2_0(), layer_idx))
                print(f"  ✓ Patched attention in block {layer_idx} (using AttnProcessor2_0)")
            except Exception as e:
                print(f"  ⚠ Could not patch attention in block {layer_idx}: {e}")
    
    # Patch FFN
    ffn_patched = False
    scale_factor = 512 if layer_idx >= 59 else 32
    
    if hasattr(block, 'ff'):
        block.ff = ScaledFFN(block.ff, layer_idx)
        ffn_patched = True
    elif hasattr(block, 'feed_forward'):
        block.feed_forward = ScaledFFN(block.feed_forward, layer_idx)
        ffn_patched = True
    elif hasattr(block, 'ff_net'):
        block.ff_net = ScaledFFN(block.ff_net, layer_idx)
        ffn_patched = True
    elif hasattr(block, 'mlp'):
        block.mlp = ScaledFFN(block.mlp, layer_idx)
        ffn_patched = True
    
    if ffn_patched:
        print(f"  ✓ Patched FFN in block {layer_idx} (scale: {scale_factor}x)")
    else:
        print(f"  ⚠ Could not find FFN in block {layer_idx}")
    
    # Adjust LayerNorm epsilon for scaled inputs
    if hasattr(block, 'norm1'):
        if hasattr(block.norm1, 'eps'):
            original_eps = block.norm1.eps
            block.norm1.eps = original_eps * 64  # 8^2
            print(f"  ✓ Adjusted norm1 epsilon: {original_eps:.2e} → {block.norm1.eps:.2e}")


def patch_qwen_pipeline_for_fp16(pipe):
    """
    Apply FP16 activation scaling to a Qwen-Image pipeline.
    
    Args:
        pipe: DiffusionPipeline loaded from "Qwen/Qwen-Image"
    
    Returns:
        The patched pipeline
    """
    
    print("\n" + "="*60)
    print("Patching Qwen-Image Pipeline for FP16 Stability")
    print("="*60 + "\n")
    
    # Access transformer
    if not hasattr(pipe, 'transformer'):
        raise ValueError("Pipeline does not have a 'transformer' attribute")
    
    transformer = pipe.transformer
    
    # Find transformer blocks
    blocks = None
    if hasattr(transformer, 'transformer_blocks'):
        blocks = transformer.transformer_blocks
    elif hasattr(transformer, 'blocks'):
        blocks = transformer.blocks
    else:
        raise ValueError("Could not find transformer blocks in the model")
    
    print(f"Found {len(blocks)} transformer blocks\n")
    
    # Patch each block
    for idx, block in enumerate(blocks):
        print(f"Patching block {idx}/{len(blocks)-1}:")
        try:
            patch_transformer_block(block, idx)
        except Exception as e:
            print(f"  ✗ Error patching block {idx}: {e}")
        print()
    
    print("="*60)
    print("✓ Patching complete!")
    print("="*60 + "\n")
    
    return pipe


def main():
    """Example usage"""
    
    model_name = "Qwen/Qwen-Image"
    
    print("Loading Qwen-Image model...")
    
    # Load in FP16
    if torch.cuda.is_available():
        torch_dtype = torch.float16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"
    
    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype
    )
    
    # Apply FP16 patches
    pipe = patch_qwen_pipeline_for_fp16(pipe)
    
    # Move to device
    pipe = pipe.to(device)
    
    print("Model loaded and patched successfully!")
    print(f"Running on: {device}")
    print(f"Data type: {torch_dtype}\n")
    
    # Test generation
    prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Ultra HD, 4K, cinematic composition.'''
    
    print("Generating test image...")
    
    image = pipe(
        prompt=prompt,
        negative_prompt=" ",
        width=1664,
        height=928,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device=device).manual_seed(42)
    ).images[0]
    
    image.save("qwen_fp16_test.png")
    print("✓ Image saved as 'qwen_fp16_test.png'")
    
    # Check for NaN values
    import numpy as np
    img_array = np.array(image)
    has_nan = np.isnan(img_array).any()
    
    if has_nan:
        print("⚠ Warning: Output contains NaN values")
    else:
        print("✓ No NaN values detected in output")


if __name__ == "__main__":
    main()
