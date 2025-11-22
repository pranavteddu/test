from diffusers import DiffusionPipeline
import torch
import torch.nn as nn
from typing import Optional

class ScaledLinear(nn.Module):
    """Linear layer with input/output scaling for FP16 stability"""
    def __init__(self, linear: nn.Linear, input_scale: float = 1.0, output_scale: float = 1.0):
        super().__init__()
        self.linear = linear
        self.input_scale = input_scale
        self.output_scale = output_scale
    
    def forward(self, x):
        # Scale down input to prevent overflow
        x = x / self.input_scale
        # Perform linear operation
        x = self.linear(x)
        # Scale up output to restore magnitude
        x = x * self.output_scale
        return x

def patch_mmdit_block_for_fp16(block, layer_idx: int, total_layers: int = 60):
    """
    Patch an MMDiT block with activation scaling for FP16 stability.
    
    Based on Draw Things' strategy:
    - Input to q/k/v projection: down-scale by 8
    - Attention output: down-scale by 2, up-scale after out_proj
    - FFN: 32× down-scale for layers 0-58, 512× for layer 59
    """
    
    # Determine FFN scaling factor based on layer
    if layer_idx >= 59:
        ffn_scale = 512.0
    else:
        ffn_scale = 32.0
    
    # Scale q/k/v projections (input scale by 8)
    if hasattr(block, 'attn'):
        attn = block.attn
        
        # Wrap q, k, v projections
        if hasattr(attn, 'to_q'):
            attn.to_q = ScaledLinear(attn.to_q, input_scale=8.0, output_scale=1.0)
        if hasattr(attn, 'to_k'):
            attn.to_k = ScaledLinear(attn.to_k, input_scale=8.0, output_scale=1.0)
        if hasattr(attn, 'to_v'):
            attn.to_v = ScaledLinear(attn.to_v, input_scale=8.0, output_scale=1.0)
        
        # Wrap output projection (down-scale by 2 before, up-scale by 2 after)
        if hasattr(attn, 'to_out'):
            if isinstance(attn.to_out, nn.ModuleList) or isinstance(attn.to_out, nn.Sequential):
                original_layer = attn.to_out[0]
                attn.to_out[0] = ScaledLinear(original_layer, input_scale=2.0, output_scale=2.0)
            else:
                attn.to_out = ScaledLinear(attn.to_out, input_scale=2.0, output_scale=2.0)
    
    # Scale FFN pathway
    if hasattr(block, 'ff'):
        ff = block.ff
        # Wrap the first linear layer in FFN
        if hasattr(ff, 'net') and isinstance(ff.net, nn.Sequential):
            if len(ff.net) > 0 and isinstance(ff.net[0], nn.Linear):
                ff.net[0] = ScaledLinear(ff.net[0], input_scale=ffn_scale, output_scale=ffn_scale)
        elif hasattr(ff, 'linear1'):
            ff.linear1 = ScaledLinear(ff.linear1, input_scale=ffn_scale, output_scale=ffn_scale)
    
    # Also adjust RMS norm epsilon if needed
    if hasattr(block, 'norm1') and hasattr(block.norm1, 'eps'):
        # Adjust epsilon proportionally to input scaling
        block.norm1.eps = block.norm1.eps * (8.0 ** 2)
    
    return block

def apply_activation_scaling_to_pipeline(pipe, total_layers: int = 60):
    """
    Apply activation scaling to all MMDiT blocks in the pipeline.
    """
    # Access the transformer model
    if hasattr(pipe, 'transformer'):
        transformer = pipe.transformer
        
        # Find and patch transformer blocks
        if hasattr(transformer, 'transformer_blocks'):
            blocks = transformer.transformer_blocks
            for idx, block in enumerate(blocks):
                patch_mmdit_block_for_fp16(block, idx, total_layers)
                print(f"Patched transformer block {idx}/{len(blocks)-1}")
        
        elif hasattr(transformer, 'blocks'):
            blocks = transformer.blocks
            for idx, block in enumerate(blocks):
                patch_mmdit_block_for_fp16(block, idx, total_layers)
                print(f"Patched transformer block {idx}/{len(blocks)-1}")
    
    return pipe

# Main execution
model_name = "Qwen/Qwen-Image"

# Load the pipeline in float16
print("Loading pipeline...")
if torch.cuda.is_available():
    torch_dtype = torch.float16  # Use float16 instead of bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)

# Apply activation scaling patches
print("\nApplying activation scaling for FP16 stability...")
pipe = apply_activation_scaling_to_pipeline(pipe)

pipe = pipe.to(device)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图."
}

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''
negative_prompt = " "

aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

print(f"\nGenerating {width}x{height} image...")
image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device=device).manual_seed(42)
).images[0]

image.save("example.png")
print("Image saved as example.png")
