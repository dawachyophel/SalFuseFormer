import torch
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.insert(0, r'C:\Users\chyop\Desktop\Image fusion! NEW\SwinFusion-master')
sys.path.insert(0, r'C:\Users\chyop\Desktop\Image fusion! NEW\SwinFusion-master\models')
sys.path.insert(0, r'C:\Users\chyop\Desktop\Image fusion! NEW\SwinFusion-master\options')

from thop import profile, clever_format
from models.network_swinfusion import define_G

# Build model same way as test_swinfusion.py
opt = {
    'scale': 1,
    'n_channels': 1,
    'net_type': 'swinfusion',
    'in_nc': 1,
    'out_nc': 1,
    'nf': 32,
    'nb': 1,
    'gc': 32,
    'ng': 2,
    'reduction': 4,
    'act_type': 'relu',
    'upsample_mode': 'upconv',
    'mode': 'CNA',
    'res_scale': 1,
    'norm_type': None,
}

class SimpleOpt:
    pass

try:
    from models.select_network import define_network
    print("Trying define_network...")
except:
    pass

# Load state dict and count params directly
ckpt_g = torch.load(
    r'C:\Users\chyop\Desktop\Image fusion! NEW\SwinFusion-master\Model\Medical_Fusion-PET-MRI\Medical_Fusion\models\10000_G.pth',
    map_location='cpu')
ckpt_e = torch.load(
    r'C:\Users\chyop\Desktop\Image fusion! NEW\SwinFusion-master\Model\Medical_Fusion-PET-MRI\Medical_Fusion\models\10000_E.pth',
    map_location='cpu')

params_g = sum(v.numel() for v in ckpt_g.values() if hasattr(v,'numel'))
params_e = sum(v.numel() for v in ckpt_e.values() if hasattr(v,'numel'))
total = params_g + params_e
print(f'Generator params: {params_g/1e6:.2f}M')
print(f'Encoder params:   {params_e/1e6:.2f}M')
print(f'Total params:     {total/1e6:.2f}M')

# Estimate FLOPs from parameter count using standard ratio
# For transformer models: FLOPs ≈ 2 × params × sequence_length
# For 256×256 input, sequence = 256*256/patch^2
# SwinFusion uses patch=1, so rough estimate:
h, w = 256, 256
flops_estimate = 2 * total * h * w / 1e9
print(f'Estimated FLOPs (rough): {flops_estimate:.2f} GFLOPs')
print(f'Note: Exact FLOPs require model instantiation with full config')