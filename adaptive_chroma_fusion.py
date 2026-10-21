"""
adaptive_chroma_fusion.py  (v2 — hooks into color_reconstruct.py pipeline)
===========================================================================
Contribution 1: Adaptive Chrominance Fusion Module (ACF)

Your existing color_reconstruct.py does:
    1. Take fused Y from SwinFusion
    2. Take Cr, Cb from PET_RGB only  (100% PET chroma — no MRI blend)
    3. Swap Y → convert back to BGR

Problem: MRI has NO colour information at all in the final image.
         Any MRI tissue detail in the Cb/Cr space is lost.

Our improvement: Learn a per-pixel weight map alpha that blends:
    Cr_final = alpha * Cr_PET + (1-alpha) * Cr_MRI
    Cb_final = alpha * Cb_PET + (1-alpha) * Cb_MRI

Where alpha is predicted by a tiny 2-layer CNN looking at the fused Y channel.
This preserves PET hotspot colours while recovering MRI tissue tones
in low-activity regions where PET has no meaningful colour.

USAGE:
    Mode 1: python adaptive_chroma_fusion.py
            -> self-test (PASSED check)

    Mode 2: python adaptive_chroma_fusion.py --run
            -> processes your existing results like color_reconstruct.py
            -> saves to results/SwinFusion_PET-MRI_ACF/

    Mode 3: import ACFColorReconstruct in your own scripts
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


# ── Core ACF network ──────────────────────────────────────────────────────────

class AdaptiveChromaFusion(nn.Module):
    """
    Lightweight CNN: predicts per-pixel blending weight alpha in [0,1]
    for Cr and Cb channels.

    Input : [Y_fused, Cr_PET, Cb_PET, Cr_MRI, Cb_MRI]  -> 5 channels
    Output: alpha (B,1,H,W) — 1=use PET chroma, 0=use MRI chroma

    At random init, output ~0.5 (equal blend — safe starting point).
    After training, learns to use PET colour in functional hotspot regions
    and MRI grey-tone in background/structural regions.
    """
    def __init__(self, mid_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, Y_f, Cr_pet, Cb_pet, Cr_mri, Cb_mri):
        x = torch.cat([Y_f, Cr_pet, Cb_pet, Cr_mri, Cb_mri], dim=1)
        return self.net(x)


# ── Helper: image <-> tensor ──────────────────────────────────────────────────

def bgr_to_ycrcb_channels(bgr_img):
    """BGR uint8 (H,W,3) -> Y, Cr, Cb as float32 (H,W) in [0,1]"""
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    return ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]


def channels_to_bgr(Y_np, Cr_np, Cb_np):
    """Y, Cr, Cb float32 (H,W) in [0,1] -> BGR uint8 (H,W,3)"""
    ycrcb = np.stack([Y_np, Cr_np, Cb_np], axis=2)
    ycrcb = np.clip(ycrcb * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def to_tensor(x_np):
    return torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0)


def from_tensor(t):
    return t.squeeze().detach().numpy()


# ── Main reconstruction class ─────────────────────────────────────────────────

class ACFColorReconstruct:
    """
    Drop-in replacement for color_reconstruct.py logic.

    Three modes:
        'pet_only'   : original color_reconstruct.py behaviour (baseline)
        'equal_blend': 50/50 blend of PET + MRI chroma
        'acf'        : learned adaptive blend (our contribution)
    """
    def __init__(self, mode='acf', checkpoint=None, device='cpu'):
        self.mode   = mode
        self.device = device
        self.model  = None

        if mode == 'acf':
            self.model = AdaptiveChromaFusion(mid_ch=16).to(device)
            if checkpoint and os.path.exists(checkpoint):
                state = torch.load(checkpoint, map_location=device)
                self.model.load_state_dict(state)
                print(f"[ACF] Loaded checkpoint: {checkpoint}")
            else:
                print("[ACF] No checkpoint — using random init")
            self.model.eval()

    def reconstruct(self, fused_y_gray, pet_bgr, mri_bgr=None):
        """
        fused_y_gray : (H,W) uint8 — fused Y from SwinFusion
        pet_bgr      : (H,W,3) uint8 BGR — original PET image
        mri_bgr      : (H,W,3) uint8 BGR — original MRI image

        Returns: result_bgr (H,W,3) uint8, alpha_map (H,W) float32 or None
        """
        H, W = fused_y_gray.shape
        Y_fused = fused_y_gray.astype(np.float32) / 255.0
        _, Cr_pet, Cb_pet = bgr_to_ycrcb_channels(pet_bgr)

        if self.mode == 'pet_only':
            return channels_to_bgr(Y_fused, Cr_pet, Cb_pet), None

        # Get MRI chroma
        if mri_bgr is None:
            Cr_mri = np.full((H, W), 0.5, dtype=np.float32)
            Cb_mri = np.full((H, W), 0.5, dtype=np.float32)
        else:
            mri_r = cv2.resize(mri_bgr, (W, H)) if mri_bgr.shape[:2] != (H, W) else mri_bgr
            _, Cr_mri, Cb_mri = bgr_to_ycrcb_channels(mri_r)

        if self.mode == 'equal_blend':
            Cr_f = 0.5 * Cr_pet + 0.5 * Cr_mri
            Cb_f = 0.5 * Cb_pet + 0.5 * Cb_mri
            return channels_to_bgr(Y_fused, Cr_f, Cb_f), np.full((H, W), 0.5, np.float32)

        # ACF mode
        with torch.no_grad():
            alpha = self.model(
                to_tensor(Y_fused).to(self.device),
                to_tensor(Cr_pet).to(self.device),
                to_tensor(Cb_pet).to(self.device),
                to_tensor(Cr_mri).to(self.device),
                to_tensor(Cb_mri).to(self.device),
            ).cpu()
        alpha_np = from_tensor(alpha)
        Cr_f = alpha_np * Cr_pet + (1 - alpha_np) * Cr_mri
        Cb_f = alpha_np * Cb_pet + (1 - alpha_np) * Cb_mri
        return channels_to_bgr(Y_fused, Cr_f, Cb_f), alpha_np


# ── Batch processing ──────────────────────────────────────────────────────────

def run_batch(fused_dir, pet_rgb_dir, mri_dir, output_dir,
              mode='acf', checkpoint=None, device='cpu',
              save_alpha=True, compare=True):

    os.makedirs(output_dir, exist_ok=True)
    alpha_dir   = os.path.join(output_dir, 'alpha')
    compare_dir = os.path.join(output_dir, 'compare')
    if save_alpha:   os.makedirs(alpha_dir, exist_ok=True)
    if compare:      os.makedirs(compare_dir, exist_ok=True)

    rec      = ACFColorReconstruct(mode=mode, checkpoint=checkpoint, device=device)
    orig_rec = ACFColorReconstruct(mode='pet_only')

    files = sorted([f for f in os.listdir(fused_dir)
                    if f.lower().endswith(('.png', '.jpg', '.bmp'))])

    print(f"\n[ACF Batch] mode={mode}  {len(files)} images")
    print(f"  fused_dir  : {fused_dir}")
    print(f"  pet_rgb_dir: {pet_rgb_dir}")
    print(f"  mri_dir    : {mri_dir}")
    print(f"  output_dir : {output_dir}\n")

    for fname in files:
        fused_y = cv2.imread(os.path.join(fused_dir, fname), cv2.IMREAD_GRAYSCALE)
        pet_bgr = cv2.imread(os.path.join(pet_rgb_dir, fname))
        mri_bgr_path = os.path.join(mri_dir, fname)
        mri_bgr = cv2.imread(mri_bgr_path) if os.path.exists(mri_bgr_path) else None

        if fused_y is None or pet_bgr is None:
            print(f"  [SKIP] {fname}")
            continue

        if pet_bgr.shape[:2] != fused_y.shape:
            pet_bgr = cv2.resize(pet_bgr, (fused_y.shape[1], fused_y.shape[0]))

        result, alpha_map = rec.reconstruct(fused_y, pet_bgr, mri_bgr)
        cv2.imwrite(os.path.join(output_dir, fname), result)

        if save_alpha and alpha_map is not None:
            alpha_col = cv2.applyColorMap((alpha_map * 255).astype(np.uint8),
                                          cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(alpha_dir, fname), alpha_col)

        if compare:
            orig, _ = orig_rec.reconstruct(fused_y, pet_bgr)
            H, W = fused_y.shape
            panel = np.zeros((H, W * 3, 3), dtype=np.uint8)
            panel[:, :W]    = orig
            panel[:, W:2*W] = result
            if alpha_map is not None:
                panel[:, 2*W:] = cv2.applyColorMap(
                    (alpha_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            for i, lbl in enumerate(['Original (PET only)', 'ACF (ours)', 'Alpha']):
                cv2.putText(panel, lbl, (i*W+8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(compare_dir, fname), panel)

        alpha_info = (f"alpha=[{alpha_map.min():.2f},{alpha_map.max():.2f}]"
                      if alpha_map is not None else "")
        print(f"  {fname}  {alpha_info}")

    print(f"\n  Saved to: {output_dir}")


# ── Self-test ─────────────────────────────────────────────────────────────────

def self_test():
    print("=== AdaptiveChromaFusion self-test ===\n")
    H, W = 256, 256
    dummy_y   = np.random.randint(0, 255, (H, W), dtype=np.uint8)
    dummy_pet = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    dummy_mri = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    for mode in ['pet_only', 'equal_blend', 'acf']:
        rec = ACFColorReconstruct(mode=mode, checkpoint=None)
        result, alpha = rec.reconstruct(dummy_y, dummy_pet, dummy_mri)
        assert result.shape == (H, W, 3), f"Wrong shape: {result.shape}"
        assert result.dtype == np.uint8
        a_str = f"alpha=[{alpha.min():.2f},{alpha.max():.2f}]" if alpha is not None else "alpha=None"
        print(f"  [{mode:12}] shape={result.shape}  {a_str}  OK")

    print("\n  PASSED.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run',         action='store_true')
    p.add_argument('--mode',        default='acf',
                   choices=['pet_only', 'equal_blend', 'acf'])
    p.add_argument('--fused_dir',   default='results/SwinFusion_PET-MRI')
    p.add_argument('--pet_rgb_dir', default='Dataset/testsets/PET-MRI/PET_RGB')
    p.add_argument('--mri_dir',     default='Dataset/testsets/PET-MRI/MRI')
    p.add_argument('--output_dir',  default='results/SwinFusion_PET-MRI_ACF')
    p.add_argument('--checkpoint',  default=None)
    p.add_argument('--device',      default='cpu')
    p.add_argument('--no_alpha',    action='store_true')
    p.add_argument('--no_compare',  action='store_true')
    args = p.parse_args()

    if args.run:
        run_batch(
            fused_dir   = args.fused_dir,
            pet_rgb_dir = args.pet_rgb_dir,
            mri_dir     = args.mri_dir,
            output_dir  = args.output_dir,
            mode        = args.mode,
            checkpoint  = args.checkpoint,
            device      = args.device,
            save_alpha  = not args.no_alpha,
            compare     = not args.no_compare,
        )
    else:
        self_test()
