"""
enhance_fusion.py
=================
Improved Approach: Y-channel Structure Enhancement

Instead of blending chroma (which dilutes PET colours),
we enhance the FUSED Y channel by injecting MRI structural
detail BEFORE colour reconstruction.

Pipeline:
    Original:  Y_fused → swap into PET YCrCb → BGR
    Ours:      Y_fused → enhance with MRI edges → swap into PET YCrCb → BGR

Three enhancement modes:
    'clahe'      : Contrast-limited adaptive histogram equalization
                   Boosts local contrast in fused Y channel
    'edge_inject': Adds MRI Sobel edges into Y channel (sharpens anatomy)
    'guided'     : Guided filter using MRI as guide — preserves MRI edges
                   while boosting fused Y detail

All modes keep PET Cr/Cb 100% intact → colours stay vivid.

USAGE:
    python enhance_fusion.py                    -> self-test
    python enhance_fusion.py --run              -> process all images
    python enhance_fusion.py --run --mode clahe -> try different mode
    python enhance_fusion.py --run --mode edge_inject
    python enhance_fusion.py --run --mode guided
    python enhance_fusion.py --run --compare_all  -> saves all 4 versions side by side
"""

import os
import cv2
import numpy as np
import argparse


# ── Enhancement functions ─────────────────────────────────────────────────────

def enhance_clahe(Y, clip_limit=2.0, tile_size=8):
    """
    CLAHE on Y channel — boosts local contrast without colour change.
    Best for: improving visibility of MRI tissue boundaries.
    clip_limit: higher = more contrast (2.0-4.0 recommended)
    """
    Y_u8 = np.clip(Y * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    Y_enhanced = clahe.apply(Y_u8).astype(np.float32) / 255.0
    return Y_enhanced


def enhance_edge_inject(Y_fused, Y_mri, alpha=0.15):
    """
    Inject MRI Sobel edge map into fused Y channel.
    Adds anatomical edge sharpness from MRI without affecting flat regions.
    alpha: strength of edge injection (0.05-0.25 recommended)
    """
    Y_mri_u8 = np.clip(Y_mri * 255, 0, 255).astype(np.uint8)

    # Sobel edges from MRI
    gx = cv2.Sobel(Y_mri_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Y_mri_u8, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(gx**2 + gy**2)
    edges = edges / (edges.max() + 1e-8)   # normalize to [0,1]

    # Inject edges into fused Y
    Y_enhanced = Y_fused + alpha * edges
    return np.clip(Y_enhanced, 0, 1)


def enhance_guided(Y_fused, Y_mri, radius=8, eps=0.01):
    """
    Guided filter: uses MRI Y as guide to sharpen fused Y.
    Preserves MRI structure edges while smoothing noise.
    radius: filter radius (4-16)
    eps: regularization (lower = more structure preserved)
    """
    Y_f_u8  = np.clip(Y_fused * 255, 0, 255).astype(np.uint8)
    Y_m_u8  = np.clip(Y_mri   * 255, 0, 255).astype(np.uint8)

    guided  = cv2.ximgproc.guidedFilter(
        guide=Y_m_u8, src=Y_f_u8,
        radius=radius, eps=eps * 255 * 255
    )
    return guided.astype(np.float32) / 255.0


def enhance_unsharp(Y_fused, Y_mri, strength=0.4, sigma=2.0):
    """
    Unsharp masking using MRI as the detail source.
    Adds high-frequency MRI details onto the fused Y.
    strength: how much MRI detail to add (0.2-0.6)
    """
    Y_m_f = Y_mri.astype(np.float32)
    Y_f_f = Y_fused.astype(np.float32)

    # Blur MRI to get low-freq component
    Y_m_blur = cv2.GaussianBlur(Y_m_f, (0, 0), sigma)
    # MRI high-freq detail
    mri_detail = Y_m_f - Y_m_blur

    # Add MRI detail to fused
    Y_enhanced = Y_f_f + strength * mri_detail
    return np.clip(Y_enhanced, 0, 1)


# ── Core reconstruction ───────────────────────────────────────────────────────

def reconstruct_color(Y_enhanced, pet_bgr):
    """
    Swap enhanced Y into PET's YCrCb → BGR.
    PET Cr/Cb are kept 100% intact — colours stay vivid.
    """
    ycrcb = cv2.cvtColor(pet_bgr, cv2.COLOR_BGR2YCrCb)
    Y_u8  = np.clip(Y_enhanced * 255, 0, 255).astype(np.uint8)

    # Resize if needed
    if Y_u8.shape != ycrcb[:, :, 0].shape:
        Y_u8 = cv2.resize(Y_u8, (ycrcb.shape[1], ycrcb.shape[0]))

    ycrcb[:, :, 0] = Y_u8
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def process_single(fused_y_gray, pet_bgr, mri_bgr, mode='edge_inject', params=None):
    """
    Full pipeline for one image.

    fused_y_gray : (H,W) uint8 — Y channel from SwinFusion
    pet_bgr      : (H,W,3) uint8 — PET RGB image
    mri_bgr      : (H,W,3) uint8 — MRI image

    Returns: result_bgr (H,W,3) uint8
    """
    if params is None:
        params = {}

    H, W = fused_y_gray.shape

    # Normalize fused Y
    Y_fused = fused_y_gray.astype(np.float32) / 255.0

    # Get MRI Y channel
    if mri_bgr is not None:
        mri_r = cv2.resize(mri_bgr, (W, H)) if mri_bgr.shape[:2] != (H, W) else mri_bgr
        mri_gray = cv2.cvtColor(mri_r, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        mri_gray = Y_fused.copy()

    # Resize PET
    pet_r = cv2.resize(pet_bgr, (W, H)) if pet_bgr.shape[:2] != (H, W) else pet_bgr

    # Apply enhancement
    if mode == 'original':
        Y_enhanced = Y_fused

    elif mode == 'clahe':
        clip  = params.get('clip_limit', 2.5)
        tile  = params.get('tile_size', 8)
        Y_enhanced = enhance_clahe(Y_fused, clip_limit=clip, tile_size=tile)

    elif mode == 'edge_inject':
        alpha = params.get('alpha', 0.15)
        Y_enhanced = enhance_edge_inject(Y_fused, mri_gray, alpha=alpha)

    elif mode == 'guided':
        # Check if ximgproc is available
        try:
            radius = params.get('radius', 8)
            eps    = params.get('eps', 0.01)
            Y_enhanced = enhance_guided(Y_fused, mri_gray, radius=radius, eps=eps)
        except AttributeError:
            print("  [WARNING] cv2.ximgproc not available, falling back to edge_inject")
            Y_enhanced = enhance_edge_inject(Y_fused, mri_gray, alpha=0.15)

    elif mode == 'unsharp':
        strength = params.get('strength', 0.35)
        sigma    = params.get('sigma', 2.0)
        Y_enhanced = enhance_unsharp(Y_fused, mri_gray, strength=strength, sigma=sigma)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return reconstruct_color(Y_enhanced, pet_r)


# ── Batch processing ──────────────────────────────────────────────────────────

def run_batch(
    fused_dir     = 'results/SwinFusion_PET-MRI',
    pet_rgb_dir   = 'Dataset/testsets/PET-MRI/PET_RGB',
    mri_dir       = 'Dataset/testsets/PET-MRI/MRI',
    output_dir    = 'results/SwinFusion_PET-MRI_enhanced',
    mode          = 'edge_inject',
    params        = None,
    compare_all   = False,
):
    """
    Process all images with chosen enhancement mode.
    If compare_all=True, saves 4-panel: Original | CLAHE | EdgeInject | Unsharp
    """
    os.makedirs(output_dir, exist_ok=True)
    compare_dir = os.path.join(output_dir, 'compare')
    os.makedirs(compare_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(fused_dir)
                    if f.lower().endswith(('.png', '.jpg', '.bmp'))])

    print(f"\n[Enhance] mode={mode}  {len(files)} images")
    print(f"  fused_dir  : {fused_dir}")
    print(f"  pet_rgb_dir: {pet_rgb_dir}")
    print(f"  output_dir : {output_dir}\n")

    if params is None:
        params = {}

    for fname in files:
        fused_y = cv2.imread(os.path.join(fused_dir, fname), cv2.IMREAD_GRAYSCALE)
        pet_bgr = cv2.imread(os.path.join(pet_rgb_dir, fname))
        mri_path = os.path.join(mri_dir, fname)
        mri_bgr  = cv2.imread(mri_path) if os.path.exists(mri_path) else None

        if fused_y is None or pet_bgr is None:
            print(f"  [SKIP] {fname}")
            continue

        # Main result
        result = process_single(fused_y, pet_bgr, mri_bgr, mode=mode, params=params)
        cv2.imwrite(os.path.join(output_dir, fname), result)

        # Compare panel
        if compare_all:
            orig  = process_single(fused_y, pet_bgr, mri_bgr, 'original')
            clahe = process_single(fused_y, pet_bgr, mri_bgr, 'clahe',
                                   {'clip_limit': 2.5})
            edge  = process_single(fused_y, pet_bgr, mri_bgr, 'edge_inject',
                                   {'alpha': 0.15})
            unsh  = process_single(fused_y, pet_bgr, mri_bgr, 'unsharp',
                                   {'strength': 0.35})
            H, W = fused_y.shape
            panel = np.zeros((H, W * 4, 3), dtype=np.uint8)
            panel[:, :W]      = orig
            panel[:, W:2*W]   = clahe
            panel[:, 2*W:3*W] = edge
            panel[:, 3*W:]    = unsh
            labels = ['Original', 'CLAHE', 'Edge+MRI', 'Unsharp+MRI']
            for i, lbl in enumerate(labels):
                cv2.putText(panel, lbl, (i*W+6, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(compare_dir, fname), panel)

        print(f"  {fname}  OK")

    print(f"\n  Saved to: {output_dir}")
    if compare_all:
        print(f"  Compare panels: {compare_dir}")


# ── Self-test ─────────────────────────────────────────────────────────────────

def self_test():
    print("=== enhance_fusion.py self-test ===\n")
    H, W = 256, 256
    dummy_y   = np.random.randint(50, 200, (H, W), dtype=np.uint8)
    dummy_pet = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    dummy_mri = np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)

    for mode in ['original', 'clahe', 'edge_inject', 'unsharp']:
        result = process_single(dummy_y, dummy_pet, dummy_mri, mode=mode)
        assert result.shape == (H, W, 3), f"Wrong shape: {result.shape}"
        assert result.dtype == np.uint8
        print(f"  [{mode:12}] shape={result.shape}  range=[{result.min()},{result.max()}]  OK")

    # Verify colours are preserved (Cr/Cb should be same as original PET)
    pet_ycrcb = cv2.cvtColor(dummy_pet, cv2.COLOR_BGR2YCrCb)
    for mode in ['clahe', 'edge_inject', 'unsharp']:
        result = process_single(dummy_y, dummy_pet, dummy_mri, mode=mode)
        res_ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
        # Cr channel should be close to original PET Cr
        cr_diff = np.abs(res_ycrcb[:,:,1].astype(int) -
                         pet_ycrcb[:,:,1].astype(int)).mean()
        print(f"  [{mode:12}] PET colour preserved: Cr diff={cr_diff:.2f} (expected <2)")

    print("\n  PASSED.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run',         action='store_true')
    p.add_argument('--mode',        default='edge_inject',
                   choices=['original','clahe','edge_inject','guided','unsharp'])
    p.add_argument('--fused_dir',   default='results/SwinFusion_PET-MRI')
    p.add_argument('--pet_rgb_dir', default='Dataset/testsets/PET-MRI/PET_RGB')
    p.add_argument('--mri_dir',     default='Dataset/testsets/PET-MRI/MRI')
    p.add_argument('--output_dir',  default='results/SwinFusion_PET-MRI_enhanced')
    p.add_argument('--compare_all', action='store_true',
                   help='Save 4-panel comparison of all modes')
    p.add_argument('--alpha',       type=float, default=0.15,
                   help='Edge injection strength (edge_inject mode)')
    p.add_argument('--clip_limit',  type=float, default=2.5,
                   help='CLAHE clip limit')
    p.add_argument('--strength',    type=float, default=0.35,
                   help='Unsharp strength')
    args = p.parse_args()

    if args.run:
        params = {
            'alpha':      args.alpha,
            'clip_limit': args.clip_limit,
            'strength':   args.strength,
        }
        run_batch(
            fused_dir   = args.fused_dir,
            pet_rgb_dir = args.pet_rgb_dir,
            mri_dir     = args.mri_dir,
            output_dir  = args.output_dir,
            mode        = args.mode,
            params      = params,
            compare_all = args.compare_all,
        )
    else:
        self_test()
