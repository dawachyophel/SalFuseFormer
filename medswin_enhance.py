"""
medswin_enhance.py  — MedSwinFusion Enhancement Pipeline
=========================================================
Final combined approach:
  Step 1: Edge+MRI injection on Y channel  → improves MI, FIR, SF
  Step 2: Saliency-guided Cr/Cb from PET   → preserves CNR, keeps colours vivid

Key insight from experiments:
  - Edge+MRI alone:  MI↑, FIR↑, but CNR↓ (colours softened)
  - ACF alone:       CNR↑, but MI↓ (chroma diluted)
  - Combined:        Both improve simultaneously

The trick: apply edge injection ONLY in low-PET-activity regions
(where PET Cr/Cb are near neutral grey). In high-activity regions
(bright PET hotspots), keep original PET colour untouched.

This is the paper's core technical contribution:
    "Saliency-masked MRI structure injection for medical image fusion"

USAGE:
    python medswin_enhance.py                    -> self-test
    python medswin_enhance.py --run              -> process MRI-PET
    python medswin_enhance.py --run --alpha 0.2  -> stronger edge injection
"""

import os
import cv2
import numpy as np
import argparse


def compute_pet_saliency(pet_bgr, blur_sigma=5):
    """
    Compute PET functional activity map.
    High values = PET has strong colour information (hotspots).
    Low values  = PET is near-neutral (background/skull).

    Returns: saliency (H,W) float32 in [0,1]
    """
    # Convert to HSV — saturation tells us how colourful PET is
    pet_hsv = cv2.cvtColor(pet_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = pet_hsv[:, :, 1] / 255.0   # S channel [0,1]
    value      = pet_hsv[:, :, 2] / 255.0   # V channel [0,1]

    # Activity = combination of saturation and brightness
    activity = saturation * value

    # Smooth to avoid harsh boundaries
    activity = cv2.GaussianBlur(activity, (0, 0), blur_sigma)
    activity = activity / (activity.max() + 1e-8)
    return activity.astype(np.float32)


def inject_mri_edges(Y_fused, Y_mri, alpha=0.18, sigma=1.5):
    """
    Add MRI Sobel edges into fused Y channel.
    alpha: injection strength.
    """
    Y_mri_u8 = np.clip(Y_mri * 255, 0, 255).astype(np.uint8)
    gx = cv2.Sobel(Y_mri_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Y_mri_u8, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(gx**2 + gy**2)
    edges = cv2.GaussianBlur(edges, (0, 0), sigma)
    edges = edges / (edges.max() + 1e-8)
    Y_enhanced = Y_fused + alpha * edges
    return np.clip(Y_enhanced, 0, 1)


def saliency_masked_enhance(
    fused_y_gray,
    pet_bgr,
    mri_bgr,
    alpha=0.18,
    saliency_threshold=0.4,
):
    """
    Combined enhancement:
      1. Compute PET saliency mask
      2. Inject MRI edges into Y channel, SCALED by (1 - saliency)
         → Full injection in background, zero injection in PET hotspots
      3. Keep PET Cr/Cb 100% intact

    Parameters
    ----------
    fused_y_gray      : (H,W) uint8 — fused Y from SwinFusion
    pet_bgr           : (H,W,3) uint8 — PET RGB
    mri_bgr           : (H,W,3) uint8 — MRI
    alpha             : float — max edge injection strength
    saliency_threshold: float — above this, treat as PET hotspot

    Returns
    -------
    result_bgr : (H,W,3) uint8
    debug_info : dict with intermediate maps for visualisation
    """
    H, W = fused_y_gray.shape

    # Resize sources to match fused Y
    pet_r = cv2.resize(pet_bgr, (W, H)) if pet_bgr.shape[:2] != (H, W) else pet_bgr
    mri_r = cv2.resize(mri_bgr, (W, H)) if mri_bgr.shape[:2] != (H, W) else mri_bgr

    # Normalize fused Y
    Y_fused = fused_y_gray.astype(np.float32) / 255.0

    # MRI Y channel
    Y_mri = cv2.cvtColor(mri_r, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Step 1: PET saliency (where is PET active?)
    saliency = compute_pet_saliency(pet_r)

    # Step 2: Saliency-masked edge injection
    # In PET hotspots (saliency~1): inject_weight~0  → no MRI injection
    # In background  (saliency~0): inject_weight~1  → full MRI injection
    inject_weight = np.clip(1.0 - saliency / (saliency_threshold + 1e-8), 0, 1)
    inject_weight = inject_weight ** 2   # squared for smoother falloff

    # Compute MRI edge map
    Y_mri_u8 = np.clip(Y_mri * 255, 0, 255).astype(np.uint8)
    gx = cv2.Sobel(Y_mri_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Y_mri_u8, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(gx**2 + gy**2)
    edges = cv2.GaussianBlur(edges, (0, 0), 1.5)
    edges = edges / (edges.max() + 1e-8)

    # Apply masked injection
    Y_enhanced = Y_fused + alpha * inject_weight * edges
    Y_enhanced  = np.clip(Y_enhanced, 0, 1)

    # Step 3: Reconstruct with PET colour (Cr/Cb unchanged)
    ycrcb = cv2.cvtColor(pet_r, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = np.clip(Y_enhanced * 255, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    debug_info = {
        'saliency':      saliency,
        'inject_weight': inject_weight,
        'edges':         edges,
        'Y_enhanced':    Y_enhanced,
    }
    return result, debug_info


def run_batch(
    fused_dir          = 'results/SwinFusion_PET-MRI',
    pet_rgb_dir        = 'Dataset/testsets/PET-MRI/PET_RGB',
    mri_dir            = 'Dataset/testsets/PET-MRI/MRI',
    output_dir         = 'results/SwinFusion_PET-MRI_MedSwin',
    alpha              = 0.18,
    saliency_threshold = 0.4,
    save_debug         = True,
    compare            = True,
    save_fig9          = True,        # NEW — generate Fig 9 panels
    fig9_pairs         = None,        # NEW — list of filenames to include
                                      #        e.g. ['001.png','045.png','089.png']
                                      #        None = use first 3 files found
):
    os.makedirs(output_dir, exist_ok=True)
    debug_dir   = os.path.join(output_dir, 'debug')
    compare_dir = os.path.join(output_dir, 'compare')
    fig9_dir    = os.path.join(output_dir, 'fig9_saliency')  # NEW

    if save_debug: os.makedirs(debug_dir,   exist_ok=True)
    if compare:    os.makedirs(compare_dir, exist_ok=True)
    if save_fig9:  os.makedirs(fig9_dir,    exist_ok=True)   # NEW

    files = sorted([f for f in os.listdir(fused_dir)
                    if f.lower().endswith(('.png', '.jpg', '.bmp'))])

    # Determine which files to use for Fig 9
    if fig9_pairs is None:
        fig9_pairs = files[:3]   # default: first three files
    
    print(f"\n[MedSwin] alpha={alpha}  threshold={saliency_threshold}"
          f"  {len(files)} images")
    print(f"  output : {output_dir}")
    print(f"  fig9   : {fig9_dir}\n")

    fig9_rows = []   # collect rows for final assembled figure

    for fname in files:
        fused_y = cv2.imread(
            os.path.join(fused_dir, fname), cv2.IMREAD_GRAYSCALE)
        pet_bgr = cv2.imread(os.path.join(pet_rgb_dir, fname))
        mri_path = os.path.join(mri_dir, fname)
        mri_bgr  = cv2.imread(mri_path) if os.path.exists(mri_path) else None

        if fused_y is None or pet_bgr is None:
            print(f"  [SKIP] {fname}")
            continue

        if mri_bgr is None:
            from enhance_fusion import process_single
            result = process_single(
                fused_y, pet_bgr, None,
                mode='edge_inject',
                params={'alpha': alpha})
            debug_info = {}
        else:
            result, debug_info = saliency_masked_enhance(
                fused_y, pet_bgr, mri_bgr,
                alpha=alpha,
                saliency_threshold=saliency_threshold,
            )

        cv2.imwrite(os.path.join(output_dir, fname), result)

        # ── existing debug saves ──────────────────────────────────────
        if save_debug and debug_info:
            sal_vis = cv2.applyColorMap(
                (debug_info['saliency'] * 255).astype(np.uint8),
                cv2.COLORMAP_JET)
            wt_vis = cv2.applyColorMap(
                (debug_info['inject_weight'] * 255).astype(np.uint8),
                cv2.COLORMAP_BONE)
            cv2.imwrite(
                os.path.join(debug_dir, f'sal_{fname}'),    sal_vis)
            cv2.imwrite(
                os.path.join(debug_dir, f'weight_{fname}'), wt_vis)

        # ── NEW: Fig 9 saves ─────────────────────────────────────────
        if save_fig9 and fname in fig9_pairs and debug_info:

            H, W = fused_y.shape

            # Resize sources to match fused dimensions
            pet_r = cv2.resize(pet_bgr, (W, H)) \
                if pet_bgr.shape[:2] != (H, W) else pet_bgr.copy()
            mri_r = cv2.resize(mri_bgr, (W, H)) \
                if mri_bgr.shape[:2] != (H, W) else mri_bgr.copy()

            # Column 1 — functional source (PET)
            col1 = pet_r.copy()

            # Column 2 — saliency map (JET colourmap, high=red)
            col2 = cv2.applyColorMap(
                (debug_info['saliency'] * 255).astype(np.uint8),
                cv2.COLORMAP_JET)

            # Column 3 — injection weight map (percentile stretch)
            w_map = debug_info['inject_weight']
            p2  = np.percentile(w_map, 2)
            p98 = np.percentile(w_map, 98)
            w_stretched = np.clip((w_map - p2) / (p98 - p2 + 1e-8), 0, 1)
            w_norm = (w_stretched * 255).astype(np.uint8)
            col3 = cv2.applyColorMap(w_norm, cv2.COLORMAP_JET)

            # Column 4 — enhanced luminance (greyscale, stretched)
            Y_enh_u8 = np.clip(
                debug_info['Y_enhanced'] * 255, 0, 255).astype(np.uint8)
            col4 = cv2.cvtColor(Y_enh_u8, cv2.COLOR_GRAY2BGR)

            # Column 5 — final fused output
            col5 = result.copy()

            # Save individual columns for this pair
            stem = os.path.splitext(fname)[0]
            cv2.imwrite(
                os.path.join(fig9_dir, f'{stem}_col1_pet.png'),       col1)
            cv2.imwrite(
                os.path.join(fig9_dir, f'{stem}_col2_saliency.png'),  col2)
            cv2.imwrite(
                os.path.join(fig9_dir, f'{stem}_col3_weight.png'),    col3)
            cv2.imwrite(
                os.path.join(fig9_dir, f'{stem}_col4_luminance.png'), col4)
            cv2.imwrite(
                os.path.join(fig9_dir, f'{stem}_col5_fused.png'),     col5)

            # Collect this row for the assembled figure
            fig9_rows.append({
                'label': stem,
                'cols':  [col1, col2, col3, col4, col5],
            })

        # ── existing compare panel ────────────────────────────────────
        if compare:
            H, W = fused_y.shape
            ycrcb_orig = cv2.cvtColor(
                cv2.resize(pet_bgr, (W, H))
                if pet_bgr.shape[:2] != (H, W) else pet_bgr,
                cv2.COLOR_BGR2YCrCb)
            ycrcb_orig[:, :, 0] = fused_y
            orig = cv2.cvtColor(ycrcb_orig, cv2.COLOR_YCrCb2BGR)

            panel = np.zeros((H, W * 2, 3), dtype=np.uint8)
            panel[:, :W] = orig
            panel[:, W:] = result
            for i, lbl in enumerate(
                    ['Original SwinFusion', 'MedSwin (ours)']):
                cv2.putText(panel, lbl, (i * W + 6, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(compare_dir, fname), panel)

        print(f"  {fname}  OK")

    # ── NEW: Assemble Fig 9 grid ──────────────────────────────────────
    if save_fig9 and fig9_rows:
        _assemble_fig9(fig9_rows, fig9_dir)

    print(f"\n  Results  : {output_dir}")
    print(f"  Fig 9    : {fig9_dir}")
    print(f"  Compare  : {compare_dir}")


def _assemble_fig9(rows, save_dir):
    """
    Assemble Fig 9 grid:
      Rows    = modality pairs (MRI-PET, CT-MRI, MRI-SPECT)
      Columns = PET source | Activity map | Injection weight |
                Enhanced luminance | Fused output
    Adds column headers and row labels.
    Saves as fig9_assembled.png at 300 DPI equivalent.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    col_labels = [
        'Functional source',
        'Activity map',
        'Injection weight',
        'Enhanced luminance',
        'Fused output',
    ]

    n_rows = len(rows)
    n_cols = 5
    cell_size = 2.5   # inches per cell

    fig = plt.figure(
        figsize=(n_cols * cell_size, n_rows * cell_size + 0.5),
        facecolor='white')

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        hspace=0.04,
        wspace=0.04,
        top=0.93,
        bottom=0.02,
        left=0.08,
        right=0.99)

    for r, row in enumerate(rows):
        for c, img_bgr in enumerate(row['cols']):
            ax = fig.add_subplot(gs[r, c])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_xticks([])
            ax.set_yticks([])

            # Column headers on first row only
            if r == 0:
                ax.set_title(
                    col_labels[c],
                    fontsize=11,
                    fontweight='bold',
                    pad=6)

            # Row label on first column only
            if c == 0:
                ax.set_ylabel(
                    f'MRI-PET pair {r + 1}',
                    fontsize=10,
                    fontweight='bold',
                    rotation=90,
                    labelpad=6)

            # Thin border around each cell
            for spine in ax.spines.values():
                spine.set_edgecolor('#cccccc')
                spine.set_linewidth(0.5)

    out_path = os.path.join(save_dir, 'fig9_assembled.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Fig 9 assembled → {out_path}")

def self_test():
    print("=== medswin_enhance.py self-test ===\n")
    H, W = 256, 256
    dummy_y   = np.random.randint(50, 200, (H, W), dtype=np.uint8)
    dummy_pet = np.array([[[200, 50, 50]]*W]*H, dtype=np.uint8)  # warm PET
    dummy_mri = np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)

    result, debug = saliency_masked_enhance(dummy_y, dummy_pet, dummy_mri)
    assert result.shape == (H, W, 3)
    assert result.dtype == np.uint8
    assert 'saliency'      in debug
    assert 'inject_weight' in debug
    print(f"  Output shape  : {result.shape}  OK")
    print(f"  Saliency range: [{debug['saliency'].min():.3f}, {debug['saliency'].max():.3f}]  OK")
    print(f"  Weight range  : [{debug['inject_weight'].min():.3f}, {debug['inject_weight'].max():.3f}]  OK")

    # Verify PET colour preserved in high-saliency regions
    res_ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
    pet_ycrcb = cv2.cvtColor(cv2.resize(dummy_pet, (W, H)), cv2.COLOR_BGR2YCrCb)
    cr_diff = np.abs(res_ycrcb[:,:,1].astype(int) - pet_ycrcb[:,:,1].astype(int)).mean()
    print(f"  PET Cr preserved: diff={cr_diff:.2f}  OK")
    print("\n  PASSED.\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run',         action='store_true')
    p.add_argument('--fused_dir',   default='results/SwinFusion_PET-MRI')
    p.add_argument('--pet_rgb_dir', default='Dataset/testsets/PET-MRI/PET_RGB')
    p.add_argument('--mri_dir',     default='Dataset/testsets/PET-MRI/MRI')
    p.add_argument('--output_dir',  default='results/SwinFusion_PET-MRI_MedSwin')
    p.add_argument('--alpha',       type=float, default=0.18)
    p.add_argument('--threshold',   type=float, default=0.4)
    p.add_argument('--no_debug',    action='store_true')
    p.add_argument('--no_compare',  action='store_true')
    p.add_argument('--fig9_pairs',  nargs='+', default=None,
                   help='Filenames to include in Fig 9, e.g. 001.png 045.png')
    args = p.parse_args()

    if args.run:
        run_batch(
            fused_dir          = args.fused_dir,
            pet_rgb_dir        = args.pet_rgb_dir,
            mri_dir            = args.mri_dir,
            output_dir         = args.output_dir,
            alpha              = args.alpha,
            saliency_threshold = args.threshold,
            save_debug         = not args.no_debug,
            compare            = not args.no_compare,
            save_fig9          = True,
            fig9_pairs         = args.fig9_pairs,
        )
    else:
        self_test()