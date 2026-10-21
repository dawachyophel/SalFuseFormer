"""
generate_fig10.py
Fig 10: Ablation study — 4 configurations x 3 pairs with zoom boxes
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

PAIRS = ['2.png', '5.png', '9.png']

COLUMNS = [
    {'label': 'MRI source',         'dir': 'Dataset/testsets/PET-MRI/MRI'},
    {'label': 'PET source',         'dir': 'Dataset/testsets/PET-MRI/PET_RGB'},
    {'label': '(A) Baseline',       'dir': 'results/ablation_A_baseline_colour'},
    {'label': '(B) Baseline + EI',  'dir': 'results/SwinFusion_PET-MRI_MedSwin'},
    {'label': '(C) Baseline + ACF', 'dir': 'results/SwinFusion_PET-MRI_ACF'},
    {'label': '(D) SalFuseFormer\n(Ours)',
                                     'dir': 'results/SalFuseFormer_FINAL_PET-MRI'},
]

# Zoom box: (x1, y1, x2, y2) in 256x256 image pixel coords
ZOOM_EI  = (80,  60, 160, 130)   # single consistent ROI for all 4 ablation cols
ZOOM_ACF = (80,  60, 160, 130)   # same region — colour difference will be visible
ZOOM_ROW = 0

OUTPUT_PATH = 'fig10_ablation.png'

n_rows = len(PAIRS)
n_cols = len(COLUMNS)
cell   = 2.2

fig = plt.figure(
    figsize=(n_cols * cell + 0.8, n_rows * cell + 0.8),
    facecolor='white')

gs = gridspec.GridSpec(
    n_rows, n_cols,
    hspace=0.05, wspace=0.04,
    top=0.91, bottom=0.02,
    left=0.07, right=0.99)

axes = {}
for r, fname in enumerate(PAIRS):
    for c, col in enumerate(COLUMNS):
        ax = fig.add_subplot(gs[r, c])
        axes[(r, c)] = ax

        path = os.path.join(col['dir'], fname)
        if not os.path.exists(path):
            img_rgb = np.ones((256, 256, 3), dtype=np.uint8) * 128
            img_bgr = img_rgb.copy()
            print(f"  [MISSING] {col['label']}/{fname}")
        else:
            img_bgr = cv2.imread(path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        ax.imshow(img_rgb)
        ax.set_xticks([]); ax.set_yticks([])

        if r == 0:
            fw = 'bold' if c >= 2 else 'normal'
            color = '#0F6E56' if c == n_cols-1 else '#333333'
            ax.set_title(col['label'], fontsize=8.5, fontweight=fw,
                         pad=4, color=color)

        if c == 0:
            ax.set_ylabel(f'Pair {r+1}', fontsize=9,
                          fontweight='bold', labelpad=4)

        if c == n_cols - 1:
            ec, lw = '#0F6E56', 2.0
        elif c >= 2:
            ec, lw = '#888888', 0.8
        else:
            ec, lw = '#cccccc', 0.4
        for spine in ax.spines.values():
            spine.set_edgecolor(ec); spine.set_linewidth(lw)

        # ── Zoom boxes on ZOOM_ROW ────────────────────────────────────────────
        if r == ZOOM_ROW and c in [2, 3, 4, 5] and os.path.exists(path):
            # Same box for cols A+B (edge injection), same box for cols C+D (ACF)
            box   = ZOOM_EI  if c in [2, 3] else ZOOM_ACF
            color = '#FFD700' if c in [2, 3] else '#FF6644'
            # Force same crop region across both pairs
            x1, y1, x2, y2 = box

            x1, y1, x2, y2 = box
            H, W = 256, 256
            rx1, rx2 = x1/W, x2/W
            ry1, ry2 = 1 - y2/H, 1 - y1/H

            # Rectangle on main image
            rect = Rectangle((rx1, ry1), rx2-rx1, ry2-ry1,
                              linewidth=1.8, edgecolor=color,
                              facecolor='none',
                              transform=ax.transAxes, zorder=5)
            ax.add_patch(rect)

            # Inset zoom
            inset = ax.inset_axes([0.02, 0.02, 0.40, 0.40])
            crop  = img_bgr[y1:y2, x1:x2]
            inset.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inset.set_xticks([]); inset.set_yticks([])
            for sp in inset.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(1.8)

fig.text(0.255, 0.955, 'Source images',
         ha='center', fontsize=9, color='#555555', style='italic')
fig.text(0.620, 0.955, 'Ablation configurations',
         ha='center', fontsize=9, color='#333333', fontweight='bold')

fig.suptitle('Ablation study — MRI-PET fusion (Harvard dataset)',
             fontsize=11, fontweight='bold', y=0.995, color='#333333')

fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nFig 10 saved → {OUTPUT_PATH}")
