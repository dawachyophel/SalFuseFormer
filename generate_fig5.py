"""
generate_fig5.py
Assembles Fig 5 — MRI-PET qualitative comparison grid.
3 rows x 13 columns.
Environment: swinfusion
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PAIRS = ['2.png', '5.png', '9.png']

COLUMNS = [
    {'label': 'MRI',          'dir': 'Dataset/testsets/PET-MRI/MRI'},
    {'label': 'PET',          'dir': 'Dataset/testsets/PET-MRI/PET_RGB'},
    {'label': 'DenseFuse',    'dir': 'comparison_results/DenseFuse/PET-MRI'},
    {'label': 'FusionGAN',    'dir': 'comparison_results/FusionGAN/PET-MRI'},
    {'label': 'DDcGAN',       'dir': 'comparison_results/DDcGAN/PET-MRI'},
    {'label': 'U2Fusion',     'dir': 'comparison_results/U2Fusion/PET-MRI'},
    {'label': 'EMFusion',     'dir': 'comparison_results/EMFusion/PET-MRI'},
    {'label': 'SeAFusion',    'dir': 'comparison_results/SeAFusion/PET-MRI'},
    {'label': 'CDDFuse',      'dir': 'comparison_results/CDDFuse/PET-MRI'},
    {'label': 'CoCoNet',      'dir': 'comparison_results/CoCoNet/PET-MRI'},
    {'label': 'MUFusion',     'dir': 'comparison_results/MUFusion/PET-MRI'},
    {'label': 'SwinFusion',   'dir': 'comparison_results/SwinFusion/PET-MRI'},
    {'label': 'SalFuseFormer\n(Ours)', 'dir': 'results/SalFuseFormer_FINAL_PET-MRI'},
]

OUTPUT_PATH = 'fig5_petmri_qualitative.png'

n_rows = len(PAIRS)
n_cols = len(COLUMNS)
cell   = 1.8

fig = plt.figure(
    figsize=(n_cols * cell + 0.8, n_rows * cell + 0.5),
    facecolor='white')

gs = gridspec.GridSpec(
    n_rows, n_cols,
    hspace=0.04,
    wspace=0.03,
    top=0.92,
    bottom=0.02,
    left=0.06,
    right=0.99)

for r, fname in enumerate(PAIRS):
    for c, col in enumerate(COLUMNS):
        ax = fig.add_subplot(gs[r, c])

        path = os.path.join(col['dir'], fname)
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.ones((256, 256, 3), dtype=np.uint8) * 128
            print(f"  [MISSING] {col['label']}/{fname}")

        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        # Column headers on first row
        if r == 0:
            ax.set_title(
                col['label'],
                fontsize=7.5,
                fontweight='bold' if c == n_cols-1 else 'normal',
                pad=3,
                color='#0F6E56' if c == n_cols-1 else '#333333')

        # Row label
        if c == 0:
            ax.set_ylabel(
                f'Pair {r+1}',
                fontsize=8,
                fontweight='bold',
                labelpad=4)

        # Highlight SalFuseFormer column
        lw = 1.8 if c == n_cols-1 else 0.4
        ec = '#0F6E56' if c == n_cols-1 else '#cccccc'
        for spine in ax.spines.values():
            spine.set_edgecolor(ec)
            spine.set_linewidth(lw)

fig.suptitle(
    'Qualitative comparison on Harvard MRI-PET dataset',
    fontsize=10, fontweight='bold', y=0.97, color='#333333')

fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nFig 5 saved → {OUTPUT_PATH}")