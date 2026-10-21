"""
generate_fig7.py — MRI-SPECT qualitative comparison
"""
import os, cv2, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PAIRS = ['9024.png', '9033.png', '9042.png']

COLUMNS = [
    {'label': 'MRI',          'dir': 'Dataset/testsets/MRI-SPECT/MRI'},
    {'label': 'SPECT',        'dir': 'Dataset/testsets/MRI-SPECT/SPECT'},
    {'label': 'DenseFuse',    'dir': 'comparison_results/DenseFuse/MRI-SPECT'},
    {'label': 'FusionGAN',    'dir': 'comparison_results/FusionGAN/MRI-SPECT'},
    {'label': 'DDcGAN',       'dir': 'comparison_results/DDcGAN/MRI-SPECT'},
    {'label': 'U2Fusion',     'dir': 'comparison_results/U2Fusion/MRI-SPECT'},
    {'label': 'EMFusion',     'dir': 'comparison_results/EMFusion/MRI-SPECT'},
    {'label': 'SeAFusion',    'dir': 'comparison_results/SeAFusion/MRI-SPECT'},
    {'label': 'CDDFuse',      'dir': 'comparison_results/CDDFuse/MRI-SPECT'},
    {'label': 'CoCoNet',      'dir': 'comparison_results/CoCoNet/MRI-SPECT'},
    {'label': 'MUFusion',     'dir': 'comparison_results/MUFusion/MRI-SPECT'},
    {'label': 'SwinFusion',   'dir': 'comparison_results/SwinFusion/MRI-SPECT'},
    {'label': 'SalFuseFormer\n(Ours)', 'dir': 'results/SalFuseFormer_SPECT_colour'},
]

OUTPUT_PATH = 'fig7_mrispect_qualitative.png'

n_rows = len(PAIRS)
n_cols = len(COLUMNS)
cell   = 1.8

fig = plt.figure(
    figsize=(n_cols * cell + 0.8, n_rows * cell + 0.5),
    facecolor='white')

gs = gridspec.GridSpec(
    n_rows, n_cols,
    hspace=0.04, wspace=0.03,
    top=0.92, bottom=0.02,
    left=0.06, right=0.99)

for r, fname in enumerate(PAIRS):
    for c, col in enumerate(COLUMNS):
        ax = fig.add_subplot(gs[r, c])
        path = os.path.join(col['dir'], fname)
        if os.path.exists(path):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        else:
            img = np.ones((256,256,3), dtype=np.uint8)*128
            print(f"  [MISSING] {col['label']}/{fname}")
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(col['label'], fontsize=7.5, pad=3,
                fontweight='bold' if c==n_cols-1 else 'normal',
                color='#0F6E56' if c==n_cols-1 else '#333333')
        if c == 0:
            ax.set_ylabel(f'Pair {r+1}', fontsize=8,
                fontweight='bold', labelpad=4)
        lw = 1.8 if c==n_cols-1 else 0.4
        ec = '#0F6E56' if c==n_cols-1 else '#cccccc'
        for spine in ax.spines.values():
            spine.set_edgecolor(ec); spine.set_linewidth(lw)

fig.suptitle('Qualitative comparison on Harvard MRI-SPECT dataset',
    fontsize=10, fontweight='bold', y=0.97, color='#333333')
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nFig 7 saved → {OUTPUT_PATH}")