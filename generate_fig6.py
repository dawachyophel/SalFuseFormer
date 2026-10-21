"""
generate_fig6.py — CT-MRI qualitative comparison
"""
import os, cv2, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PAIRS = ['41003.png', '41010.png', '41018.png']

COLUMNS = [
    {'label': 'CT',           'dir': 'Dataset/testsets/CT-MRI/CT'},
    {'label': 'MRI',          'dir': 'Dataset/testsets/CT-MRI/MRI'},
    {'label': 'DenseFuse',    'dir': 'comparison_results/DenseFuse/CT-MRI'},
    {'label': 'FusionGAN',    'dir': 'comparison_results/FusionGAN/CT-MRI'},
    {'label': 'DDcGAN',       'dir': 'comparison_results/DDcGAN/CT-MRI'},
    {'label': 'U2Fusion',     'dir': 'comparison_results/U2Fusion/CT-MRI'},
    {'label': 'EMFusion',     'dir': 'comparison_results/EMFusion/CT-MRI'},
    {'label': 'SeAFusion',    'dir': 'comparison_results/SeAFusion/CT-MRI'},
    {'label': 'CDDFuse',      'dir': 'comparison_results/CDDFuse/CT-MRI'},
    {'label': 'CoCoNet',      'dir': 'comparison_results/CoCoNet/CT-MRI'},
    {'label': 'MUFusion',     'dir': 'comparison_results/MUFusion/CT-MRI'},
    {'label': 'SwinFusion',   'dir': 'comparison_results/SwinFusion/CT-MRI'},
    {'label': 'SalFuseFormer\n(Ours)', 'dir': 'results/SwinFusion_CT-MRI_MedSwinACF_fixed'},
]

OUTPUT_PATH = 'fig6_ctmri_qualitative.png'

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

fig.suptitle('Qualitative comparison on Harvard CT-MRI dataset',
    fontsize=10, fontweight='bold', y=0.97, color='#333333')
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nFig 6 saved → {OUTPUT_PATH}")