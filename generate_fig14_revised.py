import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('adni_roi_cnr.csv')

ROI_COLS = [
    'CNR_hippocampus', 'CNR_entorhinal_cortex', 'CNR_parahippocampal_gyrus',
    'CNR_amygdala', 'CNR_precuneus', 'CNR_posterior_cingulate',
    'CNR_inferior_temporal', 'CNR_middle_temporal', 'CNR_fusiform_gyrus',
    'CNR_inferior_parietal', 'CNR_banks_sts', 'CNR_lateral_orbitofrontal',
    'CNR_superior_temporal'
]

ROI_LABELS = [
    'Hippocampus', 'Entorhinal\nCortex', 'Parahippocampal\nGyrus',
    'Amygdala', 'Precuneus', 'Posterior\nCingulate',
    'Inferior\nTemporal', 'Middle\nTemporal', 'Fusiform\nGyrus',
    'Inferior\nParietal', 'Banks\nSTS', 'Lateral\nOrbitofrontal',
    'Superior\nTemporal'
]

METHODS = ['SalFuseFormer','SwinFusion','EMFusion','SeAFusion','MUFusion']
COLORS  = {
    'SalFuseFormer': '#0F6E56',
    'SwinFusion':    '#888888',
    'EMFusion':      '#9B59B6',
    'SeAFusion':     '#E07B39',
    'MUFusion':      '#3B7DD8',
}

BONFERRONI_ALPHA = 0.05 / len(ROI_COLS)

fig, ax = plt.subplots(figsize=(16, 7), facecolor='white')

n_methods = len(METHODS)
n_rois    = len(ROI_COLS)
x = np.arange(n_rois)
width = 0.15

sal_data = df[df['method']=='SalFuseFormer'].sort_values('filename').reset_index(drop=True)

for i, method in enumerate(METHODS):
    mdata = df[df['method']==method].sort_values('filename').reset_index(drop=True)
    means = [mdata[roi].mean() for roi in ROI_COLS]
    sds   = [mdata[roi].std()  for roi in ROI_COLS]
    offset = (i - n_methods/2 + 0.5) * width
    bars = ax.bar(x + offset, means, width,
                  yerr=sds, capsize=3,
                  label=method,
                  color=COLORS[method],
                  alpha=0.85 if method != 'SalFuseFormer' else 1.0,
                  edgecolor='white' if method != 'SalFuseFormer' else '#0F6E56',
                  linewidth=0.5 if method != 'SalFuseFormer' else 1.5,
                  error_kw=dict(elinewidth=1.0, ecolor='#333333'),
                  zorder=3 if method == 'SalFuseFormer' else 2)

# No significance markers in figure - reported in manuscript text only

ax.set_xticks(x)
ax.set_xticklabels(ROI_LABELS, fontsize=8)
ax.set_ylabel('Contrast-to-Noise Ratio (Mean ± SD)', fontsize=11)
ax.set_title('ROI-Based CNR on 13 Alzheimer\'s Disease Biomarker Regions\nADNI-3 External Validation (N=66 pairs, Mean ± SD)',
             fontsize=10, fontweight='bold', pad=10)
ax.legend(fontsize=9, frameon=True, loc='upper right')
ax.grid(axis='y', alpha=0.3, linewidth=0.5)
ax.set_facecolor('#fafafa')
ax.set_ylim(0, max([df[df['method']==m][ROI_COLS].values.max() for m in METHODS])*1.25)

import os
os.makedirs('figures', exist_ok=True)
plt.tight_layout()
plt.savefig('figures\\fig14_adni_roi_cnr.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 14 saved to figures/fig14_adni_roi_cnr.png')
