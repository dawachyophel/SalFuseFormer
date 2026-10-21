"""
generate_fig11.py
Fig 11: Two-row combined sensitivity figure
Row 1: tau sweep curves + visuals at tau=0.20,0.40,0.60
Row 2: alpha sweep curves + visuals at alpha=0.05,0.18,0.30
"""
import csv, os, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

TAU_CSV   = 'results/sensitivity_new/tau_sweep.csv'
ALPHA_CSV = 'results/sensitivity_new/alpha_sweep.csv'
OUTPUT    = 'fig11_sensitivity.png'

DEFAULT_TAU   = 0.20
DEFAULT_ALPHA = 0.05
VISUAL_PAIR   = '2.png'
ZOOM_BOX      = (80, 60, 160, 130)
x1,y1,x2,y2  = ZOOM_BOX
W, H          = 256, 256

BASELINE = {'cnr':1.9353,'ssim':0.5819,'fir':0.8754,'mi':3.2278}

def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k:float(v) for k,v in r.items()})
    return rows

tau_rows   = sorted(load_csv(TAU_CSV),   key=lambda r: r['tau'])
alpha_rows = sorted(load_csv(ALPHA_CSV), key=lambda r: r['alpha'])

def add_visual(gs_sub, values, key, default_val, label_prefix, pair, zoom):
    x1,y1,x2,y2 = zoom
    for i, val in enumerate(values):
        ax = plt.subplot(gs_sub[i])
        if key == 'tau':
            path = os.path.join(f'results/sensitivity_new/a{DEFAULT_ALPHA}_t{val}', pair)
        else:
            tau_str = f'{DEFAULT_TAU:.2f}'
            path = os.path.join(f'results/sensitivity_new/a{val}_t{tau_str}', pair)

        if os.path.exists(path):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        else:
            img = np.ones((256,256,3), dtype=np.uint8)*128
            print(f"  [MISSING] {key}={val} {pair}")

        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        rect = Rectangle((x1/W,1-y2/H),(x2-x1)/W,(y2-y1)/H,
                         linewidth=1.5, edgecolor='#FFD700',
                         facecolor='none', transform=ax.transAxes, zorder=5)
        ax.add_patch(rect)
        inset = ax.inset_axes([0.02,0.02,0.40,0.40])
        if os.path.exists(path):
            inset.imshow(cv2.cvtColor(cv2.imread(path)[y1:y2,x1:x2], cv2.COLOR_BGR2RGB))
        inset.set_xticks([]); inset.set_yticks([])
        for sp in inset.spines.values():
            sp.set_edgecolor('#FFD700'); sp.set_linewidth(1.5)

        is_sel = abs(val - default_val) < 0.001
        color = '#0F6E56' if is_sel else '#333333'
        fw    = 'bold'   if is_sel else 'normal'
        suffix = ' ✓' if is_sel else ''
        ax.set_title(f'{label_prefix}={val}{suffix}',
                     fontsize=9, fontweight=fw, color=color, pad=3)
        for sp in ax.spines.values():
            sp.set_edgecolor('#0F6E56' if is_sel else '#cccccc')
            sp.set_linewidth(2.0 if is_sel else 0.5)

fig = plt.figure(figsize=(16, 10), facecolor='white')
gs  = gridspec.GridSpec(2, 2, wspace=0.30, hspace=0.50,
                        left=0.06, right=0.98,
                        top=0.93, bottom=0.07,
                        width_ratios=[1.1, 1.0])

# ── Row 1: Tau sweep ──────────────────────────────────────────────────────────
ax1  = fig.add_subplot(gs[0,0])
ax1b = ax1.twinx()

xs = [r['tau'] for r in tau_rows]
l1, = ax1.plot(xs,  [r['cnr']  for r in tau_rows], color='#E07B39',
               marker='o', linewidth=2, markersize=6, label='CNR (↑)')
l2, = ax1b.plot(xs, [r['ssim'] for r in tau_rows], color='#3B7DD8',
                marker='s', linewidth=2, markersize=6, linestyle='--', label='SSIM (↑)')
l3, = ax1b.plot(xs, [r['fir']  for r in tau_rows], color='#6AAF3D',
                marker='^', linewidth=2, markersize=6, linestyle=':', label='VIF (↑)')

for ax, metric, color in [(ax1,'cnr','#E07B39'),(ax1b,'ssim','#3B7DD8'),(ax1b,'fir','#6AAF3D')]:
    ax.axhline(BASELINE[metric], color=color, linewidth=0.8, linestyle='-.', alpha=0.5)

ax1.axvline(DEFAULT_TAU, color='#0F6E56', linewidth=1.5, linestyle=':', zorder=5)
ax1.set_xlabel('Saliency threshold τ', fontsize=10)
ax1.set_ylabel('CNR', fontsize=10, color='#E07B39')
ax1b.set_ylabel('SSIM / VIF', fontsize=10, color='#3B7DD8')
ax1.tick_params(axis='y', labelcolor='#E07B39')
ax1b.tick_params(axis='y', labelcolor='#3B7DD8')
ax1.set_title('(A) Sensitivity to saliency threshold τ',
              fontsize=11, fontweight='bold', pad=6)
ax1.grid(alpha=0.3, linewidth=0.5); ax1.set_facecolor('#fafafa')
ax1.legend([l1,l2,l3],[l.get_label() for l in [l1,l2,l3]],
           fontsize=8, frameon=False, loc='lower right')

gs_r1 = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[0,1],wspace=0.05)
add_visual(gs_r1, [0.20,0.40,0.60], 'tau', DEFAULT_TAU, 'τ', VISUAL_PAIR, ZOOM_BOX)

# ── Row 2: Alpha sweep ────────────────────────────────────────────────────────
ax2  = fig.add_subplot(gs[1,0])
ax2b = ax2.twinx()

xs2 = [r['alpha'] for r in alpha_rows]
l4, = ax2.plot(xs2,  [r['cnr']  for r in alpha_rows], color='#E07B39',
               marker='o', linewidth=2, markersize=6, label='CNR (↑)')
l5, = ax2b.plot(xs2, [r['ssim'] for r in alpha_rows], color='#3B7DD8',
                marker='s', linewidth=2, markersize=6, linestyle='--', label='SSIM (↑)')
l6, = ax2b.plot(xs2, [r['mi']   for r in alpha_rows], color='#9B59B6',
                marker='^', linewidth=2, markersize=6, linestyle=':', label='MI (↑)')

for ax, metric, color in [(ax2,'cnr','#E07B39'),(ax2b,'ssim','#3B7DD8'),(ax2b,'mi','#9B59B6')]:
    ax.axhline(BASELINE[metric], color=color, linewidth=0.8, linestyle='-.', alpha=0.5)

ax2.axvline(DEFAULT_ALPHA, color='#0F6E56', linewidth=1.5, linestyle=':', zorder=5)
ax2.set_xlabel('Injection strength α', fontsize=10)
ax2.set_ylabel('CNR', fontsize=10, color='#E07B39')
ax2b.set_ylabel('SSIM / MI', fontsize=10, color='#3B7DD8')
ax2.tick_params(axis='y', labelcolor='#E07B39')
ax2b.tick_params(axis='y', labelcolor='#3B7DD8')
ax2.set_title('(B) Sensitivity to injection strength α',
              fontsize=11, fontweight='bold', pad=6)
ax2.grid(alpha=0.3, linewidth=0.5); ax2.set_facecolor('#fafafa')
ax2.legend([l4,l5,l6],[l.get_label() for l in [l4,l5,l6]],
           fontsize=8, frameon=False, loc='lower right')

gs_r2 = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[1,1],wspace=0.05)
add_visual(gs_r2, [0.05,0.18,0.30], 'alpha', DEFAULT_ALPHA, 'α', VISUAL_PAIR, ZOOM_BOX)

fig.suptitle('Parameter sensitivity analysis — saliency threshold τ and injection strength α (MRI-PET)',
             fontsize=12, fontweight='bold', y=0.98)
fig.savefig(OUTPUT, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Fig 11 saved -> {OUTPUT}")
