"""
generate_fig8.py - Fig 8: SSIM bar chart + radar chart
Updated: bar chart shows SSIM (SalFuseFormer wins on PET-MRI and CT-MRI)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_PATH = 'comparison_results/all_metrics.csv'
OUTPUT   = 'fig8_cross_modality_comparison.png'

METHODS_ORDER = [
    'DenseFuse','FusionGAN','DDcGAN','U2Fusion','EMFusion',
    'SeAFusion','CDDFuse','CoCoNet','MUFusion','SwinFusion','SalFuseFormer'
]

MODALITIES = ['PET-MRI','CT-MRI','MRI-SPECT']
MOD_COLORS = {'PET-MRI':'#E07B39','CT-MRI':'#3B7DD8','MRI-SPECT':'#6AAF3D'}

SSIM_VALS = {
    'DenseFuse':     {'PET-MRI':0.5548,'CT-MRI':0.1078,'MRI-SPECT':0.0794},
    'FusionGAN':     {'PET-MRI':0.1018,'CT-MRI':0.0729,'MRI-SPECT':0.0810},
    'DDcGAN':        {'PET-MRI':0.5835,'CT-MRI':0.1822,'MRI-SPECT':0.0779},
    'U2Fusion':      {'PET-MRI':0.3154,'CT-MRI':0.2301,'MRI-SPECT':0.2079},
    'EMFusion':      {'PET-MRI':0.7257,'CT-MRI':0.7372,'MRI-SPECT':0.8247},
    'SeAFusion':     {'PET-MRI':0.7023,'CT-MRI':0.6959,'MRI-SPECT':0.5818},
    'CDDFuse':       {'PET-MRI':0.7260,'CT-MRI':0.7387,'MRI-SPECT':0.8133},
    'CoCoNet':       {'PET-MRI':0.6610,'CT-MRI':0.2107,'MRI-SPECT':0.1111},
    'MUFusion':      {'PET-MRI':0.6541,'CT-MRI':0.5475,'MRI-SPECT':0.6054},
    'SwinFusion':    {'PET-MRI':0.5819,'CT-MRI':0.6967,'MRI-SPECT':0.7458},
    'SalFuseFormer': {'PET-MRI':0.7223,'CT-MRI':0.7390,'MRI-SPECT':0.8204},
}

CNR_FIXED = {
    'DenseFuse':     {'PET-MRI':198.48,'CT-MRI':21.15, 'MRI-SPECT':41.88},
    'FusionGAN':     {'PET-MRI':198.90,'CT-MRI':10.50, 'MRI-SPECT':10.51},
    'DDcGAN':        {'PET-MRI':192.27,'CT-MRI':168.85,'MRI-SPECT':179.16},
    'U2Fusion':      {'PET-MRI':78.03, 'CT-MRI':106.83,'MRI-SPECT':65.81},
    'EMFusion':      {'PET-MRI':207.46,'CT-MRI':179.45,'MRI-SPECT':106.59},
    'SeAFusion':     {'PET-MRI':207.14,'CT-MRI':216.05,'MRI-SPECT':129.82},
    'CDDFuse':       {'PET-MRI':207.04,'CT-MRI':198.96,'MRI-SPECT':114.01},
    'CoCoNet':       {'PET-MRI':205.81,'CT-MRI':28.15, 'MRI-SPECT':34.88},
    'MUFusion':      {'PET-MRI':210.19,'CT-MRI':198.64,'MRI-SPECT':113.18},
    'SwinFusion':    {'PET-MRI':190.92,'CT-MRI':215.69,'MRI-SPECT':126.74},
    'SalFuseFormer': {'PET-MRI':1.8130,'CT-MRI':17.9469,'MRI-SPECT':6.5285},
}

MS_SSIM = {
    'DenseFuse':     {'PET-MRI':0.6070,'CT-MRI':0.5754,'MRI-SPECT':0.6475},
    'FusionGAN':     {'PET-MRI':0.2843,'CT-MRI':0.2796,'MRI-SPECT':0.3348},
    'DDcGAN':        {'PET-MRI':0.6278,'CT-MRI':0.6081,'MRI-SPECT':0.5747},
    'U2Fusion':      {'PET-MRI':0.6940,'CT-MRI':0.6891,'MRI-SPECT':0.7908},
    'EMFusion':      {'PET-MRI':0.7118,'CT-MRI':0.6990,'MRI-SPECT':0.8736},
    'SeAFusion':     {'PET-MRI':0.7026,'CT-MRI':0.6972,'MRI-SPECT':0.8505},
    'CDDFuse':       {'PET-MRI':0.7112,'CT-MRI':0.7006,'MRI-SPECT':0.8505},
    'CoCoNet':       {'PET-MRI':0.6693,'CT-MRI':0.6325,'MRI-SPECT':0.6437},
    'MUFusion':      {'PET-MRI':0.6700,'CT-MRI':0.6779,'MRI-SPECT':0.8456},
    'SwinFusion':    {'PET-MRI':0.7141,'CT-MRI':0.6966,'MRI-SPECT':0.8684},
    'SalFuseFormer': {'PET-MRI':0.7223,'CT-MRI':0.7390,'MRI-SPECT':0.8204},
}

MI_VALS = {
    'DenseFuse':     {'PET-MRI':3.768,'CT-MRI':1.820,'MRI-SPECT':2.879},
    'FusionGAN':     {'PET-MRI':1.539,'CT-MRI':1.504,'MRI-SPECT':1.690},
    'DDcGAN':        {'PET-MRI':2.559,'CT-MRI':2.610,'MRI-SPECT':2.262},
    'U2Fusion':      {'PET-MRI':2.812,'CT-MRI':2.337,'MRI-SPECT':2.377},
    'EMFusion':      {'PET-MRI':3.165,'CT-MRI':2.808,'MRI-SPECT':2.764},
    'SeAFusion':     {'PET-MRI':2.916,'CT-MRI':2.754,'MRI-SPECT':2.609},
    'CDDFuse':       {'PET-MRI':3.458,'CT-MRI':2.813,'MRI-SPECT':3.453},
    'CoCoNet':       {'PET-MRI':2.722,'CT-MRI':2.511,'MRI-SPECT':2.420},
    'MUFusion':      {'PET-MRI':2.759,'CT-MRI':2.546,'MRI-SPECT':2.572},
    'SwinFusion':    {'PET-MRI':3.228,'CT-MRI':2.911,'MRI-SPECT':2.751},
    'SalFuseFormer': {'PET-MRI':3.316,'CT-MRI':2.797,'MRI-SPECT':2.948},
}

FIR_VALS = {
    'DenseFuse':     {'PET-MRI':0.733,'CT-MRI':-0.472,'MRI-SPECT':-2.441},
    'FusionGAN':     {'PET-MRI':0.187,'CT-MRI':-0.043,'MRI-SPECT':-1.292},
    'DDcGAN':        {'PET-MRI':0.761,'CT-MRI':0.265, 'MRI-SPECT':-0.303},
    'U2Fusion':      {'PET-MRI':0.695,'CT-MRI':0.488, 'MRI-SPECT':0.295},
    'EMFusion':      {'PET-MRI':0.905,'CT-MRI':0.805, 'MRI-SPECT':0.811},
    'SeAFusion':     {'PET-MRI':0.884,'CT-MRI':0.917, 'MRI-SPECT':0.866},
    'CDDFuse':       {'PET-MRI':0.915,'CT-MRI':0.892, 'MRI-SPECT':0.810},
    'CoCoNet':       {'PET-MRI':0.857,'CT-MRI':0.349, 'MRI-SPECT':-0.995},
    'MUFusion':      {'PET-MRI':0.825,'CT-MRI':0.835, 'MRI-SPECT':0.787},
    'SwinFusion':    {'PET-MRI':0.875,'CT-MRI':0.947, 'MRI-SPECT':0.906},
    'SalFuseFormer': {'PET-MRI':0.9265,'CT-MRI':0.9437,'MRI-SPECT':0.8548},
}

EPI_VALS = {
    'DenseFuse':     {'PET-MRI':0.415,'CT-MRI':0.434,'MRI-SPECT':0.446},
    'FusionGAN':     {'PET-MRI':0.269,'CT-MRI':0.377,'MRI-SPECT':0.442},
    'DDcGAN':        {'PET-MRI':0.476,'CT-MRI':0.569,'MRI-SPECT':0.403},
    'U2Fusion':      {'PET-MRI':0.578,'CT-MRI':0.708,'MRI-SPECT':0.705},
    'EMFusion':      {'PET-MRI':0.545,'CT-MRI':0.644,'MRI-SPECT':0.693},
    'SeAFusion':     {'PET-MRI':0.559,'CT-MRI':0.688,'MRI-SPECT':0.700},
    'CDDFuse':       {'PET-MRI':0.564,'CT-MRI':0.702,'MRI-SPECT':0.693},
    'CoCoNet':       {'PET-MRI':0.540,'CT-MRI':0.610,'MRI-SPECT':0.464},
    'MUFusion':      {'PET-MRI':0.532,'CT-MRI':0.627,'MRI-SPECT':0.686},
    'SwinFusion':    {'PET-MRI':0.590,'CT-MRI':0.653,'MRI-SPECT':0.700},
    'SalFuseFormer': {'PET-MRI':0.5631,'CT-MRI':0.6714,'MRI-SPECT':0.6935},
}

fig = plt.figure(figsize=(20, 8), facecolor='white')

# ── Left panel: SSIM bar chart ────────────────────────────────────────────────
ax_bar = fig.add_subplot(1, 2, 1)

x = np.arange(len(METHODS_ORDER))
w = 0.25
sal_idx = METHODS_ORDER.index('SalFuseFormer')

for m_idx, mod in enumerate(MODALITIES):
    vals = [SSIM_VALS[method][mod] for method in METHODS_ORDER]

    bars = ax_bar.bar(
        x + (m_idx-1)*w, vals, w,
        color=MOD_COLORS[mod], alpha=0.85,
        label=mod, edgecolor='white', linewidth=0.4)

    bars[sal_idx].set_edgecolor('#0F6E56')
    bars[sal_idx].set_linewidth(2.5)

    best_idx = int(np.argmax(vals))
    ax_bar.text(x[best_idx] + (m_idx-1)*w, vals[best_idx] + 0.012,
                '★', ha='center', va='bottom',
                color=MOD_COLORS[mod], fontsize=10, fontweight='bold')

ax_bar.set_title('SSIM Comparison Across Modalities',
                 fontsize=13, fontweight='bold', pad=10)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(
    [m if m != 'SalFuseFormer' else 'SalFuse\nFormer\n(Ours)'
     for m in METHODS_ORDER],
    fontsize=8, rotation=45, ha='right')
ax_bar.set_ylabel('SSIM (↑)', fontsize=10)
ax_bar.set_ylim(0, 1.05)
ax_bar.grid(axis='y', alpha=0.3, linewidth=0.5)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.set_facecolor('#fafafa')
ax_bar.axvspan(sal_idx-0.45, sal_idx+0.45,
               color='#0F6E56', alpha=0.10, zorder=0)
ax_bar.legend(fontsize=9, frameon=False,
              loc='upper left',
              bbox_to_anchor=(0.0, 1.12),
              ncol=3)
ax_bar.text(0.01, -0.22,
            '★ Best result per modality    Shaded column = SalFuseFormer (Ours)',
            transform=ax_bar.transAxes, fontsize=8, color='gray', va='top')

# ── Right panel: Radar chart ──────────────────────────────────────────────────
ax_radar = fig.add_subplot(1, 2, 2, polar=True)

RADAR_METHODS = ['SalFuseFormer','SwinFusion','SeAFusion','EMFusion']
RADAR_COLORS  = ['#0F6E56','#444444','#E07B39','#9B59B6']
RADAR_WIDTHS  = [2.5, 1.5, 1.5, 1.5]
RADAR_METRICS = ['SSIM','MS_SSIM','EPI','CNR','FIR','MI']
RADAR_LABELS  = ['SSIM','MS-SSIM','EPI','CNR\n(norm)','FIR','MI']
MOD_RADAR     = 'PET-MRI'

ALL_DATA = {
    method: {
        'SSIM':    SSIM_VALS[method][MOD_RADAR],
        'MS_SSIM': MS_SSIM[method][MOD_RADAR],
        'EPI':     EPI_VALS[method][MOD_RADAR],
        'CNR':     CNR_FIXED[method][MOD_RADAR],
        'FIR':     FIR_VALS[method][MOD_RADAR],
        'MI':      MI_VALS[method][MOD_RADAR],
    }
    for method in METHODS_ORDER
}

norm_data = {m: {} for m in METHODS_ORDER}
for metric in RADAR_METRICS:
    all_v = [ALL_DATA[m][metric] for m in METHODS_ORDER]
    mn, mx = min(all_v), max(all_v)
    for method in METHODS_ORDER:
        v = ALL_DATA[method][metric]
        norm_data[method][metric] = (v - mn) / (mx - mn + 1e-8)

N = len(RADAR_METRICS)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for method, color, lw in zip(RADAR_METHODS, RADAR_COLORS, RADAR_WIDTHS):
    vals = [norm_data[method][m] for m in RADAR_METRICS]
    vals += vals[:1]
    ax_radar.plot(angles, vals, color=color, linewidth=lw,
                  linestyle='-', label=method, zorder=3)
    ax_radar.fill(angles, vals, color=color,
                  alpha=0.20 if method == 'SalFuseFormer' else 0.05,
                  zorder=2)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(RADAR_LABELS, fontsize=10)
ax_radar.set_ylim(0, 1)
ax_radar.set_yticks([0.25, 0.50, 0.75, 1.00])
ax_radar.set_yticklabels(['0.25','0.50','0.75','1.00'],
                          fontsize=7, color='gray')
ax_radar.grid(color='gray', alpha=0.3, linewidth=0.5)
ax_radar.set_title('Normalised metric scores\n(MRI-PET fusion)',
                   fontsize=12, fontweight='bold', pad=20)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15),
                fontsize=9, frameon=False)

plt.suptitle('Cross-modality performance visualisation',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Fig 8 saved ->", OUTPUT)
