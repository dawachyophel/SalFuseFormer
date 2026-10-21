import cv2, numpy as np, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PET_DIR = 'Dataset/testsets/PET-MRI/PET_RGB'
os.makedirs('figures', exist_ok=True)

fname = '5.png'
pet_bgr = cv2.resize(cv2.imread(os.path.join(PET_DIR, fname)), (256,256))
pet_gray = cv2.cvtColor(pet_bgr, cv2.COLOR_BGR2GRAY)

def hsv_saliency(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[:,:,1]/255.0; V = hsv[:,:,2]/255.0
    raw = S*V
    g = cv2.GaussianBlur(raw,(0,0),5.0)
    mx = g.max()
    return g/(mx+1e-8) if mx>0 else g

def sal_pet_corr(sal, pg):
    mask = pg > 5
    if mask.sum()<100: return 0.0
    s=sal[mask].flatten(); p=(pg[mask].astype(np.float32)/255.0).flatten()
    if s.std()<1e-8 or p.std()<1e-8: return 0.0
    return float(np.corrcoef(s,p)[0,1])

CMAPS = [
    ('JET',       cv2.COLORMAP_JET,     pet_bgr),
    ('HOT',       cv2.COLORMAP_HOT,     None),
    ('HSV',       cv2.COLORMAP_HSV,     None),
    ('COOL',      cv2.COLORMAP_COOL,    None),
    ('RAINBOW',   cv2.COLORMAP_RAINBOW, None),
    ('Grayscale', None,                 None),
]

fig, axes = plt.subplots(2, 6, figsize=(20,8), facecolor='white')
fig.subplots_adjust(hspace=0.05, wspace=0.05, left=0.06, right=0.92, top=0.88, bottom=0.02)

for i, (label, cmap_id, orig) in enumerate(CMAPS):
    if orig is not None:
        alt_pet = orig
    elif cmap_id is None:
        alt_pet = cv2.cvtColor(pet_gray, cv2.COLOR_GRAY2BGR)
    else:
        alt_pet = cv2.applyColorMap(pet_gray, cmap_id)

    sal = hsv_saliency(alt_pet)
    r = sal_pet_corr(sal, pet_gray)

    color = '#0F6E56' if label=='JET' else ('#E07B39' if label=='HOT' else '#CC3333')
    sym = '+' if r>=0 else ''

    axes[0,i].imshow(cv2.cvtColor(alt_pet, cv2.COLOR_BGR2RGB))
    axes[0,i].set_title(f'{label}\n(r={sym}{r:.3f})', fontsize=9,
                        fontweight='bold', color=color, pad=4)
    axes[0,i].axis('off')
    axes[0,i].set_facecolor('white')
    for sp in axes[0,i].spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(2 if label=='JET' else 1)

    axes[1,i].imshow(sal, cmap='inferno', vmin=0, vmax=1)
    axes[1,i].axis('off')
    axes[1,i].set_facecolor('white')
    for sp in axes[1,i].spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(2 if label=='JET' else 1)

axes[0,0].set_ylabel('PET Input', fontsize=10, color='#333333', fontweight='bold')
axes[1,0].set_ylabel('HSV Saliency\nMap', fontsize=10, color='#333333', fontweight='bold')

cax = fig.add_axes([0.93, 0.05, 0.01, 0.4])
sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(0,1))
cb = fig.colorbar(sm, cax=cax)
cb.set_label('Saliency', color='#333333', fontsize=9)
cb.ax.yaxis.set_tick_params(color='#333333')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='#333333')


plt.savefig('figures/fig_colourmap_robustness.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print('Fig saved.')
