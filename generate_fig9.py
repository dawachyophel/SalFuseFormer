"""
generate_fig9.py - Saliency and injection weight visualisation
5 columns x 3 rows: PET, Saliency, Injection Weight, Enhanced Y, Fused Output
"""
import os, cv2, numpy as np, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.insert(0, '.')
from medswin_acf_combined import compute_pet_saliency, medswin_acf_fuse

PAIRS = ['2.png', '5.png', '9.png']
PET_DIR = 'Dataset/testsets/PET-MRI/PET_RGB'
MRI_DIR = 'Dataset/testsets/PET-MRI/MRI'
SAL_DIR = 'results/SalFuseFormer_FINAL_PET-MRI'
SW_DIR  = 'results/SwinFusion_PET-MRI'
OUTPUT  = 'fig9_saliency_visualisation.png'

ALPHA = 0.05
TAU   = 0.20

COLS = ['PET Source', 'Saliency Map', 'Injection Weight', 'Enhanced Luminance', 'Fused Output']
n_rows = len(PAIRS)
n_cols = len(COLS)
cell   = 2.0

fig = plt.figure(figsize=(n_cols*cell+0.6, n_rows*cell+0.4), facecolor='white')
gs  = gridspec.GridSpec(n_rows, n_cols,
                        hspace=0.04, wspace=0.03,
                        top=0.92, bottom=0.02,
                        left=0.07, right=0.99)

for r, fname in enumerate(PAIRS):
    pet = cv2.imread(os.path.join(PET_DIR, fname))
    mri = cv2.imread(os.path.join(MRI_DIR, fname))
    sw  = cv2.imread(os.path.join(SW_DIR,  fname), cv2.IMREAD_GRAYSCALE)
    sal = cv2.imread(os.path.join(SAL_DIR, fname))

    if pet is None or mri is None: continue
    pet = cv2.resize(pet,(256,256))
    mri = cv2.resize(mri,(256,256))

    # Compute saliency map
    sal_map = compute_pet_saliency(pet)

    # Injection weight = 1 - saliency (clipped)
    inj_w = np.clip(1.0 - sal_map/(TAU+1e-8), 0, 1)**2

    # Enhanced luminance
    if sw is not None:
        sw = cv2.resize(sw,(256,256))
        Y = sw.astype(np.float32)/255.0
        mri_g = cv2.cvtColor(mri,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        mri_u8 = np.clip(mri_g*255,0,255).astype(np.uint8)
        gx = cv2.Sobel(mri_u8,cv2.CV_32F,1,0,ksize=3)
        gy = cv2.Sobel(mri_u8,cv2.CV_32F,0,1,ksize=3)
        edges = np.sqrt(gx**2+gy**2)
        edges = cv2.GaussianBlur(edges,(0,0),1.5)
        edges = edges/(edges.max()+1e-8)
        Y_enh = np.clip(Y + ALPHA*inj_w*edges,0,1)
        Y_enh_vis = (Y_enh*255).astype(np.uint8)
    else:
        Y_enh_vis = np.zeros((256,256),dtype=np.uint8)

    # Visualisations
    pet_rgb     = cv2.cvtColor(pet, cv2.COLOR_BGR2RGB)
    sal_vis     = cv2.applyColorMap((sal_map*255).astype(np.uint8), cv2.COLORMAP_JET)
    sal_vis_rgb = cv2.cvtColor(sal_vis, cv2.COLOR_BGR2RGB)
    # Use same colormap as original - blue (low injection) to yellow (high injection)
    import matplotlib.cm as cm
    inj_colored = cm.RdYlBu_r(inj_w)
    inj_vis_rgb = (inj_colored[:,:,:3]*255).astype(np.uint8)
    yenh_rgb    = cv2.cvtColor(Y_enh_vis, cv2.COLOR_GRAY2RGB)
    fused_rgb   = cv2.cvtColor(sal, cv2.COLOR_BGR2RGB) if sal is not None else np.zeros((256,256,3),dtype=np.uint8)

    imgs = [pet_rgb, sal_vis_rgb, inj_vis_rgb, yenh_rgb, fused_rgb]

    for c, (img, col_label) in enumerate(zip(imgs, COLS)):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(col_label, fontsize=8, pad=3, fontweight='bold',
                         color='#0F6E56' if c==n_cols-1 else '#333333')
        if c == 0:
            ax.set_ylabel('Pair '+str(r+1), fontsize=8, fontweight='bold', labelpad=4)
        lw = 1.8 if c==n_cols-1 else 0.4
        ec = '#0F6E56' if c==n_cols-1 else '#cccccc'
        for spine in ax.spines.values():
            spine.set_edgecolor(ec); spine.set_linewidth(lw)

fig.suptitle('Saliency and injection weight visualisation — Harvard MRI-PET',
             fontsize=10, fontweight='bold', y=0.97, color='#333333')
fig.savefig(OUTPUT, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Fig 9 saved to', OUTPUT)
