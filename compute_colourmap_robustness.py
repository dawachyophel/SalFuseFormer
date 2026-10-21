"""
compute_colourmap_robustness.py
Tests HSV saliency under non-JET colourmaps on Harvard MRI-PET test set
"""
import cv2, numpy as np, os
from skimage.metrics import structural_similarity as sk_ssim

PET_DIR = 'Dataset/testsets/PET-MRI/PET_RGB'
MRI_DIR = 'Dataset/testsets/PET-MRI/MRI'
OUT_DIR = 'results/colourmap_robustness'
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted([f for f in os.listdir(PET_DIR) if f.endswith('.png')])

def apply_colourmap(gray, cmap):
    return cv2.applyColorMap(gray, cmap)

def hsv_saliency(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[:,:,1]/255.0
    V = hsv[:,:,2]/255.0
    raw = S*V
    g = cv2.GaussianBlur(raw,(0,0),5.0)
    mx = g.max()
    return g/(mx+1e-8) if mx>0 else g

def sal_pet_corr(sal, pet_gray):
    mask = pet_gray > 5
    if mask.sum() < 100: return 0.0
    s = sal[mask].flatten()
    p = (pet_gray[mask].astype(np.float32)/255.0).flatten()
    if s.std()<1e-8 or p.std()<1e-8: return 0.0
    return float(np.corrcoef(s,p)[0,1])

COLOURMAPS = {
    'JET (original)': cv2.COLORMAP_JET,
    'HSV':            cv2.COLORMAP_HSV,
    'HOT':            cv2.COLORMAP_HOT,
    'COOL':           cv2.COLORMAP_COOL,
    'RAINBOW':        cv2.COLORMAP_RAINBOW,
    'GRAYSCALE':      None,
}

print('HSV Saliency-PET Correlation under Alternative Colourmaps')
print('='*70)
print(f'{"Colourmap":<20} {"Mean Pearson r":>15} {"Std":>8}')
print('-'*70)

results = {}
for cmap_name, cmap_id in COLOURMAPS.items():
    corrs = []
    for fname in files:
        pet_bgr = cv2.imread(os.path.join(PET_DIR, fname))
        if pet_bgr is None: continue
        pet_bgr = cv2.resize(pet_bgr,(256,256))
        pet_gray = cv2.cvtColor(pet_bgr, cv2.COLOR_BGR2GRAY)

        if cmap_id is None:
            # Grayscale
            alt_pet = cv2.cvtColor(pet_gray, cv2.COLOR_GRAY2BGR)
        else:
            alt_pet = apply_colourmap(pet_gray, cmap_id)

        sal = hsv_saliency(alt_pet)
        r = sal_pet_corr(sal, pet_gray)
        corrs.append(r)

    mean_r = np.mean(corrs)
    std_r  = np.std(corrs)
    results[cmap_name] = (mean_r, std_r)
    print(f'{cmap_name:<20} {mean_r:>15.4f} {std_r:>8.4f}')

print()
print('Degradation relative to JET:')
jet_r = results['JET (original)'][0]
for cmap_name, (mean_r, std_r) in results.items():
    if cmap_name == 'JET (original)': continue
    deg = jet_r - mean_r
    pct = (deg/abs(jet_r))*100 if jet_r!=0 else 0
    print(f'  {cmap_name:<20} delta_r = {deg:+.4f} ({pct:+.1f}%)')
