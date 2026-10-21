import cv2, numpy as np, os, sys
sys.path.insert(0, '.')

# Use pair 5 as representative
PAIR = '5.png'
OUT_DIR = 'figures/fig1_components'
os.makedirs(OUT_DIR, exist_ok=True)

MRI_DIR  = 'Dataset/testsets/PET-MRI/MRI'
PET_DIR  = 'Dataset/testsets/PET-MRI/PET_RGB'
SAL_DIR  = 'results/SalFuseFormer_FINAL_PET-MRI'
FUSED    = 'results/SalFuseFormer_FINAL_PET-MRI'

# 1. MRI input
mri = cv2.resize(cv2.imread(os.path.join(MRI_DIR, PAIR)), (256,256))
cv2.imwrite(os.path.join(OUT_DIR, '1_MRI_input.png'), mri)
print('1. MRI input saved')

# 2. PET input
pet = cv2.resize(cv2.imread(os.path.join(PET_DIR, PAIR)), (256,256))
cv2.imwrite(os.path.join(OUT_DIR, '2_PET_input.png'), pet)
print('2. PET input saved')

# 3. Saliency map
pet_hsv = cv2.cvtColor(pet, cv2.COLOR_BGR2HSV).astype(np.float32)
S = pet_hsv[:,:,1]/255.0
V = pet_hsv[:,:,2]/255.0
raw = S*V
g = cv2.GaussianBlur(raw,(0,0),5.0)
sal = (g/g.max()*255).astype(np.uint8)
sal_color = cv2.applyColorMap(sal, cv2.COLORMAP_HOT)
cv2.imwrite(os.path.join(OUT_DIR, '3_saliency_map.png'), sal_color)
print('3. Saliency map saved')

# 4. HSV components
h_map = cv2.applyColorMap((pet_hsv[:,:,0]/180*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
s_map = cv2.applyColorMap(pet_hsv[:,:,1].astype(np.uint8), cv2.COLORMAP_HOT)
v_map = cv2.cvtColor(pet_hsv[:,:,2].astype(np.uint8), cv2.COLOR_GRAY2BGR)
cv2.imwrite(os.path.join(OUT_DIR, '4a_HSV_hue.png'), h_map)
cv2.imwrite(os.path.join(OUT_DIR, '4b_HSV_saturation.png'), s_map)
cv2.imwrite(os.path.join(OUT_DIR, '4c_HSV_value.png'), v_map)
print('4. HSV components saved')

# 5. Injection weight map
sal_norm = g/g.max()
tau = 0.20
eps = 1e-8
w_inj = np.clip(1 - sal_norm/(tau+eps), 0, 1)**2
w_color = cv2.applyColorMap((w_inj*255).astype(np.uint8), cv2.COLORMAP_COOL)
cv2.imwrite(os.path.join(OUT_DIR, '5_injection_weight.png'), w_color)
print('5. Injection weight map saved')

# 6. MRI Sobel edge map
mri_g = cv2.cvtColor(mri, cv2.COLOR_BGR2GRAY).astype(np.float32)
gx = cv2.Sobel(mri_g, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(mri_g, cv2.CV_32F, 0, 1, ksize=3)
edge = np.sqrt(gx**2+gy**2)
edge_blur = cv2.GaussianBlur(edge,(0,0),1.5)
edge_norm = (edge_blur/edge_blur.max()*255).astype(np.uint8)
edge_bgr = cv2.cvtColor(edge_norm, cv2.COLOR_GRAY2BGR)
cv2.imwrite(os.path.join(OUT_DIR, '6_MRI_edge_map.png'), edge_bgr)
print('6. MRI Sobel edge map saved')

# 7. Enhanced luminance
alpha = 0.05
y_hat = cv2.cvtColor(mri, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
edge_n = edge_blur/edge_blur.max()
y_enh = np.clip(y_hat + alpha*w_inj*edge_n, 0, 1)
y_enh_bgr = cv2.cvtColor((y_enh*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
cv2.imwrite(os.path.join(OUT_DIR, '7_enhanced_luminance.png'), y_enh_bgr)
print('7. Enhanced luminance saved')

# 8. Fused chrominance
pet_ycrcb = cv2.cvtColor(pet, cv2.COLOR_BGR2YCrCb).astype(np.float32)
mri_ycrcb = cv2.cvtColor(mri, cv2.COLOR_BGR2YCrCb).astype(np.float32)
alpha_c = np.clip(sal_norm/(tau+eps), 0, 1)
beta = 0.25
Cr_f = alpha_c*pet_ycrcb[:,:,1] + (1-alpha_c)*(beta*pet_ycrcb[:,:,1]+(1-beta)*mri_ycrcb[:,:,1])
Cb_f = alpha_c*pet_ycrcb[:,:,2] + (1-alpha_c)*(beta*pet_ycrcb[:,:,2]+(1-beta)*mri_ycrcb[:,:,2])
chroma = np.zeros((256,256,3), dtype=np.float32)
chroma[:,:,0] = 128
chroma[:,:,1] = Cr_f
chroma[:,:,2] = Cb_f
chroma_bgr = cv2.cvtColor(np.clip(chroma,0,255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
cv2.imwrite(os.path.join(OUT_DIR, '8_fused_chrominance.png'), chroma_bgr)
print('8. Fused chrominance saved')

# 9. Final fused output
fused = cv2.resize(cv2.imread(os.path.join(FUSED, PAIR)), (256,256))
cv2.imwrite(os.path.join(OUT_DIR, '9_final_fused_output.png'), fused)
print('9. Final fused output saved')

print()
print('All components saved to:', OUT_DIR)
files = sorted(os.listdir(OUT_DIR))
for f in files:
    print(' ', f)
