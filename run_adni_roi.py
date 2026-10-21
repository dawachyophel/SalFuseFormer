import cv2, os, sys, numpy as np, pandas as pd
import nibabel as nib

sys.path.insert(0, '.')

FREESURFER_MASKS = r'C:\ADNI_Paper1\freesurfer_masks'
PET_DIR  = 'Dataset/testsets/ADNI-PET-MRI/PET_RGB'
MRI_DIR  = 'Dataset/testsets/ADNI-PET-MRI/MRI'
METHODS  = ['SalFuseFormer','SwinFusion','EMFusion','SeAFusion','MUFusion']
BASE_DIR = 'comparison_results/ADNI'

ROI_MASKS = {
    'hippocampus':           'hippocampus_mask.nii.gz',
    'entorhinal_cortex':     'entorhinal_cortex_mask.nii.gz',
    'parahippocampal_gyrus': 'parahippocampal_gyrus_mask.nii.gz',
    'amygdala':              'amygdala_mask.nii.gz',
    'precuneus':             'precuneus_mask.nii.gz',
    'posterior_cingulate':   'posterior_cingulate_mask.nii.gz',
    'inferior_temporal':     'inferior_temporal_mask.nii.gz',
    'middle_temporal':       'middle_temporal_mask.nii.gz',
    'fusiform_gyrus':        'fusiform_gyrus_mask.nii.gz',
    'inferior_parietal':     'inferior_parietal_mask.nii.gz',
    'banks_sts':             'banks_sts_mask.nii.gz',
    'lateral_orbitofrontal': 'lateral_orbitofrontal_mask.nii.gz',
    'superior_temporal':     'superior_temporal_mask.nii.gz',
}

print('Loading FreeSurfer ROI masks...')
roi_masks_2d = {}
for roi_name, mask_file in ROI_MASKS.items():
    mask_path = os.path.join(FREESURFER_MASKS, mask_file)
    if not os.path.exists(mask_path):
        print(f'  [SKIP] {mask_file}')
        continue
    try:
        mask_nib  = nib.load(mask_path)
        mask_data = mask_nib.get_fdata()
        mask_2d   = mask_data.max(axis=2)
        mask_2d   = cv2.resize(
            (mask_2d > 0).astype(np.float32),
            (256, 256), interpolation=cv2.INTER_NEAREST)
        roi_masks_2d[roi_name] = mask_2d > 0
        print(f'  Loaded {roi_name}: {roi_masks_2d[roi_name].sum()} pixels')
    except Exception as e:
        print(f'  [ERROR] {roi_name}: {e}')

all_roi_union = np.zeros((256,256), dtype=bool)
for mask in roi_masks_2d.values():
    all_roi_union |= mask
bg_mask = ~all_roi_union
print(f'\nBackground mask pixels: {bg_mask.sum()}')
print(f'Total ROI masks loaded: {len(roi_masks_2d)}')

roi_results = []
for method in METHODS:
    method_dir = os.path.join(BASE_DIR, method)
    if not os.path.exists(method_dir):
        print(f'[SKIP] {method}')
        continue
    files = sorted([f for f in os.listdir(method_dir) if f.endswith('.png')])
    print(f'\nROI CNR for {method}: {len(files)} images')
    for fname in files:
        fused_path = os.path.join(method_dir, fname)
        try:
            fused_bgr  = cv2.imread(fused_path)
            if fused_bgr is None:
                continue
            fused_bgr  = cv2.resize(fused_bgr, (256,256))
            fused_gray = cv2.cvtColor(fused_bgr,
                                       cv2.COLOR_BGR2GRAY).astype(np.float32)
            bg_vals = fused_gray[bg_mask]
            mu_bg   = bg_vals.mean() if len(bg_vals) > 0 else 0.0
            std_bg  = max(bg_vals.std(), 1.0) if len(bg_vals) > 0 else 1.0

            row = {'method': method, 'filename': fname}
            for roi_name, roi_mask in roi_masks_2d.items():
                roi_vals = fused_gray[roi_mask]
                if len(roi_vals) == 0:
                    row[f'CNR_{roi_name}'] = 0.0
                    continue
                mu_roi = roi_vals.mean()
                row[f'CNR_{roi_name}'] = round(float(abs(mu_roi-mu_bg)/std_bg), 4)
            roi_results.append(row)
        except Exception as e:
            print(f'  [ERROR] {fname}: {e}')

df_roi = pd.DataFrame(roi_results)
df_roi.to_csv('adni_roi_cnr.csv', index=False)
print(f'\nROI CNR results saved to adni_roi_cnr.csv')

roi_cols = [c for c in df_roi.columns if c.startswith('CNR_')]
print('\nROI-BASED CNR SUMMARY (mean across 66 pairs):')
print('='*80)
roi_summary = df_roi.groupby('method')[roi_cols].mean().round(3)
print(roi_summary.to_string())
roi_summary.to_csv('adni_roi_cnr_summary.csv')
print('\nROI summary saved to adni_roi_cnr_summary.csv')

print('\nMEAN ROI CNR ACROSS ALL 13 REGIONS:')
roi_summary['mean_roi_cnr'] = roi_summary[roi_cols].mean(axis=1)
print(roi_summary[['mean_roi_cnr']].sort_values('mean_roi_cnr', ascending=False).to_string())
