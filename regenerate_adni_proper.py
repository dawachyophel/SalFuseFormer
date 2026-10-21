"""
regenerate_adni_proper.py
Regenerate ADNI pairs from properly co-registered NIfTI files.
MRI: C:\ADNI_Paper1\preprocessed\{subj}\{subj}_final.nii.gz
PET: D:\ADNI_Paper2\data\pet_coreg\{subj}\{subj}_FDG_coreg.nii.gz
Applies brain masking to PET before JET colormap.
"""
import os, sys, cv2, numpy as np
import nibabel as nib
sys.path.insert(0, '.')

MRI_BASE  = r'C:\ADNI_Paper1\preprocessed'
PET_BASE  = r'D:\ADNI_Paper2\data\pet_coreg'
OUT_MRI   = 'Dataset/testsets/ADNI_PROPER/MRI'
OUT_PET   = 'Dataset/testsets/ADNI_PROPER/PET_RGB'
SUBJ_MAP  = 'adni_proper_subject_map.csv'

os.makedirs(OUT_MRI, exist_ok=True)
os.makedirs(OUT_PET, exist_ok=True)

# Find matched subjects
pet_subjs = set(os.listdir(PET_BASE))
mri_subjs = set(os.listdir(MRI_BASE))
matched   = sorted(pet_subjs & mri_subjs)
print('Matched subjects:', len(matched))

def load_midslice(nii_path):
    img  = nib.load(nii_path)
    data = img.get_fdata().astype(np.float32)
    mid  = data.shape[2] // 2
    return data[:,:,mid]

def norm_uint8(arr):
    mn,mx = arr.min(), arr.max()
    if mx-mn < 1e-8: return np.zeros((256,256),dtype=np.uint8)
    return ((arr-mn)/(mx-mn)*255).astype(np.uint8)

def brain_mask_from_nifti(mri_raw):
    """Use raw NIfTI values to create brain mask before normalization."""
    mask = (np.abs(mri_raw) > 0.01).astype(np.uint8)
    mask = cv2.resize(mask.astype(np.float32),(256,256),
                      interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask

def pet_mask_from_nifti(pet_raw):
    """Use raw PET NIfTI values to create activity mask."""
    mask = (pet_raw > 0.001).astype(np.uint8)
    mask = cv2.resize(mask.astype(np.float32),(256,256),
                      interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

import csv
saved = 0
rows  = []

with open(SUBJ_MAP, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['idx','filename','subject_id'])

    for idx, subj in enumerate(matched):
        mri_path = os.path.join(MRI_BASE, subj, subj+'_final.nii.gz')
        pet_path = os.path.join(PET_BASE, subj, subj+'_FDG_coreg.nii.gz')

        if not os.path.exists(mri_path) or not os.path.exists(pet_path):
            continue

        try:
            mri_sl = load_midslice(mri_path)
            pet_sl = load_midslice(pet_path)

            # Use raw NIfTI values for masking
            brain_mask_256 = brain_mask_from_nifti(mri_sl)
            pet_mask_256   = pet_mask_from_nifti(pet_sl)
            combined_mask  = ((brain_mask_256 > 0) | (pet_mask_256 > 0)).astype(np.uint8)

            # Normalize and resize
            mri_u8 = cv2.resize(norm_uint8(np.clip(mri_sl, -3, 3)), (256,256))
            pet_u8 = cv2.resize(norm_uint8(pet_sl), (256,256))

            # Apply combined mask
            mri_u8[combined_mask == 0] = 0
            pet_u8[combined_mask == 0] = 0

            # Apply JET colormap then mask background
            pet_color = cv2.applyColorMap(pet_u8, cv2.COLORMAP_JET)
            pet_color[combined_mask == 0] = 0

            fname = str(idx+1).zfill(4) + '.png'
            mri_bgr = cv2.cvtColor(mri_u8, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(os.path.join(OUT_MRI, fname), mri_bgr)
            cv2.imwrite(os.path.join(OUT_PET, fname), pet_color)

            writer.writerow([idx+1, fname, subj])
            saved += 1

            if saved % 20 == 0:
                print('  Processed', saved, 'pairs...')

        except Exception as e:
            print('  ERROR', subj, str(e))
            continue

print('Done. Saved', saved, 'proper co-registered pairs')
print('MRI:', OUT_MRI)
print('PET:', OUT_PET)

# Verify one pair
files = sorted(os.listdir(OUT_MRI))[:1]
if files:
    fname = files[0]
    mri = cv2.imread(os.path.join(OUT_MRI, fname))
    pet = cv2.imread(os.path.join(OUT_PET, fname))
    mg  = cv2.cvtColor(mri, cv2.COLOR_BGR2GRAY)
    pg  = cv2.cvtColor(pet, cv2.COLOR_BGR2GRAY)
    mm  = (mg > 8).astype(np.float32)
    pm  = (pg > 0).astype(np.float32)
    overlap = (mm*pm).sum()
    union   = ((mm+pm)>0).sum()
    iou     = overlap/union if union > 0 else 0
    print()
    print('Verification -', fname)
    print('MRI brain pixels:', int(mm.sum()))
    print('PET active pixels:', int(pm.sum()))
    print('IoU:', round(iou,3))
    print('PET corner pixels:', [pg[0,0],pg[0,-1],pg[-1,0],pg[-1,-1]])
    print('PET background = 0:', pg[0,0]==0)
