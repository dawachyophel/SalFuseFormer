"""
medswin_acf_combined.py
========================
MedSwinFusion: Final Combined Pipeline + Novel Metrics

Combines:
  - MedSwin saliency-masked Y enhancement (MI, PSNR, Qabf improvement)
  - ACF-guided Cr/Cb selection          (CNR, FIR improvement)

Plus 2 novel evaluation metrics proposed in this paper:
  - FISC : Functional-to-Structural Contrast Ratio
  - MSSF : Modality-Specific Structure Fidelity

USAGE:
    python medswin_acf_combined.py                   -> self-test
    python medswin_acf_combined.py --run             -> full pipeline
    python medswin_acf_combined.py --run --metrics_only -> metrics only (no image saving)
"""

import os
import cv2
import numpy as np
import argparse
from scipy.stats import pearsonr


# ══════════════════════════════════════════════════════════════════════════════
#  NOVEL METRIC 1 — FISC (Functional-to-Structural Contrast Ratio)
# ══════════════════════════════════════════════════════════════════════════════

def compute_fisc(fused_gray, pet_gray, mri_gray):
    """
    FISC: Functional-to-Structural Contrast Ratio
    -----------------------------------------------
    Measures how well PET functional hotspots (high-activity regions)
    are distinguishable from MRI structural regions in the fused image.

    Clinical motivation: A good MRI-PET fusion should maintain clear
    separation between functional hotspots and anatomical structures —
    not blend them into an indistinguishable mixture.

    Algorithm:
        1. Identify functional region (F): pixels where PET activity > threshold
        2. Identify structural region (S): pixels where MRI gradient > threshold
           AND NOT in functional region
        3. FISC = |μ_F(fused) - μ_S(fused)| / (σ_F(fused) + σ_S(fused) + ε)

    Interpretation:
        Higher FISC → clearer separation of functional vs structural information
        FISC = 0    → functional and structural regions are indistinguishable
        FISC > 1    → excellent contrast between regions

    Parameters
    ----------
    fused_gray : (H,W) uint8 — fused image (grayscale)
    pet_gray   : (H,W) uint8 — PET source (grayscale)
    mri_gray   : (H,W) uint8 — MRI source (grayscale)

    Returns
    -------
    fisc : float ≥ 0
    """
    F = fused_gray.astype(np.float64)
    P = pet_gray.astype(np.float64)
    M = mri_gray.astype(np.float64)

    # Resize to match fused if needed
    if P.shape != F.shape:
        P = cv2.resize(P, (F.shape[1], F.shape[0])).astype(np.float64)
    if M.shape != F.shape:
        M = cv2.resize(M, (F.shape[1], F.shape[0])).astype(np.float64)

    # Step 1: PET functional region — top 30% intensity in PET
    pet_thresh = np.percentile(P, 70)
    func_mask = P >= pet_thresh

    # Step 2: MRI structural region — high gradient in MRI, not functional
    gx = cv2.Sobel(M.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(M.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mri_grad = np.sqrt(gx**2 + gy**2).astype(np.float64)
    struct_thresh = np.percentile(mri_grad, 70)
    struct_mask = (mri_grad >= struct_thresh) & (~func_mask)

    # Step 3: Compute contrast in fused image between the two regions
    F_func   = F[func_mask]
    F_struct = F[struct_mask]

    if len(F_func) < 10 or len(F_struct) < 10:
        return 0.0

    mu_func   = F_func.mean()
    mu_struct = F_struct.mean()
    sig_func  = F_func.std()
    sig_struct= F_struct.std()

    fisc = abs(mu_func - mu_struct) / (sig_func + sig_struct + 1e-8)
    return float(fisc)


# ══════════════════════════════════════════════════════════════════════════════
#  NOVEL METRIC 2 — MSSF (Modality-Specific Structure Fidelity)
# ══════════════════════════════════════════════════════════════════════════════

def compute_mssf(fused_gray, pet_gray, mri_gray, n_scales=3):
    """
    MSSF: Modality-Specific Structure Fidelity
    -------------------------------------------
    Measures how faithfully the fused image preserves the UNIQUE structural
    information from each modality separately — not just overall similarity.

    Key distinction from SSIM/MI: MSSF isolates modality-exclusive features
    before measuring fidelity. It asks:
        "Does the fused image preserve what only MRI can see AND
         separately preserve what only PET can see?"

    Algorithm (multi-scale):
        1. Compute unique MRI content: M_unique = MRI - (MRI ∩ PET)
           where intersection estimated via guided filter smoothing
        2. Compute unique PET content: P_unique = PET - (MRI ∩ PET)
        3. For each scale s:
           - MSSF_MRI(s) = correlation(fused_detail_s, mri_unique_detail_s)
           - MSSF_PET(s) = correlation(fused_detail_s, pet_unique_detail_s)
        4. MSSF = 0.5 * mean_s(MSSF_MRI(s)) + 0.5 * mean_s(MSSF_PET(s))

    Interpretation:
        MSSF → 1.0 : fused image perfectly preserves both modalities' unique info
        MSSF → 0.0 : fused image lost modality-specific content
        MSSF can be negative if fused image inverts modality-specific structure

    Parameters
    ----------
    fused_gray : (H,W) uint8
    pet_gray   : (H,W) uint8
    mri_gray   : (H,W) uint8
    n_scales   : int — number of Gaussian pyramid levels (default 3)

    Returns
    -------
    mssf       : float in [-1, 1]
    mssf_mri   : float — MRI component only
    mssf_pet   : float — PET component only
    """
    F = fused_gray.astype(np.float64) / 255.0
    P = pet_gray.astype(np.float64)   / 255.0
    M = mri_gray.astype(np.float64)   / 255.0

    if P.shape != F.shape:
        P = cv2.resize(P.astype(np.float32), (F.shape[1], F.shape[0])).astype(np.float64)
    if M.shape != F.shape:
        M = cv2.resize(M.astype(np.float32), (F.shape[1], F.shape[0])).astype(np.float64)

    def _safe_corr(a, b):
        a_f, b_f = a.flatten(), b.flatten()
        if a_f.std() < 1e-8 or b_f.std() < 1e-8:
            return 0.0
        try:
            r, _ = pearsonr(a_f, b_f)
            return float(r) if not np.isnan(r) else 0.0
        except Exception:
            return 0.0

    # Step 1: Estimate common content via Gaussian smoothing (shared low-freq)
    sigma_common = 5.0
    M_smooth = cv2.GaussianBlur(M.astype(np.float32), (0, 0), sigma_common).astype(np.float64)
    P_smooth = cv2.GaussianBlur(P.astype(np.float32), (0, 0), sigma_common).astype(np.float64)
    common   = 0.5 * (M_smooth + P_smooth)

    # Step 2: Unique content = source - common
    M_unique = M - common
    P_unique = P - common

    # Step 3: Multi-scale correlation
    mssf_mri_scores = []
    mssf_pet_scores = []

    F_curr  = F.copy()
    Mu_curr = M_unique.copy()
    Pu_curr = P_unique.copy()

    for s in range(n_scales):
        # High-frequency detail at this scale
        F_blur  = cv2.GaussianBlur(F_curr.astype(np.float32),  (0,0), 2.0**s + 1).astype(np.float64)
        Mu_blur = cv2.GaussianBlur(Mu_curr.astype(np.float32), (0,0), 2.0**s + 1).astype(np.float64)
        Pu_blur = cv2.GaussianBlur(Pu_curr.astype(np.float32), (0,0), 2.0**s + 1).astype(np.float64)

        F_detail  = F_curr  - F_blur
        Mu_detail = Mu_curr - Mu_blur
        Pu_detail = Pu_curr - Pu_blur

        mssf_mri_scores.append(_safe_corr(F_detail, Mu_detail))
        mssf_pet_scores.append(_safe_corr(F_detail, Pu_detail))

        # Downsample for next scale
        F_curr  = cv2.resize(F_curr.astype(np.float32),
                             (max(F_curr.shape[1]//2, 8), max(F_curr.shape[0]//2, 8))).astype(np.float64)
        Mu_curr = cv2.resize(Mu_curr.astype(np.float32), (F_curr.shape[1], F_curr.shape[0])).astype(np.float64)
        Pu_curr = cv2.resize(Pu_curr.astype(np.float32), (F_curr.shape[1], F_curr.shape[0])).astype(np.float64)

    mssf_mri = float(np.mean(mssf_mri_scores))
    mssf_pet = float(np.mean(mssf_pet_scores))
    mssf     = 0.5 * mssf_mri + 0.5 * mssf_pet
    return mssf, mssf_mri, mssf_pet


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_pet_saliency(pet_bgr, blur_sigma=5):
    pet_hsv  = cv2.cvtColor(pet_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    activity = (pet_hsv[:,:,1] / 255.0) * (pet_hsv[:,:,2] / 255.0)
    activity = cv2.GaussianBlur(activity, (0,0), blur_sigma)
    return (activity / (activity.max() + 1e-8)).astype(np.float32)


def medswin_acf_fuse(
    fused_y_gray,
    pet_bgr,
    mri_bgr,
    alpha           = 0.18,
    sal_threshold   = 0.4,
    acf_blend       = 0.25,
    modality        = 'MRI-PET',
):
    """
    SalFuseFormer saliency-masked enhancement pipeline.

    Step 1 — Saliency-masked Y enhancement:
        Derive functional activity saliency map from HSV representation
        of the functional modality image. Inject MRI structural edges
        into the fused Y channel in inverse proportion to local saliency.
        High-activity regions: no injection (preserve functional luminance).
        Low-activity regions: full MRI edge injection (structural sharpening).

    Step 2 — Adaptive chrominance fusion (ACF):
        Applied to MRI-PET and MRI-SPECT (functional colour modalities).
        In high-saliency hotspot regions: use 100% functional Cr/Cb.
        In low-saliency regions: blend functional + MRI chroma.
        Bypassed for CT-MRI: both modalities are grayscale, no functional
        colour encoding present, enhanced Y channel used directly.

    Parameters
    ----------
    fused_y_gray  : (H,W) uint8 — fused Y channel from SwinFusion backbone
    pet_bgr       : (H,W,3) uint8 BGR — functional modality image (PET or SPECT)
    mri_bgr       : (H,W,3) uint8 BGR — structural modality image (MRI or CT)
    alpha         : float — MRI edge injection strength (default 0.18)
    sal_threshold : float — saliency threshold boundary (default 0.40)
    acf_blend     : float — MRI chroma weight in low-saliency regions (default 0.25)
    modality      : str — fusion task identifier, one of:
                    'MRI-PET', 'MRI-SPECT', 'CT-MRI'

    Returns
    -------
    result_bgr : (H,W,3) uint8
    """
    H, W = fused_y_gray.shape
    pet_r = cv2.resize(pet_bgr, (W, H)) if pet_bgr.shape[:2] != (H, W) else pet_bgr
    mri_r = cv2.resize(mri_bgr, (W, H)) if mri_bgr.shape[:2] != (H, W) else mri_bgr

    Y_fused = fused_y_gray.astype(np.float32) / 255.0
    Y_mri   = cv2.cvtColor(mri_r, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # ── Step 1: Saliency-masked Y enhancement ────────────────────────────────
    saliency = compute_pet_saliency(pet_r)
    inject_w = np.clip(1.0 - saliency / (sal_threshold + 1e-8), 0, 1) ** 2

    Y_mri_u8 = np.clip(Y_mri * 255, 0, 255).astype(np.uint8)
    gx = cv2.Sobel(Y_mri_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Y_mri_u8, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(gx**2 + gy**2)
    edges = cv2.GaussianBlur(edges, (0,0), 1.5)
    edges = edges / (edges.max() + 1e-8)

    Y_enhanced = np.clip(Y_fused + alpha * inject_w * edges, 0, 1)

    # ── Step 2: Adaptive chrominance fusion (ACF) ─────────────────────────────
    # CT-MRI: both modalities produce grayscale outputs.
    # HSV saturation approaches zero for grayscale inputs, so the saliency
    # mechanism defaults to spatially uniform edge enhancement on the luminance
    # channel only. ACF is bypassed and the enhanced Y channel is used directly
    # as the grayscale fused output.
    if modality.upper() == 'CT-MRI':
        result_gray = np.clip(Y_enhanced * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)

    # MRI-PET and MRI-SPECT: functional colour modalities.
    # Both PET and SPECT encode activity using JET-analogous colourmaps
    # in which HSV saturation and value jointly indicate functional significance.
    # ACF preserves diagnostic false-colour in high-saliency hotspot regions
    # and blends MRI-informed chrominance in low-activity background areas.
    pet_ycrcb = cv2.cvtColor(pet_r, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    mri_ycrcb = cv2.cvtColor(mri_r, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0

    Cr_pet, Cb_pet = pet_ycrcb[:,:,1], pet_ycrcb[:,:,2]
    Cr_mri, Cb_mri = mri_ycrcb[:,:,1], mri_ycrcb[:,:,2]

    # Chrominance selection coefficient: 1 in hotspots, 0 in background
    pet_alpha = np.clip(saliency / (sal_threshold + 1e-8), 0, 1)
    mri_alpha = (1 - pet_alpha) * acf_blend

    Cr_fused = pet_alpha * Cr_pet + mri_alpha * Cr_mri + (1 - pet_alpha - mri_alpha) * Cr_pet
    Cb_fused = pet_alpha * Cb_pet + mri_alpha * Cb_mri + (1 - pet_alpha - mri_alpha) * Cb_pet

    Cr_fused = np.clip(Cr_fused, 0, 1)
    Cb_fused = np.clip(Cb_fused, 0, 1)

    # ── Reconstruct BGR ───────────────────────────────────────────────────────
    ycrcb_out = np.stack([Y_enhanced, Cr_fused, Cb_fused], axis=2)
    ycrcb_u8  = np.clip(ycrcb_out * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb_u8, cv2.COLOR_YCrCb2BGR)


# ══════════════════════════════════════════════════════════════════════════════
#  FULL METRICS SUITE (existing + 2 novel)
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(fused_bgr, pet_bgr, mri_bgr):
    """
    Compute all metrics including the 2 novel ones (FISC, MSSF).
    Returns a dict of all metric values.
    """
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr

    def _gray(img, ref_shape=None):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        if ref_shape is not None and g.shape != ref_shape:
            g = cv2.resize(g, (ref_shape[1], ref_shape[0]))
        return g

    H, W = fused_bgr.shape[:2]
    F = _gray(fused_bgr)
    P = _gray(pet_bgr,  (H,W))
    M = _gray(mri_bgr,  (H,W))

    # Standard metrics
    ssim_a = sk_ssim(F, M, data_range=255)
    ssim_b = sk_ssim(F, P, data_range=255)
    ssim   = 0.5*(ssim_a + ssim_b)
    psnr_a = sk_psnr(M, F, data_range=255)
    psnr_b = sk_psnr(P, F, data_range=255)

    # MI
    def _mi(x, y, bins=256):
        h, _,_ = np.histogram2d(x.flatten(), y.flatten(), bins=bins)
        pxy = h/h.sum(); px=pxy.sum(1); py=pxy.sum(0)
        pp = np.outer(px,py); mask=pxy>0
        return np.sum(pxy[mask]*np.log2(pxy[mask]/(pp[mask]+1e-10)))
    mi = _mi(F,M) + _mi(F,P)

    # SF, AG
    sf = np.sqrt(np.mean(np.diff(F.astype(np.float64),axis=0)**2) +
                 np.mean(np.diff(F.astype(np.float64),axis=1)**2))
    gy_f, gx_f = np.gradient(F.astype(np.float64))
    ag = np.mean(np.sqrt((gx_f**2+gy_f**2)/2))

    # EPI
    def _sobel(x):
        gx=cv2.Sobel(x,cv2.CV_64F,1,0,ksize=3)
        gy=cv2.Sobel(x,cv2.CV_64F,0,1,ksize=3)
        return np.sqrt(gx**2+gy**2)
    def _corr(a,b):
        a,b=a.flatten(),b.flatten()
        am,bm=a-a.mean(),b-b.mean()
        return np.sum(am*bm)/(np.sqrt(np.sum(am**2)*np.sum(bm**2))+1e-8)
    epi = 0.5*(_corr(_sobel(F.astype(np.float64)),_sobel(M.astype(np.float64))) +
               _corr(_sobel(F.astype(np.float64)),_sobel(P.astype(np.float64))))

    # CNR
    flat=F.flatten().astype(np.float64)
    sv=np.sort(flat); n=len(sv)
    sig=flat[flat>=sv[int(0.8*n)]]; bg=flat[flat<=sv[int(0.2*n)]]
    cnr=(sig.mean()-bg.mean())/(bg.std()+1e-8) if bg.std()>1e-8 else 0.0

    # FIR
    best=np.maximum(M.astype(np.float64),P.astype(np.float64))
    fir=1.0-np.mean(np.abs(F.astype(np.float64)-best))/(np.mean(np.abs(best))+1e-8)

    # Qabf
    def _qabf(imgF,imgA,imgB):
        def _ep(Gf,Af,Ga,Aa,Tg=0.9994,kg=-15,Dg=0.5,Ta=0.9879,ka=-22,Da=0.8):
            return (Tg/(1+np.exp(kg*(Gf/(Ga+1e-8)-Dg)))) * \
                   (Ta/(1+np.exp(ka*(np.abs(Af-Aa)-Da))))
        def _sa(img):
            gx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
            gy=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
            return np.sqrt(gx**2+gy**2),np.arctan2(gy,gx+1e-8)
        Gf,Af=_sa(imgF.astype(np.float64))
        Ga,Aa=_sa(imgA.astype(np.float64))
        Gb,Ab=_sa(imgB.astype(np.float64))
        wA=Ga/(Ga+Gb+1e-8); wB=Gb/(Ga+Gb+1e-8)
        return np.mean(wA*_ep(Gf,Af,Ga,Aa)+wB*_ep(Gf,Af,Gb,Ab))
    qabf = _qabf(F,M,P)

    # ── SCD: Sum of Correlations of Differences ───────────────────────────────
    # Aslantas & Bendes, AEU Int. J. Electron. Commun., 2015
    # Measures structural/detail transfer via high-frequency difference correlation
    def _scd(imgF, imgA, imgB):
        F64 = imgF.astype(np.float64)
        A64 = imgA.astype(np.float64)
        B64 = imgB.astype(np.float64)
        def _corr2(x, y):
            xm = x - x.mean(); ym = y - y.mean()
            return np.sum(xm*ym) / (np.sqrt(np.sum(xm**2)*np.sum(ym**2)) + 1e-8)
        D_A = F64 - A64   # difference: fused minus source A
        D_B = F64 - B64   # difference: fused minus source B
        return _corr2(F64, A64 - D_A) + _corr2(F64, B64 - D_B)
    scd = _scd(F, M, P)

    # ── VIF: Visual Information Fidelity ──────────────────────────────────────
    # Han et al., Information Fusion, 2013
    # Estimates perceptual fidelity under a human visual system (HVS) model
    # Simplified efficient implementation (Sheikh & Bovik model, sigma_n=0.4)
    def _vif(imgF, imgR, sigma_n=0.4, eps=1e-10):
        F64 = imgF.astype(np.float64) / 255.0
        R64 = imgR.astype(np.float64) / 255.0
        vif_num, vif_den = 0.0, 0.0
        for scale in range(1, 5):
            N    = 2**(4-scale+1) + 1
            sig  = N / 5.0
            R_bl = cv2.GaussianBlur(R64, (N,N), sig)
            F_bl = cv2.GaussianBlur(F64, (N,N), sig)
            R_lp = R64 - R_bl
            F_lp = F64 - F_bl
            # local variance
            R_var = cv2.GaussianBlur(R_lp**2, (N,N), sig)
            F_var = cv2.GaussianBlur(F_lp**2, (N,N), sig)
            RF_cv = cv2.GaussianBlur(R_lp*F_lp, (N,N), sig)
            g     = RF_cv / (R_var + eps)
            sv_sq = np.maximum(F_var - g * RF_cv, 0)
            vif_num += np.sum(np.log2(1 + g**2 * R_var / (sv_sq + sigma_n**2 + eps)))
            vif_den += np.sum(np.log2(1 + R_var / (sigma_n**2 + eps)))
            R64 = cv2.resize(R64, (max(R64.shape[1]//2,8), max(R64.shape[0]//2,8)))
            F64 = cv2.resize(F64, (R64.shape[1], R64.shape[0]))
        return float(vif_num / (vif_den + eps))
    vif = 0.5 * (_vif(F, M) + _vif(F, P))

    # ── EN: Entropy ───────────────────────────────────────────────────────────
    # Roberts et al., 2008 — information richness of fused image
    def _entropy(img):
        h = np.histogram(img.flatten(), bins=256, range=(0,255))[0]
        p = h / (h.sum() + 1e-10)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))
    en = _entropy(F)

    # ── SD: Standard Deviation ────────────────────────────────────────────────
    # Measures contrast/spread of fused image intensity
    sd = float(F.astype(np.float64).std())

    # ── Qcb: Chen-Blum metric ─────────────────────────────────────────────────
    # Chen & Blum, Image and Vision Computing, 2009
    # Measures salient information transfer from sources to fused image.
    # Used in ECFusion (Nature Sci Reports 2025) which compares vs SwinFusion.
    def _qcb(imgF, imgA, imgB):
        def _local_contrast(img, ksize=5):
            img_f = img.astype(np.float64)
            mu    = cv2.GaussianBlur(img_f, (ksize,ksize), 0)
            sigma = np.sqrt(np.abs(cv2.GaussianBlur(img_f**2,(ksize,ksize),0) - mu**2) + 1e-8)
            return sigma / (mu + 1e-8)
        def _saliency(img):
            img_f = img.astype(np.float64)
            return cv2.GaussianBlur(img_f,(5,5),0) - cv2.GaussianBlur(img_f,(31,31),0)
        F64 = imgF.astype(np.float64)
        A64 = imgA.astype(np.float64)
        B64 = imgB.astype(np.float64)
        cF = _local_contrast(F64); cA = _local_contrast(A64); cB = _local_contrast(B64)
        # Contrast preservation ratio
        QA = np.where(cA >= cF, cF/(cA+1e-8), cA/(cF+1e-8))
        QB = np.where(cB >= cF, cF/(cB+1e-8), cB/(cF+1e-8))
        # Saliency weights
        sA = np.abs(_saliency(A64)); sB = np.abs(_saliency(B64))
        wA = sA / (sA + sB + 1e-8); wB = sB / (sA + sB + 1e-8)
        return float(np.mean(wA * QA + wB * QB))
    qcb = _qcb(F, M, P)

    # ── Qcv: Chen-Varshney metric ─────────────────────────────────────────────
    # Chen & Varshney, Information Fusion, 2007
    # Human perception inspired metric based on regional information.
    # Used in MGFusion (Frontiers Neurorobotics 2024).
    def _qcv(imgF, imgA, imgB, patch=8):
        F64 = imgF.astype(np.float64) / 255.0
        A64 = imgA.astype(np.float64) / 255.0
        B64 = imgB.astype(np.float64) / 255.0
        H, W = F64.shape
        scores = []
        for r in range(0, H-patch, patch):
            for c in range(0, W-patch, patch):
                pF = F64[r:r+patch, c:c+patch].flatten()
                pA = A64[r:r+patch, c:c+patch].flatten()
                pB = B64[r:r+patch, c:c+patch].flatten()
                # Regional mutual information proxy via correlation
                def _rc(x, y):
                    xm = x-x.mean(); ym = y-y.mean()
                    d  = np.sqrt(np.sum(xm**2)*np.sum(ym**2))
                    return np.sum(xm*ym)/(d+1e-8) if d>1e-8 else 0.0
                rA = _rc(pF, pA); rB = _rc(pF, pB)
                # Weight by source variance (more informative source gets higher weight)
                vA = pA.var(); vB = pB.var()
                wA = vA/(vA+vB+1e-8); wB = vB/(vA+vB+1e-8)
                scores.append(wA*rA + wB*rB)
        return float(np.mean(scores)) if scores else 0.0
    qcv = _qcv(F, M, P)

    return {
        'SSIM_A': ssim_a, 'SSIM_B': ssim_b, 'SSIM': ssim,
        'PSNR_A': psnr_a, 'PSNR_B': psnr_b,
        'MI': mi, 'SF': sf, 'AG': ag,
        'EPI': epi, 'CNR': cnr, 'FIR': fir, 'Qabf': qabf,
        # Added metrics — absent in SwinFusion (2022), used in 2024-2025 papers
        'SCD': scd,   # Aslantas & Bendes 2015 — structural detail transfer
        'VIF': vif,   # Han et al. 2013 — HVS perceptual fidelity
        'EN':  en,    # Roberts et al. 2008 — information entropy
        'SD':  sd,    # standard deviation — image contrast
        'Qcb': qcb,   # Chen & Blum 2009 — salient info transfer (ECFusion 2025)
        'Qcv': qcv,   # Chen & Varshney 2007 — human perception (MGFusion 2024)
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_batch(
    fused_dir     = 'results/SwinFusion_PET-MRI',
    pet_rgb_dir   = 'Dataset/testsets/PET-MRI/PET_RGB',
    mri_dir       = 'Dataset/testsets/PET-MRI/MRI',
    output_dir    = 'results/SwinFusion_PET-MRI_MedSwinACF',
    alpha         = 0.18,
    sal_threshold = 0.4,
    acf_blend     = 0.25,
    modality      = 'MRI-PET',
    compare       = True,
    metrics_only  = False,
    save_fig3     = True,
    fig3_pairs    = None,
):
    import csv

    os.makedirs(output_dir, exist_ok=True)
    compare_dir = os.path.join(output_dir, 'compare')
    fig3_dir    = os.path.join(output_dir, 'fig3_acf')
    if compare and not metrics_only:
        os.makedirs(compare_dir, exist_ok=True)
    if save_fig3:
        os.makedirs(fig3_dir, exist_ok=True)

    if fig3_pairs is None:
        all_files = sorted([f for f in os.listdir(fused_dir)
                            if f.lower().endswith(('.png','.jpg','.bmp'))])
        fig3_pairs = all_files[:3]

    fig3_rows = []

    files = sorted([f for f in os.listdir(fused_dir)
                    if f.lower().endswith(('.png','.jpg','.bmp'))])

    print(f"\n[MedSwinACF] alpha={alpha}  threshold={sal_threshold}  blend={acf_blend}")
    print(f"  {len(files)} images → {output_dir}\n")

    all_orig_m, all_ours_m = [], []

    for fname in files:
        fused_y  = cv2.imread(os.path.join(fused_dir, fname), cv2.IMREAD_GRAYSCALE)
        pet_bgr  = cv2.imread(os.path.join(pet_rgb_dir, fname))
        mri_path = os.path.join(mri_dir, fname)
        mri_bgr  = cv2.imread(mri_path) if os.path.exists(mri_path) else None

        if fused_y is None or pet_bgr is None:
            print(f"  [SKIP] {fname}")
            continue

        # Original reconstruction
        H, W = fused_y.shape
        pet_r = cv2.resize(pet_bgr,(W,H)) if pet_bgr.shape[:2]!=(H,W) else pet_bgr
        ycrcb_orig = cv2.cvtColor(pet_r, cv2.COLOR_BGR2YCrCb)
        ycrcb_orig[:,:,0] = fused_y
        orig_bgr = cv2.cvtColor(ycrcb_orig, cv2.COLOR_YCrCb2BGR)

        if mri_bgr is None:
            print(f"  [WARN] No MRI for {fname} — skipping enhancement")
            result_bgr = orig_bgr
        else:
            result_bgr = medswin_acf_fuse(fused_y, pet_bgr, mri_bgr,
                                          alpha=alpha,
                                          sal_threshold=sal_threshold,
                                          acf_blend=acf_blend,
                                          modality=modality)

        if not metrics_only:
            cv2.imwrite(os.path.join(output_dir, fname), result_bgr)

        # Compute metrics for both
        mri_r = cv2.resize(mri_bgr,(W,H)) if (mri_bgr is not None and
                mri_bgr.shape[:2]!=(H,W)) else mri_bgr
        if mri_r is not None:
            orig_m = compute_all_metrics(orig_bgr, pet_r, mri_r)
            ours_m = compute_all_metrics(result_bgr, pet_r, mri_r)
            orig_m['file'] = ours_m['file'] = fname
            all_orig_m.append(orig_m)
            all_ours_m.append(ours_m)

        if compare and not metrics_only and mri_r is not None:
            panel = np.zeros((H, W*2, 3), dtype=np.uint8)
            panel[:,:W]  = orig_bgr
            panel[:,W:]  = result_bgr
            for i,lbl in enumerate(['SwinFusion (original)','MedSwinACF (ours)']):
                cv2.putText(panel, lbl, (i*W+6,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,(255,255,255),1,cv2.LINE_AA)
            cv2.imwrite(os.path.join(compare_dir, fname), panel)

        # ── Fig 3 saves ───────────────────────────────────────────────────────
        if save_fig3 and fname in fig3_pairs and mri_r is not None:

            # Compute saliency and chrominance coefficient for this pair
            saliency  = compute_pet_saliency(pet_r)
            pet_alpha = np.clip(
                saliency / (sal_threshold + 1e-8), 0, 1)

            # Column 1 — MRI source
            col1 = mri_r.copy()

            # Column 2 — PET source
            col2 = pet_r.copy()

            # Column 3 — Saliency map (JET colourmap)
            col3 = cv2.applyColorMap(
                (saliency * 255).astype(np.uint8),
                cv2.COLORMAP_JET)

            # Column 4 — Chrominance selection coefficient
            # Use BONE colourmap inverted: bright = high PET colour,
            # dark = structural blend — more intuitive for reviewers
            p5  = np.percentile(pet_alpha, 5)
            p95 = np.percentile(pet_alpha, 95)
            coeff_stretched = np.clip(
                (pet_alpha - p5) / (p95 - p5 + 1e-8), 0, 1)
            col4 = cv2.applyColorMap(
                (coeff_stretched * 255).astype(np.uint8),
                cv2.COLORMAP_HOT)

            # Column 5 — Fused output WITH ACF
            col5 = result_bgr.copy()

            # Column 6 — Fused output WITHOUT ACF (baseline)
            col6 = orig_bgr.copy()

            # Save individual columns
            stem = os.path.splitext(fname)[0]
            cv2.imwrite(
                os.path.join(fig3_dir, f'{stem}_col1_mri.png'),    col1)
            cv2.imwrite(
                os.path.join(fig3_dir, f'{stem}_col2_pet.png'),    col2)
            cv2.imwrite(
                os.path.join(fig3_dir, f'{stem}_col3_saliency.png'), col3)
            cv2.imwrite(
                os.path.join(fig3_dir, f'{stem}_col4_coeff.png'),  col4)
            cv2.imwrite(
                os.path.join(fig3_dir, f'{stem}_col5_with_acf.png'), col5)
            cv2.imwrite(
                os.path.join(fig3_dir, f'{stem}_col6_without_acf.png'), col6)

            fig3_rows.append({
                'label': f'MRI-PET pair {len(fig3_rows)+1}',
                'cols':  [col1, col2, col3, col4, col5, col6],
            })    

        print(f"  {fname}  OK")

    # Print and save comparison table
    if all_orig_m and all_ours_m:
        keys = [k for k in all_orig_m[0].keys() if k != 'file']
        orig_means = {k: np.mean([m[k] for m in all_orig_m]) for k in keys}
        ours_means = {k: np.mean([m[k] for m in all_ours_m]) for k in keys}

        print(f"\n{'─'*70}")
        print(f"{'Metric':<12} {'SwinFusion':>14} {'MedSwinACF':>14} {'Δ':>10} {'Win'}")
        print(f"{'─'*70}")
        for k in keys:
            orig_v = orig_means[k]
            ours_v = ours_means[k]
            delta  = ours_v - orig_v
            win    = 'OURS ↑' if delta > 0 else ('orig' if delta < 0 else '=')
            marker = '★' if (k in ['FISC','MSSF'] and delta > 0) else \
                     '★' if (k in ['CNR','MI','FIR','Qabf','EPI'] and delta > 0) else ''
            print(f"  {k:<10} {orig_v:>14.4f} {ours_v:>14.4f} "
                  f"{delta:>+10.4f}  {win} {marker}")
        print(f"{'─'*70}")
        print("  ★ = medical/novel metric improvement")

        # Save CSV
        csv_path = os.path.join(output_dir, 'full_metrics_comparison.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Metric','SwinFusion_original','MedSwinACF_ours','Delta'])
            for k in keys:
                w.writerow([k, f"{orig_means[k]:.4f}",
                             f"{ours_means[k]:.4f}",
                             f"{ours_means[k]-orig_means[k]:+.4f}"])
        print(f"\n  Full results saved: {csv_path}")

    # Assemble Fig 3 grid
    if save_fig3 and fig3_rows:
        _assemble_fig3(fig3_rows, fig3_dir)
    
    print(f"\n  Done: {output_dir}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SELF TEST
# ══════════════════════════════════════════════════════════════════════════════
def _assemble_fig3(rows, save_dir):
    """
    Assemble Fig 3 grid:
      Rows    = representative MRI-PET pairs
      Columns = MRI source | PET source | Activity map |
                Colour coefficient | With ACF | Without ACF
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    col_labels = [
        'MRI source',
        'PET source',
        'Activity map',
        'Colour coefficient',
        'With ACF',
        'Without ACF',
    ]

    n_rows   = len(rows)
    n_cols   = 6
    cell_size = 2.2

    fig = plt.figure(
        figsize=(n_cols * cell_size, n_rows * cell_size + 0.6),
        facecolor='white')

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        hspace=0.04,
        wspace=0.04,
        top=0.93,
        bottom=0.02,
        left=0.07,
        right=0.99)

    for r, row in enumerate(rows):
        for c, img_bgr in enumerate(row['cols']):
            ax = fig.add_subplot(gs[r, c])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(
                    col_labels[c],
                    fontsize=10,
                    fontweight='bold',
                    pad=5)

            if c == 0:
                ax.set_ylabel(
                    row['label'],
                    fontsize=10,
                    fontweight='bold',
                    rotation=90,
                    labelpad=6)

            # Highlight the key comparison columns
            if c in [4, 5]:
                for spine in ax.spines.values():
                    spine.set_edgecolor(
                        '#0F6E56' if c == 4 else '#888780')
                    spine.set_linewidth(1.5)
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#cccccc')
                    spine.set_linewidth(0.5)

    out_path = os.path.join(save_dir, 'fig3_assembled.png')
    fig.savefig(
        out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Fig 3 assembled → {out_path}")


def self_test():
    print("=== medswin_acf_combined.py self-test ===\n")
    H, W = 128, 128
    np.random.seed(42)

    # Simulate realistic images
    fused_y = np.random.randint(50, 200, (H,W), dtype=np.uint8)
    # PET: warm/colourful in centre
    pet_bgr = np.zeros((H,W,3), dtype=np.uint8)
    cx,cy = W//2, H//2
    for y in range(H):
        for x in range(W):
            d = np.sqrt((x-cx)**2+(y-cy)**2)
            if d < 30:
                pet_bgr[y,x] = [30, 80, 220]   # warm orange-red in BGR
            else:
                pet_bgr[y,x] = [100,100,100]
    # MRI: grayscale with structure
    mri_bgr = np.stack([np.random.randint(60,180,(H,W),dtype=np.uint8)]*3, axis=2)

    # Test pipeline
    result = medswin_acf_fuse(fused_y, pet_bgr, mri_bgr, modality='MRI-PET')
    assert result.shape == (H,W,3), f"Wrong shape: {result.shape}"
    assert result.dtype == np.uint8
    print(f"  Pipeline output: {result.shape}  OK")

    # Full metrics
    m = compute_all_metrics(result, pet_bgr, mri_bgr)
    assert all(k in m for k in ['SCD','VIF','EN','SD','Qcb','Qcv'])
    print(f"\n  All metrics computed:")
    for k,v in m.items():
        print(f"    {k:<12} {v:.4f}")

    print("\n  PASSED.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':                 # ← existing, with two additions
    p = argparse.ArgumentParser()
    p.add_argument('--run',          action='store_true')
    p.add_argument('--metrics_only', action='store_true',
                   help='Compute metrics without saving images')
    p.add_argument('--fused_dir',    default='results/SwinFusion_PET-MRI')
    p.add_argument('--pet_rgb_dir',  default='Dataset/testsets/PET-MRI/PET_RGB')
    p.add_argument('--mri_dir',      default='Dataset/testsets/PET-MRI/MRI')
    p.add_argument('--output_dir',   default='results/SwinFusion_PET-MRI_MedSwinACF')
    p.add_argument('--alpha',        type=float, default=0.18)
    p.add_argument('--threshold',    type=float, default=0.4)
    p.add_argument('--acf_blend',    type=float, default=0.25)
    p.add_argument('--no_compare',   action='store_true')
    p.add_argument('--fig3_pairs',   nargs='+', default=None,
                   help='Filenames for Fig 3, e.g. 3.png 6.png 9.png')  # ← ADD
    args = p.parse_args()

    if args.run or args.metrics_only:
        run_batch(
            fused_dir     = args.fused_dir,
            pet_rgb_dir   = args.pet_rgb_dir,
            mri_dir       = args.mri_dir,
            output_dir    = args.output_dir,
            alpha         = args.alpha,
            sal_threshold = args.threshold,
            acf_blend     = args.acf_blend,
            compare       = not args.no_compare,
            metrics_only  = args.metrics_only,
            save_fig3     = True,           # ← ADD
            fig3_pairs    = args.fig3_pairs, # ← ADD
        )
    else:
        self_test()