"""
run_ct_spect_metrics.py
========================
Runs MedSwinACF pipeline + full metrics for CT-MRI and MRI-SPECT.

Key difference from PET-MRI: CT and SPECT are grayscale — no colour
reconstruction. Enhancement is applied only to the Y (luminance) channel.

USAGE:
    # CT-MRI
    python run_ct_spect_metrics.py --modality CT-MRI `
      --fused_dir   results/SwinFusion_CT-MRI `
      --source_A    Dataset/testsets/CT-MRI/MRI `
      --source_B    Dataset/testsets/CT-MRI/CT `
      --output_dir  results/SwinFusion_CT-MRI_MedSwinACF

    # MRI-SPECT
    python run_ct_spect_metrics.py --modality MRI-SPECT `
      --fused_dir   results/SwinFusion_MRI-SPECT `
      --source_A    Dataset/testsets/MRI-SPECT/MRI `
      --source_B    Dataset/testsets/MRI-SPECT/SPECT `
      --output_dir  results/SwinFusion_MRI-SPECT_MedSwinACF
"""

import os
import cv2
import csv
import numpy as np
import argparse
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from scipy.stats import pearsonr


# ── Grayscale MedSwin enhancement (no colour pipeline) ───────────────────────

def compute_pet_saliency_gray(source_b_gray, blur_sigma=5):
    """Activity map from grayscale source B (CT/SPECT intensity)."""
    img = source_b_gray.astype(np.float32) / 255.0
    activity = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    return (activity / (activity.max() + 1e-8)).astype(np.float32)


def medswin_enhance_gray(fused_gray, mri_gray, source_b_gray,
                          alpha=0.18, sal_threshold=0.4):
    """
    Grayscale MedSwin enhancement.
    Injects MRI edges into fused image, masked by source B activity.
    Returns enhanced grayscale image (H,W) uint8.
    """
    H, W = fused_gray.shape
    Y_f  = fused_gray.astype(np.float32) / 255.0
    Y_m  = cv2.resize(mri_gray,      (W, H)).astype(np.float32) / 255.0 \
           if mri_gray.shape[:2] != (H, W) else mri_gray.astype(np.float32) / 255.0
    Y_b  = cv2.resize(source_b_gray, (W, H)).astype(np.float32) / 255.0 \
           if source_b_gray.shape[:2] != (H, W) else source_b_gray.astype(np.float32) / 255.0

    # Saliency from source B
    saliency  = compute_pet_saliency_gray(
        np.clip(Y_b * 255, 0, 255).astype(np.uint8))
    inject_w  = np.clip(1.0 - saliency / (sal_threshold + 1e-8), 0, 1) ** 2

    # MRI edge map
    Y_m_u8 = np.clip(Y_m * 255, 0, 255).astype(np.uint8)
    gx     = cv2.Sobel(Y_m_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy     = cv2.Sobel(Y_m_u8, cv2.CV_32F, 0, 1, ksize=3)
    edges  = np.sqrt(gx**2 + gy**2)
    edges  = cv2.GaussianBlur(edges, (0, 0), 1.5)
    edges  = edges / (edges.max() + 1e-8)

    Y_enhanced = np.clip(Y_f + alpha * inject_w * edges, 0, 1)
    return np.clip(Y_enhanced * 255, 0, 255).astype(np.uint8)


# ── Full metrics (grayscale version) ─────────────────────────────────────────

def compute_metrics_gray(imgF, imgA, imgB):
    """
    All metrics for grayscale fused image.
    imgF, imgA, imgB: (H,W) uint8 grayscale
    """
    H, W = imgF.shape
    if imgA.shape != (H, W): imgA = cv2.resize(imgA, (W, H))
    if imgB.shape != (H, W): imgB = cv2.resize(imgB, (W, H))

    F = imgF.astype(np.float64)
    A = imgA.astype(np.float64)
    B = imgB.astype(np.float64)

    # SSIM
    ssim_a = sk_ssim(imgF, imgA, data_range=255)
    ssim_b = sk_ssim(imgF, imgB, data_range=255)
    ssim   = 0.5 * (ssim_a + ssim_b)

    # PSNR
    psnr_a = sk_psnr(imgA, imgF, data_range=255)
    psnr_b = sk_psnr(imgB, imgF, data_range=255)

    # MI
    def _mi(x, y, bins=256):
        h, _, _ = np.histogram2d(x.flatten(), y.flatten(), bins=bins)
        pxy = h / h.sum(); px = pxy.sum(1); py = pxy.sum(0)
        pp  = np.outer(px, py); mask = pxy > 0
        return float(np.sum(pxy[mask] * np.log2(pxy[mask] / (pp[mask] + 1e-10))))
    mi = _mi(F, A) + _mi(F, B)

    # SF, AG
    sf = float(np.sqrt(np.mean(np.diff(F, axis=0)**2) +
                       np.mean(np.diff(F, axis=1)**2)))
    gy_f, gx_f = np.gradient(F)
    ag = float(np.mean(np.sqrt((gx_f**2 + gy_f**2) / 2)))

    # EPI
    def _sobel(x):
        gx = cv2.Sobel(x.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(x.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2)
    def _corr(a, b):
        am = a.flatten() - a.mean(); bm = b.flatten() - b.mean()
        return float(np.sum(am*bm) / (np.sqrt(np.sum(am**2)*np.sum(bm**2)) + 1e-8))
    epi = 0.5 * (_corr(_sobel(F), _sobel(A)) + _corr(_sobel(F), _sobel(B)))

    # CNR
    flat = F.flatten(); sv = np.sort(flat); n = len(sv)
    sig  = flat[flat >= sv[int(0.8*n)]]; bg = flat[flat <= sv[int(0.2*n)]]
    cnr  = float((sig.mean() - bg.mean()) / (bg.std() + 1e-8))

    # FIR
    best = np.maximum(A, B)
    fir  = float(1.0 - np.mean(np.abs(F - best)) / (np.mean(np.abs(best)) + 1e-8))

    # Qabf
    def _qabf(f, a, b):
        def _sa(x):
            gx = cv2.Sobel(x.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(x.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(gx**2+gy**2), np.arctan2(gy, gx+1e-8)
        def _ep(Gf, Af, Ga, Aa):
            Qg = 0.9994/(1+np.exp(-15*(Gf/(Ga+1e-8)-0.5)))
            Qa = 0.9879/(1+np.exp(-22*(np.abs(Af-Aa)-0.8)))
            return Qg*Qa
        Gf,Af=_sa(f); Ga,Aa=_sa(a); Gb,Ab=_sa(b)
        wA=Ga/(Ga+Gb+1e-8); wB=Gb/(Ga+Gb+1e-8)
        return float(np.mean(wA*_ep(Gf,Af,Ga,Aa)+wB*_ep(Gf,Af,Gb,Ab)))
    qabf = _qabf(F, A, B)

    # SCD (Aslantas & Bendes 2015)
    def _scd(f, a, b):
        def _c2(x,y):
            xm=x-x.mean(); ym=y-y.mean()
            return float(np.sum(xm*ym)/(np.sqrt(np.sum(xm**2)*np.sum(ym**2))+1e-8))
        Da=f-a; Db=f-b
        return _c2(f, a-Da) + _c2(f, b-Db)
    scd = _scd(F, A, B)

    # VIF simplified (Han et al. 2013)
    def _vif(f, r, sigma_n=0.4, eps=1e-10):
        f64=f/255.0; r64=r/255.0; vn=0.0; vd=0.0
        for s in range(1, 5):
            N=2**(4-s+1)+1; sig=N/5.0
            rb=cv2.GaussianBlur(r64.astype(np.float32),(N,N),sig).astype(np.float64)
            fb=cv2.GaussianBlur(f64.astype(np.float32),(N,N),sig).astype(np.float64)
            rl=r64-rb; fl=f64-fb
            rv=cv2.GaussianBlur((rl**2).astype(np.float32),(N,N),sig).astype(np.float64)
            fv=cv2.GaussianBlur((fl**2).astype(np.float32),(N,N),sig).astype(np.float64)
            rc=cv2.GaussianBlur((rl*fl).astype(np.float32),(N,N),sig).astype(np.float64)
            g=rc/(rv+eps); sv2=np.maximum(fv-g*rc,0)
            vn+=np.sum(np.log2(1+g**2*rv/(sv2+sigma_n**2+eps)))
            vd+=np.sum(np.log2(1+rv/(sigma_n**2+eps)))
            r64=cv2.resize(r64.astype(np.float32),(max(r64.shape[1]//2,8),max(r64.shape[0]//2,8))).astype(np.float64)
            f64=cv2.resize(f64.astype(np.float32),(r64.shape[1],r64.shape[0])).astype(np.float64)
        return float(vn/(vd+eps))
    vif = 0.5*(_vif(imgF,imgA) + _vif(imgF,imgB))

    # EN, SD
    h = np.histogram(imgF.flatten(), bins=256, range=(0,255))[0]
    p = h/(h.sum()+1e-10); p=p[p>0]
    en = float(-np.sum(p*np.log2(p)))
    sd = float(F.std())

    # Qcb (Chen & Blum 2009)
    def _qcb(f,a,b):
        def _lc(x,k=5):
            xf=x.astype(np.float64)
            mu=cv2.GaussianBlur(xf.astype(np.float32),(k,k),0).astype(np.float64)
            sig=np.sqrt(np.abs(cv2.GaussianBlur((xf**2).astype(np.float32),(k,k),0).astype(np.float64)-mu**2)+1e-8)
            return sig/(mu+1e-8)
        def _sal(x):
            xf=x.astype(np.float64)
            return cv2.GaussianBlur(xf.astype(np.float32),(5,5),0).astype(np.float64) - \
                   cv2.GaussianBlur(xf.astype(np.float32),(31,31),0).astype(np.float64)
        cF=_lc(f); cA=_lc(a); cB=_lc(b)
        QA=np.where(cA>=cF,cF/(cA+1e-8),cA/(cF+1e-8))
        QB=np.where(cB>=cF,cF/(cB+1e-8),cB/(cF+1e-8))
        sA=np.abs(_sal(a)); sB=np.abs(_sal(b))
        wA=sA/(sA+sB+1e-8); wB=sB/(sA+sB+1e-8)
        return float(np.mean(wA*QA+wB*QB))
    qcb = _qcb(F,A,B)

    # Qcv (Chen & Varshney 2007)
    def _qcv(f,a,b,patch=8):
        f64=f/255.0; a64=a/255.0; b64=b/255.0
        H2,W2=f64.shape; scores=[]
        for r in range(0,H2-patch,patch):
            for c in range(0,W2-patch,patch):
                pF=f64[r:r+patch,c:c+patch].flatten()
                pA=a64[r:r+patch,c:c+patch].flatten()
                pB=b64[r:r+patch,c:c+patch].flatten()
                def _rc(x,y):
                    xm=x-x.mean();ym=y-y.mean();d=np.sqrt(np.sum(xm**2)*np.sum(ym**2))
                    return float(np.sum(xm*ym)/(d+1e-8)) if d>1e-8 else 0.0
                rA=_rc(pF,pA); rB=_rc(pF,pB)
                vA=pA.var(); vB=pB.var()
                wA=vA/(vA+vB+1e-8); wB=vB/(vA+vB+1e-8)
                scores.append(wA*rA+wB*rB)
        return float(np.mean(scores)) if scores else 0.0
    qcv = _qcv(F,A,B)

    return {
        'SSIM_A':ssim_a,'SSIM_B':ssim_b,'SSIM':ssim,
        'PSNR_A':psnr_a,'PSNR_B':psnr_b,
        'MI':mi,'SF':sf,'AG':ag,'EPI':epi,
        'CNR':cnr,'FIR':fir,'Qabf':qabf,
        'SCD':scd,'VIF':vif,'EN':en,'SD':sd,
        'Qcb':qcb,'Qcv':qcv,
    }


# ── Batch runner ──────────────────────────────────────────────────────────────

def run(fused_dir, source_a_dir, source_b_dir, output_dir,
        alpha=0.18, sal_threshold=0.4, modality='CT-MRI'):

    os.makedirs(output_dir, exist_ok=True)
    compare_dir = os.path.join(output_dir, 'compare')
    os.makedirs(compare_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(fused_dir)
                    if f.lower().endswith(('.png','.jpg','.bmp'))])

    print(f"\n[{modality}] MedSwinACF  {len(files)} images")
    print(f"  fused    : {fused_dir}")
    print(f"  source_A : {source_a_dir}  (MRI)")
    print(f"  source_B : {source_b_dir}  (CT/SPECT)")
    print(f"  output   : {output_dir}\n")

    all_orig, all_ours = [], []

    for fname in files:
        fused = cv2.imread(os.path.join(fused_dir, fname), cv2.IMREAD_GRAYSCALE)
        mri   = cv2.imread(os.path.join(source_a_dir, fname), cv2.IMREAD_GRAYSCALE)
        b_path = os.path.join(source_b_dir, fname)
        srcB  = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(b_path) else None

        if fused is None or mri is None:
            print(f"  [SKIP] {fname}")
            continue

        srcB_use = srcB if srcB is not None else fused

        # Enhanced output
        enhanced = medswin_enhance_gray(fused, mri, srcB_use,
                                        alpha=alpha,
                                        sal_threshold=sal_threshold)
        cv2.imwrite(os.path.join(output_dir, fname), enhanced)

        # Compare panel
        H, W = fused.shape
        panel = np.zeros((H, W*2), dtype=np.uint8)
        panel[:, :W]  = fused
        panel[:, W:]  = enhanced
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        for i, lbl in enumerate([f'SwinFusion ({modality})', 'MedSwinACF (ours)']):
            cv2.putText(panel_bgr, lbl, (i*W+6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(compare_dir, fname), panel_bgr)

        # Metrics
        orig_m = compute_metrics_gray(fused,    mri, srcB_use)
        ours_m = compute_metrics_gray(enhanced, mri, srcB_use)
        orig_m['file'] = ours_m['file'] = fname
        all_orig.append(orig_m); all_ours.append(ours_m)
        print(f"  {fname}  CNR: {orig_m['CNR']:.1f} → {ours_m['CNR']:.1f}  "
              f"Qabf: {orig_m['Qabf']:.4f} → {ours_m['Qabf']:.4f}")

    if not all_orig:
        print("  No images processed."); return

    keys = [k for k in all_orig[0] if k != 'file']
    orig_m = {k: np.mean([m[k] for m in all_orig]) for k in keys}
    ours_m = {k: np.mean([m[k] for m in all_ours]) for k in keys}

    print(f"\n{'─'*68}")
    print(f"{'Metric':<10} {'SwinFusion':>14} {'MedSwinACF':>14} {'Δ':>10}  Win")
    print(f"{'─'*68}")
    for k in keys:
        d = ours_m[k] - orig_m[k]
        win = 'OURS ↑ ★' if (d > 0 and k in ['CNR','Qabf','EN','SD','VIF','MI','SCD']) \
              else 'OURS ↑' if d > 0 else 'orig'
        print(f"  {k:<10} {orig_m[k]:>14.4f} {ours_m[k]:>14.4f} {d:>+10.4f}  {win}")
    print(f"{'─'*68}")

    csv_path = os.path.join(output_dir, f'{modality}_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric','SwinFusion','MedSwinACF','Delta'])
        for k in keys:
            w.writerow([k, f"{orig_m[k]:.4f}", f"{ours_m[k]:.4f}",
                        f"{ours_m[k]-orig_m[k]:+.4f}"])
    print(f"\n  Saved: {csv_path}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--modality',  default='CT-MRI', choices=['CT-MRI','MRI-SPECT'])
    p.add_argument('--fused_dir', required=True)
    p.add_argument('--source_A',  required=True, help='MRI folder')
    p.add_argument('--source_B',  required=True, help='CT or SPECT folder')
    p.add_argument('--output_dir',required=True)
    p.add_argument('--alpha',     type=float, default=0.18)
    p.add_argument('--threshold', type=float, default=0.4)
    args = p.parse_args()
    run(args.fused_dir, args.source_A, args.source_B, args.output_dir,
        alpha=args.alpha, sal_threshold=args.threshold, modality=args.modality)
