import cv2, numpy as np, os, sys, pandas as pd
sys.path.insert(0, '.')
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from scipy import stats

def mi(x,y,bins=256):
    h,_,_=np.histogram2d(x.flatten(),y.flatten(),bins=bins)
    pxy=h/h.sum();px=pxy.sum(1);py=pxy.sum(0)
    pp=np.outer(px,py);mask=pxy>0
    return np.sum(pxy[mask]*np.log2(pxy[mask]/(pp[mask]+1e-10)))

def sob(x):
    gx=cv2.Sobel(x.astype(np.float32),cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(x.astype(np.float32),cv2.CV_32F,0,1,ksize=3)
    return np.sqrt(gx**2+gy**2).astype(np.float64)

def corr(a,b):
    a,b=a.flatten(),b.flatten();am,bm=a-a.mean(),b-b.mean()
    return np.sum(am*bm)/(np.sqrt(np.sum(am**2)*np.sum(bm**2))+1e-8)

def cnr(f,p):
    fg=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float64)
    pg=cv2.cvtColor(p,cv2.COLOR_BGR2GRAY).astype(np.float64)
    sm=pg>=np.percentile(pg,80); bm=pg<=np.percentile(pg,20)
    if sm.sum()==0 or bm.sum()==0: return 0.0
    return float(abs(fg[sm].mean()-fg[bm].mean())/max(fg[bm].std(),1.0))

def fir(f,a,b):
    fg=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float64)
    ag=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY).astype(np.float64)
    bg=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY).astype(np.float64)
    best=np.maximum(ag,bg)
    return 1.0-np.mean(np.abs(fg-best))/(np.mean(np.abs(best))+1e-8)

def de(f,p):
    try:
        fl=cv2.cvtColor(f.astype(np.uint8),cv2.COLOR_BGR2LAB).astype(np.float32)
        pl=cv2.cvtColor(p.astype(np.uint8),cv2.COLOR_BGR2LAB).astype(np.float32)
        pg=cv2.cvtColor(p.astype(np.uint8),cv2.COLOR_BGR2GRAY)
        h=pg>=np.percentile(pg,80)
        if h.sum()==0: return 0.0
        return float(np.sqrt(np.sum((fl-pl)**2,axis=2))[h].mean())
    except: return 0.0

def get_per_image_metrics(mdir, adir, bdir, exclude=set()):
    files=sorted([f for f in os.listdir(mdir) if f.endswith('.png') and f not in exclude])
    results=[]
    for fname in files:
        F=cv2.imread(os.path.join(mdir,fname))
        A=cv2.imread(os.path.join(adir,fname))
        B=cv2.imread(os.path.join(bdir,fname))
        if any(x is None for x in [F,A,B]): continue
        F=cv2.resize(F,(256,256)); A=cv2.resize(A,(256,256)); B=cv2.resize(B,(256,256))
        def g(img):
            return (cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img).astype(np.float64)
        Fg=g(F);Ag=g(A);Bg=g(B)
        ssim_val=0.5*(sk_ssim(Fg.astype(np.uint8),Ag.astype(np.uint8),data_range=255)+
                      sk_ssim(Fg.astype(np.uint8),Bg.astype(np.uint8),data_range=255))
        results.append({
            'file': fname,
            'ssim': ssim_val,
            'psnr': sk_psnr(Ag,Fg,data_range=255),
            'mi':   mi(Fg,Ag)+mi(Fg,Bg),
            'epi':  0.5*(corr(sob(Fg),sob(Ag))+corr(sob(Fg),sob(Bg))),
            'cnr':  cnr(F,B),
            'fir':  fir(F,A,B),
            'de':   de(F,B)
        })
    return pd.DataFrame(results)

def ci95(vals):
    n = len(vals)
    se = np.std(vals, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    mean = np.mean(vals)
    return mean, np.std(vals, ddof=1), mean - t_crit*se, mean + t_crit*se

print('HARVARD MRI-PET STATISTICAL ANALYSIS (N=10)')
print('='*80)

MRI_DIR = 'Dataset/testsets/PET-MRI/MRI'
PET_DIR = 'Dataset/testsets/PET-MRI/PET_RGB'

METHODS_PET = {
    'SalFuseFormer': 'results/SalFuseFormer_FINAL_PET-MRI',
    'CDDFuse':       'comparison_results/CDDFuse/PET-MRI',
    'EMFusion':      'comparison_results/EMFusion/PET-MRI',
    'SwinFusion':    'comparison_results/SwinFusion/PET-MRI',
}

pet_data = {}
for method, mdir in METHODS_PET.items():
    if not os.path.exists(mdir): continue
    df = get_per_image_metrics(mdir, MRI_DIR, PET_DIR)
    pet_data[method] = df
    mean,sd,lo,hi = ci95(df['ssim'].values)
    fir_mean,fir_sd,fir_lo,fir_hi = ci95(df['fir'].values)
    print(f'{method}: SSIM={mean:.4f}±{sd:.4f} 95%CI[{lo:.4f},{hi:.4f}] FIR={fir_mean:.4f}±{fir_sd:.4f} 95%CI[{fir_lo:.4f},{fir_hi:.4f}]')

print()
print('Wilcoxon tests vs SalFuseFormer (MRI-PET):')
sal = pet_data['SalFuseFormer']
for method in ['CDDFuse','EMFusion','SwinFusion']:
    if method not in pet_data: continue
    comp = pet_data[method]
    for metric in ['ssim','fir']:
        stat,p = stats.wilcoxon(sal[metric].values, comp[metric].values)
        print(f'  SalFuse vs {method} {metric.upper()}: p={p:.4f}')

print()
print('HARVARD CT-MRI STATISTICAL ANALYSIS (N=18)')
print('='*80)

CT_DIR  = 'Dataset/testsets/CT-MRI/CT'
MRI_CT  = 'Dataset/testsets/CT-MRI/MRI'
EXCLUDE = {'41022.png','41023.png'}

METHODS_CT = {
    'SalFuseFormer': 'results/SwinFusion_CT-MRI_MedSwinACF_fixed',
    'CDDFuse':       'comparison_results/CDDFuse/CT-MRI',
    'EMFusion':      'comparison_results/EMFusion/CT-MRI',
    'SwinFusion':    'comparison_results/SwinFusion/CT-MRI',
}

ct_data = {}
for method, mdir in METHODS_CT.items():
    if not os.path.exists(mdir): continue
    df = get_per_image_metrics(mdir, CT_DIR, MRI_CT, exclude=EXCLUDE)
    ct_data[method] = df
    mean,sd,lo,hi = ci95(df['ssim'].values)
    print(f'{method}: SSIM={mean:.4f}±{sd:.4f} 95%CI[{lo:.4f},{hi:.4f}]')

print()
print('Wilcoxon tests vs SalFuseFormer (CT-MRI):')
sal_ct = ct_data['SalFuseFormer']
for method in ['CDDFuse','EMFusion','SwinFusion']:
    if method not in ct_data: continue
    comp = ct_data[method]
    stat,p = stats.wilcoxon(sal_ct['ssim'].values, comp['ssim'].values)
    print(f'  SalFuse vs {method} SSIM: p={p:.4f}')

print()
print('HARVARD MRI-SPECT STATISTICAL ANALYSIS (N=20)')
print('='*80)

SPECT_MRI = 'Dataset/testsets/MRI-SPECT/MRI'
SPECT_DIR = 'Dataset/testsets/MRI-SPECT/SPECT'

METHODS_SP = {
    'SalFuseFormer': 'results/SalFuseFormer_SPECT_colour',
    'EMFusion':      'comparison_results/EMFusion/MRI-SPECT',
    'CDDFuse':       'comparison_results/CDDFuse/MRI-SPECT',
    'SwinFusion':    'comparison_results/SwinFusion/MRI-SPECT',
}

sp_data = {}
for method, mdir in METHODS_SP.items():
    if not os.path.exists(mdir): continue
    df = get_per_image_metrics(mdir, SPECT_MRI, SPECT_DIR)
    sp_data[method] = df
    mean,sd,lo,hi = ci95(df['ssim'].values)
    print(f'{method}: SSIM={mean:.4f}±{sd:.4f} 95%CI[{lo:.4f},{hi:.4f}]')

print()
print('Wilcoxon tests vs SalFuseFormer (MRI-SPECT):')
sal_sp = sp_data['SalFuseFormer']
for method in ['EMFusion','CDDFuse','SwinFusion']:
    if method not in sp_data: continue
    comp = sp_data[method]
    stat,p = stats.wilcoxon(sal_sp['ssim'].values, comp['ssim'].values)
    print(f'  SalFuse vs {method} SSIM: p={p:.4f}')

print()
print('Saving per-image data...')
for method, df in pet_data.items():
    df.to_csv(f'harvard_petmri_{method}_perimage.csv', index=False)
print('Done.')
