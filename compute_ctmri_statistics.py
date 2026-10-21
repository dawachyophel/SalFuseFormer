import cv2, numpy as np, os, sys, pandas as pd
sys.path.insert(0, '.')
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from scipy import stats

EXCLUDE = {'41022.png', '41023.png'}

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

def fir_m(f,a,b):
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

CT_DIR  = 'Dataset/testsets/CT-MRI/CT'
MRI_DIR = 'Dataset/testsets/CT-MRI/MRI'

METHODS = {
    'DenseFuse':     'comparison_results/DenseFuse/CT-MRI',
    'FusionGAN':     'comparison_results/FusionGAN/CT-MRI',
    'DDcGAN':        'comparison_results/DDcGAN/CT-MRI',
    'U2Fusion':      'comparison_results/U2Fusion/CT-MRI',
    'EMFusion':      'comparison_results/EMFusion/CT-MRI',
    'SeAFusion':     'comparison_results/SeAFusion/CT-MRI',
    'CDDFuse':       'comparison_results/CDDFuse/CT-MRI',
    'CoCoNet':       'comparison_results/CoCoNet/CT-MRI',
    'MUFusion':      'comparison_results/MUFusion/CT-MRI',
    'SwinFusion':    'comparison_results/SwinFusion/CT-MRI',
    'SalFuseFormer': 'results/SwinFusion_CT-MRI_MedSwinACF_fixed',
}

print('Harvard CT-MRI Per-Image Statistics (N=18, excluding 41022 and 41023):')
print('='*120)
print(f'{"Method":<15} {"SSIM mean±SD":>15} {"PSNR-CT mean±SD":>17} {"PSNR-MRI mean±SD":>18} {"FIR mean±SD":>14} {"DeltaE mean±SD":>16}')
print('-'*120)

all_data = {}
for method, mdir in METHODS.items():
    if not os.path.exists(mdir):
        print(method, 'NOT FOUND'); continue
    files=sorted([f for f in os.listdir(mdir) if f.endswith('.png') and f not in EXCLUDE])
    ssims=[]; pct=[]; pmri=[]; mis=[]; epis=[]; cnrs=[]; firs=[]; des=[]
    for fname in files:
        F=cv2.imread(os.path.join(mdir,fname))
        A=cv2.imread(os.path.join(CT_DIR,fname))
        B=cv2.imread(os.path.join(MRI_DIR,fname))
        if any(x is None for x in [F,A,B]): continue
        F=cv2.resize(F,(256,256)); A=cv2.resize(A,(256,256)); B=cv2.resize(B,(256,256))
        def g(img):
            return (cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img).astype(np.float64)
        Fg=g(F);Ag=g(A);Bg=g(B)
        ssims.append(0.5*(sk_ssim(Fg.astype(np.uint8),Ag.astype(np.uint8),data_range=255)+
                          sk_ssim(Fg.astype(np.uint8),Bg.astype(np.uint8),data_range=255)))
        pct.append(sk_psnr(Ag,Fg,data_range=255))
        pmri.append(sk_psnr(Bg,Fg,data_range=255))
        mis.append(mi(Fg,Ag)+mi(Fg,Bg))
        epis.append(0.5*(corr(sob(Fg),sob(Ag))+corr(sob(Fg),sob(Bg))))
        cnrs.append(cnr(F,B))
        firs.append(fir_m(F,A,B))
        des.append(de(F,B))
    all_data[method]={'ssim':ssims,'pct':pct,'pmri':pmri,'mi':mis,'epi':epis,'cnr':cnrs,'fir':firs,'de':des}
    print(f'{method:<15} {np.mean(ssims):.4f}±{np.std(ssims):.4f}    {np.mean(pct):.2f}±{np.std(pct):.2f}      {np.mean(pmri):.2f}±{np.std(pmri):.2f}       {np.mean(firs):.4f}±{np.std(firs):.4f}   {np.mean(des):.2f}±{np.std(des):.2f}')

print()
print('Wilcoxon signed-rank tests vs SalFuseFormer (CT-MRI):')
sal = all_data['SalFuseFormer']
for method in ['CDDFuse','EMFusion','SeAFusion','SwinFusion']:
    if method not in all_data: continue
    comp = all_data[method]
    for metric in ['ssim','fir']:
        stat,p = stats.wilcoxon(sal[metric], comp[metric])
        direction = 'better' if np.mean(sal[metric])>np.mean(comp[metric]) else 'worse'
        print(f'  SalFuse vs {method} {metric.upper()}: p={p:.4f} ({direction})')
