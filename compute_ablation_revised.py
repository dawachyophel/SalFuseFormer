import cv2, numpy as np, os, sys
sys.path.insert(0, '.')
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

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

def de(f,p):
    try:
        fl=cv2.cvtColor(f.astype(np.uint8),cv2.COLOR_BGR2LAB).astype(np.float32)
        pl=cv2.cvtColor(p.astype(np.uint8),cv2.COLOR_BGR2LAB).astype(np.float32)
        pg=cv2.cvtColor(p.astype(np.uint8),cv2.COLOR_BGR2GRAY)
        h=pg>=np.percentile(pg,80)
        if h.sum()==0: return 0.0
        return float(np.sqrt(np.sum((fl-pl)**2,axis=2))[h].mean())
    except: return 0.0

def compute_metrics(mdir,adir,bdir):
    files=sorted([f for f in os.listdir(mdir) if f.endswith('.png')])
    s=dict(ssim=0,pa=0,mi=0,epi=0,cnr=0,fir=0,de=0); n=0
    for fname in files:
        F=cv2.imread(os.path.join(mdir,fname))
        A=cv2.imread(os.path.join(adir,fname))
        B=cv2.imread(os.path.join(bdir,fname))
        if any(x is None for x in [F,A,B]): continue
        F=cv2.resize(F,(256,256)); A=cv2.resize(A,(256,256)); B=cv2.resize(B,(256,256))
        def g(img):
            return (cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img).astype(np.float64)
        Fg=g(F);Ag=g(A);Bg=g(B)
        s['ssim']+=0.5*(sk_ssim(Fg.astype(np.uint8),Ag.astype(np.uint8),data_range=255)+
                        sk_ssim(Fg.astype(np.uint8),Bg.astype(np.uint8),data_range=255))
        s['pa']+=sk_psnr(Ag,Fg,data_range=255)
        s['mi']+=mi(Fg,Ag)+mi(Fg,Bg)
        s['epi']+=0.5*(corr(sob(Fg),sob(Ag))+corr(sob(Fg),sob(Bg)))
        s['cnr']+=cnr(F,B)
        best=np.maximum(Ag,Bg)
        s['fir']+=1.0-np.mean(np.abs(Fg-best))/(np.mean(np.abs(best))+1e-8)
        s['de']+=de(F,B)
        n+=1
    if n==0: return None
    return {k:round(v/n,4) for k,v in s.items()}

MRI_DIR = 'Dataset/testsets/PET-MRI/MRI'
PET_DIR = 'Dataset/testsets/PET-MRI/PET_RGB'

CONFIGS = {
    'A_Baseline_SwinFusion':  'results/ablation_A_baseline_colour',
    'B_Baseline_plus_EI':     'results/SwinFusion_PET-MRI_MedSwin',
    'C_Baseline_plus_ACF':    'results/SwinFusion_PET-MRI_ACF',
    'D_SalFuseFormer_full':   'results/SalFuseFormer_FINAL_PET-MRI',
}

print('REVISED ABLATION METRICS (consistent colour reconstruction):')
print('='*95)
print('Configuration              SSIM    PSNR-MRI  MI      EPI     CNR     FIR     DeltaE')
print('-'*95)

for name, mdir in CONFIGS.items():
    if not os.path.exists(mdir):
        print(name, 'NOT FOUND'); continue
    # Check colour
    f = sorted(os.listdir(mdir))[0]
    img = cv2.imread(os.path.join(mdir,f))
    is_color = not np.allclose(img[:,:,0], img[:,:,1])
    m = compute_metrics(mdir, MRI_DIR, PET_DIR)
    if m is None: continue
    print(str(name).ljust(26)+' '+
          str(m['ssim']).ljust(8)+str(m['pa']).ljust(10)+
          str(m['mi']).ljust(8)+str(m['epi']).ljust(8)+
          str(m['cnr']).ljust(8)+str(m['fir']).ljust(8)+
          str(m['de'])+'  colour:'+str(is_color))
