import cv2, numpy as np, os, sys, pandas as pd
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

def hsv_corr(p):
    from scipy import stats
    ph=cv2.cvtColor(p.astype(np.uint8),cv2.COLOR_BGR2HSV).astype(np.float32)
    S=ph[:,:,1]/255.0; V=ph[:,:,2]/255.0
    sv=(S*V).ravel()
    pg=cv2.cvtColor(p.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()/255.0
    r,p_val=stats.pearsonr(sv,pg)
    sr,_=stats.spearmanr(sv,pg)
    return float(r),float(p_val),float(sr)

MRI_DIR = 'Dataset/testsets/ADNI_PROPER/MRI'
PET_DIR = 'Dataset/testsets/ADNI_PROPER/PET_RGB'
BASE    = 'comparison_results/ADNI_PROPER'
METHODS = ['SalFuseFormer','SwinFusion','EMFusion','SeAFusion','MUFusion']

all_rows=[]
for method in METHODS:
    mdir=os.path.join(BASE,method)
    if not os.path.exists(mdir): print(method,'NOT FOUND'); continue
    files=sorted([f for f in os.listdir(mdir) if f.endswith('.png')])
    print('Computing metrics for',method,'-',len(files),'files...')
    s=dict(ssim=0,pa=0,pb=0,mi=0,epi=0,cnr=0,fir=0,de=0,hr=0,hp=0,hsr=0); n=0
    for fname in files:
        F=cv2.imread(os.path.join(mdir,fname))
        M=cv2.imread(os.path.join(MRI_DIR,fname))
        P=cv2.imread(os.path.join(PET_DIR,fname))
        if any(x is None for x in [F,M,P]): continue
        F=cv2.resize(F,(256,256)); M=cv2.resize(M,(256,256)); P=cv2.resize(P,(256,256))
        def g(img):
            return (cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img).astype(np.float64)
        Fg=g(F);Mg=g(M);Pg=g(P)
        s['ssim']+=0.5*(sk_ssim(Fg.astype(np.uint8),Mg.astype(np.uint8),data_range=255)+
                        sk_ssim(Fg.astype(np.uint8),Pg.astype(np.uint8),data_range=255))
        s['pa']+=sk_psnr(Mg,Fg,data_range=255)
        s['pb']+=sk_psnr(Pg,Fg,data_range=255)
        s['mi']+=mi(Fg,Mg)+mi(Fg,Pg)
        s['epi']+=0.5*(corr(sob(Fg),sob(Mg))+corr(sob(Fg),sob(Pg)))
        s['cnr']+=cnr(F,P)
        best=np.maximum(Mg,Pg)
        s['fir']+=1.0-np.mean(np.abs(Fg-best))/(np.mean(np.abs(best))+1e-8)
        s['de']+=de(F,P)
        if method=='SalFuseFormer':
            r,pv,sr=hsv_corr(P)
            s['hr']+=r; s['hp']+=pv; s['hsr']+=sr
        n+=1
    if n==0: continue
    row={'method':method,'N':n,
         'SSIM':round(s['ssim']/n,4),'PSNR_MRI':round(s['pa']/n,2),
         'PSNR_PET':round(s['pb']/n,2),'MI':round(s['mi']/n,4),
         'EPI':round(s['epi']/n,4),'CNR':round(s['cnr']/n,4),
         'FIR':round(s['fir']/n,4),'Delta_E':round(s['de']/n,4)}
    if method=='SalFuseFormer':
        row['HSV_pearson_r']=round(s['hr']/n,4)
        row['HSV_pearson_p']=round(s['hp']/n,6)
        row['HSV_spearman_r']=round(s['hsr']/n,4)
    all_rows.append(row)
    print(' ',row)

df=pd.DataFrame(all_rows)
df.to_csv('adni_proper_metrics.csv',index=False)
print()
print('='*80)
print('ADNI_PROPER METRICS SUMMARY (N=147)')
print('='*80)
cols=['method','SSIM','PSNR_MRI','PSNR_PET','MI','EPI','CNR','FIR','Delta_E']
print(df[cols].to_string(index=False))
print()
sal=df[df['method']=='SalFuseFormer']
if len(sal)>0 and 'HSV_pearson_r' in sal.columns:
    print('HSV Saliency Validation:')
    print('  Pearson r:', sal['HSV_pearson_r'].values[0])
    print('  Spearman r:', sal['HSV_spearman_r'].values[0])
print('Saved to adni_proper_metrics.csv')
