import cv2, sys, os, numpy as np, csv, torch
sys.path.insert(0, '.')
from utils import utils_option as option
from models.select_model import define_Model
from medswin_acf_combined import medswin_acf_fuse, compute_pet_saliency
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

DEVICE = torch.device('cuda')

def load_finetuned():
    opt=option.parse('options/swinir/train_swinfusion_med.json',is_train=True)
    exp='finetune_experiments'
    opt['path']['root']=exp
    opt['path']['models']=os.path.join(exp,'models')
    opt['path']['log']=os.path.join(exp,'log')
    opt['path']['options']=os.path.join(exp,'options')
    opt['path']['images']=os.path.join(exp,'images')
    opt['path']['pretrained_netG']=None
    opt['path']['pretrained_netE']=None
    opt['gpu_ids']=[0]; opt['rank']=0; opt['dist']=False; opt['is_train']=False
    opt['netG']['init_type']='default'
    opt['netG']['init_bn_type']='uniform'
    opt['netG']['init_gain']=0.2
    for k,v in opt['path'].items():
        if v and isinstance(v,str) and 'pretrained' not in k:
            os.makedirs(v,exist_ok=True)
    model=define_Model(opt)
    ckpt=torch.load('finetune_checkpoints/best_model.pth',map_location='cpu',weights_only=False)
    model.netG.load_state_dict(ckpt['params'],strict=True)
    return model.netG.to(DEVICE).eval()

def cnr_fixed(f,p):
    fg=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float64)
    pg=cv2.cvtColor(p,cv2.COLOR_BGR2GRAY).astype(np.float64)
    sm=pg>=np.percentile(pg,80); bm=pg<=np.percentile(pg,20)
    if sm.sum()==0 or bm.sum()==0: return 0.0
    return float(abs(fg[sm].mean()-fg[bm].mean())/max(fg[bm].std(),1.0))

def mi(x,y,bins=256):
    h,_,_=np.histogram2d(x.flatten(),y.flatten(),bins=bins)
    pxy=h/h.sum();px=pxy.sum(1);py=pxy.sum(0)
    pp=np.outer(px,py);mask=pxy>0
    return np.sum(pxy[mask]*np.log2(pxy[mask]/(pp[mask]+1e-10)))

PET_DIR = 'Dataset/testsets/PET-MRI/PET_RGB'
MRI_DIR = 'Dataset/testsets/PET-MRI/MRI'
SW_DIR  = 'results/Finetuned_SwinFusion_PET-MRI'

print('Loading fine-tuned backbone...')
netG = load_finetuned()
files = sorted([f for f in os.listdir(PET_DIR) if f.endswith('.png')])

# Pre-generate backbone outputs
print('Generating backbone outputs...')
backbone = {}
with torch.no_grad():
    for fname in files:
        mri=cv2.imread(os.path.join(MRI_DIR,fname))
        pet=cv2.imread(os.path.join(PET_DIR,fname))
        if mri is None or pet is None: continue
        mg=cv2.resize(cv2.cvtColor(mri,cv2.COLOR_BGR2GRAY),(256,256))
        py=cv2.resize(cv2.cvtColor(pet,cv2.COLOR_BGR2YCrCb)[:,:,0],(256,256))
        mt=torch.from_numpy(mg.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)/255.0
        pt=torch.from_numpy(py.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)/255.0
        fy=netG(pt,mt)
        fy_np=(fy.squeeze().cpu().numpy()*255).clip(0,255).astype(np.uint8)
        backbone[fname]=(fy_np, cv2.resize(mri,(256,256)), cv2.resize(pet,(256,256)))

def evaluate_params(alpha, tau, save_dir=None):
    s=dict(ssim=0,psnr=0,mi=0,cnr=0,fir=0); n=0
    for fname,(fy_np,mri,pet) in backbone.items():
        mri_bgr=cv2.cvtColor(cv2.cvtColor(mri,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        result=medswin_acf_fuse(fy_np,pet,mri_bgr,alpha=alpha,sal_threshold=tau,modality='MRI-PET')
        if save_dir:
            os.makedirs(save_dir,exist_ok=True)
            cv2.imwrite(os.path.join(save_dir,fname),result)
        rg=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY).astype(np.float64)
        mg=cv2.cvtColor(mri,cv2.COLOR_BGR2GRAY).astype(np.float64)
        pg=cv2.cvtColor(pet,cv2.COLOR_BGR2GRAY).astype(np.float64)
        s['ssim']+=0.5*(sk_ssim(rg.astype(np.uint8),mg.astype(np.uint8),data_range=255)+
                        sk_ssim(rg.astype(np.uint8),pg.astype(np.uint8),data_range=255))
        s['psnr']+=sk_psnr(mg,rg,data_range=255)
        s['mi']+=mi(rg,mg)+mi(rg,pg)
        s['cnr']+=cnr_fixed(result,pet)
        best=np.maximum(mg,pg)
        s['fir']+=1.0-np.mean(np.abs(rg-best))/(np.mean(np.abs(best))+1e-8)
        n+=1
    return {k:round(v/n,4) for k,v in s.items()} if n>0 else s

TAUS   = [0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60]
ALPHAS = [0.05,0.10,0.12,0.15,0.18,0.20,0.25,0.30]
os.makedirs('results/sensitivity_new',exist_ok=True)

print('Running tau sweep (alpha=0.05 fixed)...')
tau_rows=[]
for tau in TAUS:
    sdir='results/sensitivity_new/a0.05_t'+str(tau)
    m=evaluate_params(0.05,tau,save_dir=sdir)
    m['tau']=tau; m['alpha']=0.05
    tau_rows.append(m)
    print(' tau='+str(tau),m)

print('Running alpha sweep (tau=0.20 fixed)...')
alpha_rows=[]
for alpha in ALPHAS:
    sdir='results/sensitivity_new/a'+str(alpha)+'_t0.20'
    m=evaluate_params(alpha,0.20,save_dir=sdir)
    m['alpha']=alpha; m['tau']=0.20
    alpha_rows.append(m)
    print(' alpha='+str(alpha),m)

with open('results/sensitivity_new/tau_sweep.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['tau','alpha','ssim','psnr','mi','cnr','fir'])
    w.writeheader(); w.writerows(tau_rows)

with open('results/sensitivity_new/alpha_sweep.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['alpha','tau','ssim','psnr','mi','cnr','fir'])
    w.writeheader(); w.writerows(alpha_rows)

print('Done. Saved to results/sensitivity_new/')
