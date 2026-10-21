import cv2, sys, os, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '.')
from utils import utils_option as option
from models.select_model import define_Model
from medswin_acf_combined import medswin_acf_fuse, compute_pet_saliency
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

def compute_saliency_np(pet_bgr):
    pet_hsv = cv2.cvtColor(pet_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    S = pet_hsv[:,:,1]/255.0; V = pet_hsv[:,:,2]/255.0
    sv = S*V; sv_blur = cv2.GaussianBlur(sv,(0,0),5.0)
    mx = sv_blur.max()
    return sv_blur/mx if mx>1e-8 else np.zeros_like(sv_blur)

class SaliencyGuidedFusion(nn.Module):
    def __init__(self, base_netG):
        super().__init__()
        self.netG = base_netG
        self.sal_adapter = nn.Sequential(
            nn.Conv2d(2,16,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16,1,1), nn.Sigmoid()
        )
    def forward(self, mri, pet_sal):
        adapted = self.sal_adapter(pet_sal)
        return self.netG(adapted, mri)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_MRI = 'Dataset/testsets/PET-MRI/MRI'
TEST_PET = 'Dataset/testsets/PET-MRI/PET_RGB'

def load_option_c():
    JSON_PATH='options/swinir/train_swinfusion_med.json'
    opt=option.parse(JSON_PATH,is_train=True)
    exp='finetune_experiments_sal'
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
    base=define_Model(opt)
    model=SaliencyGuidedFusion(base.netG)
    ckpt=torch.load('finetune_checkpoints_saliency/best_model.pth',map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['params'],strict=True)
    print('Loaded Option C model, epoch',ckpt['epoch'],'train_ssim',round(ckpt['ssim'],4))
    return model.to(DEVICE).eval()

def cnr_fixed(fused_bgr, pet_bgr):
    fg=cv2.cvtColor(fused_bgr,cv2.COLOR_BGR2GRAY).astype(np.float64)
    pg=cv2.cvtColor(pet_bgr,cv2.COLOR_BGR2GRAY).astype(np.float64)
    sm=pg>=np.percentile(pg,80); bm=pg<=np.percentile(pg,20)
    if sm.sum()==0 or bm.sum()==0: return 0.0
    return float(abs(fg[sm].mean()-fg[bm].mean())/max(fg[bm].std(),1.0))

def delta_e(fused_bgr, pet_bgr):
    try:
        fl=cv2.cvtColor(fused_bgr.astype(np.uint8),cv2.COLOR_BGR2LAB).astype(np.float32)
        pl=cv2.cvtColor(pet_bgr.astype(np.uint8),cv2.COLOR_BGR2LAB).astype(np.float32)
        pg=cv2.cvtColor(pet_bgr.astype(np.uint8),cv2.COLOR_BGR2GRAY)
        h=pg>=np.percentile(pg,80)
        if h.sum()==0: return 0.0
        return float(np.sqrt(np.sum((fl-pl)**2,axis=2))[h].mean())
    except: return 0.0

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

model_c = load_option_c()

OUT_C   = 'results/OptionC_backbone_PET-MRI'
OUT_CS  = 'results/OptionC_SalFuseFormer_PET-MRI'
os.makedirs(OUT_C,  exist_ok=True)
os.makedirs(OUT_CS, exist_ok=True)

files=sorted([f for f in os.listdir(TEST_MRI) if f.endswith('.png')])
metrics = {k: [] for k in ['ssim_c','ssim_cs','psnr_c','psnr_cs','mi_c','mi_cs',
                             'epi_c','epi_cs','cnr_c','cnr_cs','fir_c','fir_cs',
                             'de_c','de_cs']}

with torch.no_grad():
    for fname in files:
        mri=cv2.imread(os.path.join(TEST_MRI,fname))
        pet=cv2.imread(os.path.join(TEST_PET,fname))
        if mri is None or pet is None: continue
        mg=cv2.resize(cv2.cvtColor(mri,cv2.COLOR_BGR2GRAY),(256,256))
        pet256=cv2.resize(pet,(256,256))
        py=cv2.cvtColor(pet256,cv2.COLOR_BGR2YCrCb)[:,:,0]
        sal=compute_saliency_np(pet256).astype(np.float32)
        mt=torch.from_numpy(mg.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)/255.0
        pt=torch.from_numpy(py.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)/255.0
        st=torch.from_numpy(sal).unsqueeze(0).unsqueeze(0).to(DEVICE)
        pet_sal=torch.cat([pt,st],dim=1)
        out=model_c(mt,pet_sal)
        out_np=(out.squeeze().cpu().numpy()*255).clip(0,255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT_C,fname),out_np)
        mri_bgr=cv2.cvtColor(mg,cv2.COLOR_GRAY2BGR)
        sal_result=medswin_acf_fuse(out_np,pet256,mri_bgr,modality='MRI-PET')
        cv2.imwrite(os.path.join(OUT_CS,fname),sal_result)
        sal_g=cv2.cvtColor(sal_result,cv2.COLOR_BGR2GRAY)
        Fg=out_np.astype(np.float64); Mg=mg.astype(np.float64)
        Pg=py.astype(np.float64)
        metrics['ssim_c'].append(0.5*(sk_ssim(out_np,mg,data_range=255)+sk_ssim(out_np,py,data_range=255)))
        metrics['ssim_cs'].append(0.5*(sk_ssim(sal_g,mg,data_range=255)+sk_ssim(sal_g,py,data_range=255)))
        metrics['psnr_c'].append(sk_psnr(Mg,Fg,data_range=255))
        metrics['psnr_cs'].append(sk_psnr(Mg,sal_g.astype(np.float64),data_range=255))
        metrics['mi_c'].append(mi(Fg,Mg)+mi(Fg,Pg))
        Fg_s=sal_g.astype(np.float64)
        metrics['mi_cs'].append(mi(Fg_s,Mg)+mi(Fg_s,Pg))
        metrics['epi_c'].append(0.5*(corr(sob(Fg),sob(Mg))+corr(sob(Fg),sob(Pg))))
        metrics['epi_cs'].append(0.5*(corr(sob(Fg_s),sob(Mg))+corr(sob(Fg_s),sob(Pg))))
        metrics['cnr_c'].append(cnr_fixed(cv2.cvtColor(out_np,cv2.COLOR_GRAY2BGR),pet256))
        metrics['cnr_cs'].append(cnr_fixed(sal_result,pet256))
        best=np.maximum(Mg,Pg)
        metrics['fir_c'].append(1.0-np.mean(np.abs(Fg-best))/(np.mean(np.abs(best))+1e-8))
        metrics['fir_cs'].append(1.0-np.mean(np.abs(Fg_s-best))/(np.mean(np.abs(best))+1e-8))
        metrics['de_c'].append(delta_e(cv2.cvtColor(out_np,cv2.COLOR_GRAY2BGR),pet256))
        metrics['de_cs'].append(delta_e(sal_result,pet256))

n=len(files)
print('OPTION C RESULTS (N='+str(n)+'):')
print('Method               SSIM   PSNR_A  MI      EPI     CNR     FIR     DeltaE')
print('-'*80)
print('OptionC_backbone     '+str(round(np.mean(metrics['ssim_c']),4)).ljust(7)+str(round(np.mean(metrics['psnr_c']),2)).ljust(8)+str(round(np.mean(metrics['mi_c']),4)).ljust(8)+str(round(np.mean(metrics['epi_c']),4)).ljust(8)+str(round(np.mean(metrics['cnr_c']),4)).ljust(8)+str(round(np.mean(metrics['fir_c']),4)).ljust(8)+str(round(np.mean(metrics['de_c']),4)))
print('OptionC_SalFuse      '+str(round(np.mean(metrics['ssim_cs']),4)).ljust(7)+str(round(np.mean(metrics['psnr_cs']),2)).ljust(8)+str(round(np.mean(metrics['mi_cs']),4)).ljust(8)+str(round(np.mean(metrics['epi_cs']),4)).ljust(8)+str(round(np.mean(metrics['cnr_cs']),4)).ljust(8)+str(round(np.mean(metrics['fir_cs']),4)).ljust(8)+str(round(np.mean(metrics['de_cs']),4)))
print()
print('Reference:')
print('FT_SwinFusion        0.7235  24.62   3.419   0.5744  1.9076  0.9267  86.6059')
print('SalFuseFormer(orig)  0.7183  23.59   2.9843  0.5737  1.7078  0.8851  28.5714')
print('CDDFuse              0.726   24.38   3.4576  0.5637  1.8665  0.9148  25.3823')
print('SwinFusion           0.5819  24.02   3.2278  0.5898  1.9353  0.8754  85.3016')
