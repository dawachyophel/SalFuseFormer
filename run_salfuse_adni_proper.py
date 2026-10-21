import cv2, sys, os, numpy as np, torch
sys.path.insert(0, '.')
from utils import utils_option as option
from models.select_model import define_Model
from medswin_acf_combined import medswin_acf_fuse

DEVICE  = torch.device('cuda')
MRI_DIR = 'Dataset/testsets/ADNI_PROPER/MRI'
PET_DIR = 'Dataset/testsets/ADNI_PROPER/PET_RGB'
OUT_DIR = 'comparison_results/ADNI_PROPER/SalFuseFormer'
os.makedirs(OUT_DIR, exist_ok=True)

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
    ckpt=torch.load('finetune_checkpoints/best_model.pth',
                    map_location='cpu',weights_only=False)
    model.netG.load_state_dict(ckpt['params'],strict=True)
    return model.netG.to(DEVICE).eval()

print('Loading fine-tuned backbone...')
netG = load_finetuned()

files = sorted([f for f in os.listdir(MRI_DIR) if f.endswith('.png')])
print('Processing', len(files), 'ADNI_PROPER pairs...')

with torch.no_grad():
    for i, fname in enumerate(files):
        mri = cv2.imread(os.path.join(MRI_DIR, fname))
        pet = cv2.imread(os.path.join(PET_DIR, fname))
        if mri is None or pet is None: continue
        mg = cv2.cvtColor(mri, cv2.COLOR_BGR2GRAY)
        py = cv2.cvtColor(pet, cv2.COLOR_BGR2YCrCb)[:,:,0]
        mg = cv2.resize(mg,(256,256))
        py = cv2.resize(py,(256,256))
        mt = torch.from_numpy(mg.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)/255.0
        pt = torch.from_numpy(py.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)/255.0
        fy = netG(pt, mt)
        fy_np = (fy.squeeze().cpu().numpy()*255).clip(0,255).astype(np.uint8)
        pet256  = cv2.resize(pet,(256,256))
        mri_bgr = cv2.cvtColor(mg, cv2.COLOR_GRAY2BGR)
        result  = medswin_acf_fuse(fy_np, pet256, mri_bgr,
                                   alpha=0.05, sal_threshold=0.20,
                                   acf_blend=0.25, modality='MRI-PET')
        cv2.imwrite(os.path.join(OUT_DIR, fname), result)
        if (i+1)%30==0: print(' ', i+1, '/', len(files))

print('Done.', len(os.listdir(OUT_DIR)), 'files saved to', OUT_DIR)
