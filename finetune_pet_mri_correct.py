import os, sys, time, random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, '.')

TRAIN_MRI = r'D:\Harvard_Training_Data\PET-MRI\MRI'
TRAIN_PET = r'D:\Harvard_Training_Data\PET-MRI\PET'
TEST_MRI  = r'Dataset\testsets\PET-MRI\MRI'
TEST_PET  = r'Dataset\testsets\PET-MRI\PET_RGB'
JSON_PATH = 'options/swinir/train_swinfusion_med.json'
SAVE_DIR  = 'finetune_checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR         = 2e-5
BATCH_SIZE = 8
PATCH_SIZE = 64
EPOCHS     = 30
VAL_EVERY  = 1
SAVE_EVERY = 5

print('Device:', DEVICE)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

class MedPatchDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, patch_size=64, augment=True):
        self.mri_dir=mri_dir; self.pet_dir=pet_dir
        self.patch_size=patch_size; self.augment=augment
        self.files=sorted(set(os.listdir(mri_dir))&set(os.listdir(pet_dir)))
        self.ppi=16
        print('Dataset:', len(self.files), 'images,', len(self.files)*self.ppi, 'patches/epoch')
    def __len__(self): return len(self.files)*self.ppi
    def __getitem__(self,idx):
        f=self.files[idx//self.ppi]
        mri=cv2.imread(os.path.join(self.mri_dir,f))
        pet=cv2.imread(os.path.join(self.pet_dir,f))
        if mri is None or pet is None: return self.__getitem__((idx+1)%len(self))
        mg=cv2.cvtColor(mri,cv2.COLOR_BGR2GRAY)
        py=cv2.cvtColor(pet,cv2.COLOR_BGR2YCrCb)[:,:,0]
        H,W=mg.shape; ps=self.patch_size
        if H>ps and W>ps:
            t=random.randint(0,H-ps); l=random.randint(0,W-ps)
            mg=mg[t:t+ps,l:l+ps]; py=py[t:t+ps,l:l+ps]
        else:
            mg=cv2.resize(mg,(ps,ps)); py=cv2.resize(py,(ps,ps))
        if self.augment:
            if random.random()>0.5: mg=np.fliplr(mg).copy(); py=np.fliplr(py).copy()
            if random.random()>0.5: mg=np.flipud(mg).copy(); py=np.flipud(py).copy()
            k=random.randint(0,3); mg=np.rot90(mg,k).copy(); py=np.rot90(py,k).copy()
        return (torch.from_numpy(mg.astype(np.float32)).unsqueeze(0)/255.0,
                torch.from_numpy(py.astype(np.float32)).unsqueeze(0)/255.0, f)

def load_model():
    from utils import utils_option as option
    from models.select_model import define_Model
    opt=option.parse(JSON_PATH,is_train=True)
    exp='finetune_experiments'
    opt['path']['root']=exp
    opt['path']['models']=os.path.join(exp,'models')
    opt['path']['log']=os.path.join(exp,'log')
    opt['path']['options']=os.path.join(exp,'options')
    opt['path']['images']=os.path.join(exp,'images')
    opt['path']['pretrained_netG']='Model/Medical_Fusion-PET-MRI/Medical_Fusion/models/10000_G.pth'
    opt['path']['pretrained_netE']='Model/Medical_Fusion-PET-MRI/Medical_Fusion/models/10000_E.pth'
    opt['gpu_ids']=[0]; opt['rank']=0; opt['dist']=False; opt['is_train']=True
    opt['netG']['init_type']='default'
    opt['netG']['init_bn_type']='uniform'
    opt['netG']['init_gain']=0.2
    opt['train']['G_optimizer_lr']=LR
    opt['train']['G_optimizer_reuse']=False
    for k,v in opt['path'].items():
        if v and isinstance(v,str) and 'pretrained' not in k:
            os.makedirs(v,exist_ok=True)
    model=define_Model(opt)
    print('Params:', sum(p.numel() for p in model.netG.parameters())/1e6, 'M')
    return model

def validate(netG):
    netG.eval(); scores=[]
    with torch.no_grad():
        for fname in sorted(os.listdir(TEST_MRI)):
            mri=cv2.imread(os.path.join(TEST_MRI,fname))
            pet=cv2.imread(os.path.join(TEST_PET,fname))
            if mri is None or pet is None: continue
            mg=cv2.resize(cv2.cvtColor(mri,cv2.COLOR_BGR2GRAY),(256,256))
            py=cv2.resize(cv2.cvtColor(pet,cv2.COLOR_BGR2YCrCb)[:,:,0],(256,256))
            mt=torch.from_numpy(mg.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)/255.0
            pt=torch.from_numpy(py.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)/255.0
            out=netG(pt,mt)
            fn=(out.squeeze().cpu().numpy()*255).clip(0,255).astype(np.uint8)
            scores.append(sk_ssim(fn,mg,data_range=255))
    netG.train()
    return float(np.mean(scores)) if scores else 0.0

def train():
    print('='*60)
    print('SwinFusion Fine-tuning - Harvard MRI-PET 269 pairs')
    print('='*60)
    model=load_model()
    netG=model.netG.to(DEVICE)
    from models.loss_med import fusion_loss_med
    criterion=fusion_loss_med().to(DEVICE)
    ds=MedPatchDataset(TRAIN_MRI,TRAIN_PET,patch_size=PATCH_SIZE,augment=True)
    dl=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=True)
    optimizer=optim.Adam(netG.parameters(),lr=LR,betas=(0.9,0.999))
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-6)
    baseline=validate(netG)
    print('Baseline SSIM:', round(baseline,4))
    best_ssim=baseline; best_epoch=0
    print('Training',len(ds),'patches for',EPOCHS,'epochs')
    for epoch in range(1,EPOCHS+1):
        losses=[]; t0=time.time()
        for mt,pt,_ in dl:
            mt=mt.to(DEVICE); pt=pt.to(DEVICE)
            optimizer.zero_grad()
            fused=netG(pt,mt)
            loss,lg,li,ls=criterion(mt,pt,fused)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(),1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        elapsed=time.time()-t0
        if epoch%VAL_EVERY==0:
            val_ssim=validate(netG)
            marker=''
            if val_ssim>best_ssim:
                best_ssim=val_ssim; best_epoch=epoch
                torch.save({'epoch':epoch,'params':netG.state_dict(),'ssim':val_ssim},
                           os.path.join(SAVE_DIR,'best_model.pth'))
                marker=' **BEST**'
            print('Ep'+str(epoch).zfill(3)+'/'+str(EPOCHS)+
                  ' Loss='+str(round(np.mean(losses),4))+
                  ' ValSSIM='+str(round(val_ssim,4))+
                  ' Best='+str(round(best_ssim,4))+
                  '(ep'+str(best_epoch)+')'+
                  ' '+str(round(elapsed,1))+'s'+marker)
        else:
            print('Ep'+str(epoch).zfill(3)+'/'+str(EPOCHS)+
                  ' Loss='+str(round(np.mean(losses),4))+
                  ' '+str(round(elapsed,1))+'s')
        if epoch%SAVE_EVERY==0:
            torch.save({'epoch':epoch,'params':netG.state_dict()},
                       os.path.join(SAVE_DIR,'epoch_'+str(epoch).zfill(3)+'.pth'))
    print('Done. Best SSIM='+str(round(best_ssim,4))+' at epoch '+str(best_epoch))

if __name__=='__main__':
    train()
