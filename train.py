import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset import KittiDataset
from utils import get_query

from torchvision.utils import save_image
from torchvision.utils import make_grid

from models.cotr import COTR

import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 1
LR = 1e-4
LR_BB = 0
IMG_SIZE = 256

# for Adam: beta1 = 0.9, beta2 = 0.98 , smallE = 10e-9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#still testing on cpu !
device = torch.device('cpu')

dataset = KittiDataset(root='../dataset')
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

torch.cuda.empty_cache()
model = COTR()
model = model.to(device)

opt_list = [{'params': model.transformer.parameters(), 'lr': LR},
            {'params': model.mlp.parameters(), 'lr': LR},
            {'params': model.pos_emb.parameters(), 'lr': LR},
            {'params': model.proj_q.parameters(), 'lr': LR}]

# backbone train?
if LR_BB > 0:
    opt_list.append({'params': model.backbone.parameters(), 'lr':LR_BB})

opt = optim.Adam(opt_list)

for batchid, (img, _, corrs) in enumerate(loader):
    
    img = img.to(device)
    corrs = corrs.to(device)
    
    query = corrs[:, 0, :, :]
    target = corrs[:, 1, :, :]
    

    opt.zero_grad()
    
    pred = model(img, query)['pred_corrs']
    loss = F.mse_loss(pred, target)
    
    img_reverse = torch.cat([img[..., IMG_SIZE:], img[..., :IMG_SIZE]], axis=-1)
    query_reverse = pred.clone()
    
    cycle = model(img_reverse, query_reverse)['pred_corrs']
    mask = torch.norm(cycle - query, dim=-1) < 100 / IMG_SIZE
    cycle_loss = 0
    if mask.sum() > 0:
        cycle_loss = F.mse_loss(cycle[mask], query[mask])
        loss += cycle_loss
        
    loss_data = loss.data.item()
    if np.isnan(loss_data):
        print('loss is nan')
        opt.zero_grad()
    else:
        loss.backward()
    opt.step()    
    
    print(batchid)
    if batchid % 10 == 0:    
        torch.save(model.state_dict(), f'./saved/bid{batchid}.pth')
        print(f'Loss in b_id{batchid}: { loss.detach().numpy() }')
    #plt.imshow(preview)
    #print(preview.shape)
