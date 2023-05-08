import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset import KittiDataset
from utils import get_query, plot_predictions, PCK_N, AEPE

from torchvision.utils import save_image
from torchvision.utils import make_grid

from models.cotr import COTR

import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 1
LR = 1e-4
LR_BB = 0
IMG_SIZE = 256
EPOCHS = 1000
NUM_KP = 100

#model params
EMB_DIM = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
RETURN_INTERMEDIATE = True
DROPOUT = 0.1
NLAYERS = 3

# for Adam: beta1 = 0.9, beta2 = 0.98 , smallE = 10e-9

def start(path='../dataset'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #still testing on cpu !
    device = torch.device('cpu')
    
    dataset = KittiDataset(root=path, num_kp=NUM_KP)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    torch.cuda.empty_cache()
    model = COTR(
        emd_dim=EMB_DIM,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        return_intermediate=RETURN_INTERMEDIATE,
        dropout=DROPOUT,
        nlayers=NLAYERS
    )
    model = model.to(device)

    total_params = sum(
        param.numel() for param in model.parameters()
    )

    bb_params = sum(
        param.numel() for param in model.backbone.parameters()
    )

    print(f'Number of parameters: {total_params}')
    print(f'Backbone parameters: {bb_params}')
    
    opt_list = [{'params': model.transformer.parameters(), 'lr': LR},
                {'params': model.mlp.parameters(), 'lr': LR},
                {'params': model.proj_x.parameters(), 'lr': LR},
                {'params': model.proj_q.parameters(), 'lr': LR}]
    
    # backbone train?
    if LR_BB > 0:
        opt_list.append({'params': model.backbone.parameters(), 'lr':LR_BB})
    
    opt = optim.Adam(opt_list)
    
    for _ in range(EPOCHS):
        for batchid, (img, _, corrs) in enumerate(loader):
            
            img = img.to(device)
            corrs = corrs.to(device)

            query = corrs[:, 0, :, :]
            target = corrs[:, 1, :, :]
            
            opt.zero_grad()
            
            pred = model(img, query)['pred_corrs']
            loss = torch.mean((pred-target)**2)
            
            img_reverse = torch.cat([img[..., IMG_SIZE:], img[..., :IMG_SIZE]], axis=-1)
            query_reverse = pred.clone()
            query_reverse[..., 0] = query_reverse[..., 0] - 0.5
        
            cycle = model(img_reverse, query_reverse)['pred_corrs']
            cycle[..., 0] = cycle[..., 0] - 0.5 
        
            mask = torch.norm(cycle - query, dim=-1) < 10 / IMG_SIZE
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

            if batchid % 10 == 0:
                #torch.save(model.state_dict(), f'{path}/saved/bid{batchid}.pth')
                print(f'Loss in b_id{batchid}: { loss.detach().numpy() }')
                pck = PCK_N(img, query, pred, target, threshold=1)
                aepe = AEPE(img, query, pred, target)
                print(f'PCK: {pck}, AEPE: {aepe}')
                plot_predictions(img, query, pred, target, batchid, 'plot_test')
                breakpoint()


        #plt.imshow(preview)
        #print(preview.shape)
        
start()
