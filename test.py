import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset import KittiDataset
from utils import plot_predictions, plot_real, AEPE, PCK_N

from torchvision.utils import save_image
from torchvision.utils import make_grid

from models.cotr import COTR

import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF

BATCH_SIZE = 1
IMG_SIZE = 256
NUM_KP = 21

EMB_DIM = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
RETURN_INTERMEDIATE = True
DROPOUT = 0.1
NLAYERS = 3


device = torch.device('cpu')

dataset = KittiDataset(root = '../dataset', split='val', num_kp=NUM_KP)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

model.load_state_dict(torch.load('./saved/ep400_bid39.pth'))
model.eval()

for batchid, (img, w, corrs) in enumerate(loader):
    
    img = img.to(device)
    corrs = corrs.to(device)

    query = corrs[:,0,:,:]
    target = corrs[:,1,:,:]

    pred = model(img, query)['pred_corrs']

    loss = F.mse_loss(pred,target)

    img_reverse = torch.cat([img[..., IMG_SIZE:], img[..., :IMG_SIZE]], axis=-1)
    query_reverse = pred.clone()
    query_reverse[..., 0] = query_reverse[..., 0] - 0.5

    cycle = model(img_reverse, query_reverse)['pred_corrs']

    mask = torch.norm(cycle - query, dim=-1) < 10 / IMG_SIZE
    cycle_loss = 0
    if mask.sum() > 0:
        cycle_loss = F.mse_loss(cycle[mask], query[mask])
        loss += cycle_loss
    
    loss_data = loss.data.item()
    if np.isnan(loss_data):
        print('loss is nan')

    if batchid % 2 == 0:
        print(f'Loss     in bid_{batchid}: {loss.cpu().detach().numpy():.8f}')
        pck1 = PCK_N(img, query, pred, target, threshold=1)
        pck3 = PCK_N(img, query, pred, target, threshold=3)
        pck5 = PCK_N(img, query, pred, target, threshold=5)
        aepe = AEPE(img, query, pred, target)
        print(f' PCK-1px: {pck1}, PCK-3px: {pck3}, PCK-5px: {pck5}, AEPE: {aepe}')
        plot_predictions(img, query, pred, target, 'example_1', 'plot_test')
    #plot_real(w,query,pred)
    #sketch 
