import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset import KittiDataset, MiddleBury 
from utils import plot_predictions, plot_real, AEPE, PCK_N

from torchvision.utils import save_image
from torchvision.utils import make_grid

from models.cotr import COTR

import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF

from inference.simple_engine import simple_engine

BATCH_SIZE = 1
IMG_SIZE = 256
NUM_KP = 100

EMB_DIM = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
RETURN_INTERMEDIATE = True
DROPOUT = 0.1
NLAYERS = 3

PATH = './saved/ep400_bid39.pth'

device = torch.device('cpu')

dataset = MiddleBury(root = '../dataset', num_kp=NUM_KP)
dataset = KittiDataset(root = '../dataset', transforms='tile' ,split='val', num_kp=NUM_KP)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

model = COTR(
    emd_dim=EMB_DIM,
    nhead=NHEAD,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    return_intermediate=RETURN_INTERMEDIATE,
    dropout=DROPOUT,
    nlayers=NLAYERS
)

model.load_state_dict(torch.load(PATH, map_location=device))
model = model.to(device)
model.eval()

engine = simple_engine(model)

losses = []
pck1s = []
pck3s = []
pck5s = []
aepes = []


breakpoint() #(img, w, corrs) for kittistereo # (img, w, corrs, vm) for middlebury
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

    losses.append(loss.detach())
    pck1s.append(PCK_N(img, query, pred, target, threshold=1))
    pck3s.append(PCK_N(img, query, pred, target, threshold=3))
    pck5s.append(PCK_N(img, query, pred, target, threshold=5))
    aepes.append(AEPE(img, query, pred, target))

    if batchid % 2 == 0:
        #print(f'Loss     in bid_{batchid}: {loss.cpu().detach().numpy():.8f}')
        #pck1 = PCK_N(img, query, pred, target, threshold=1)
        #pck3 = PCK_N(img, query, pred, target, threshold=3)
        #pck5 = PCK_N(img, query, pred, target, threshold=5)
        #aepe = AEPE(img, query, pred, target)
        #print(f' PCK-1px: {pck1}, PCK-3px: {pck3}, PCK-5px: {pck5}, AEPE: {aepe}')
        #breakpoint()
        #temp_img = img
        #temp_img[..., 256:] = temp_img[..., :256]
        #plot_predictions(temp_img, query, cycle+0.5, query, 'example_1', 'plot_test')
        #engine.interpolation_disparity_predict(img)
        #engine.simple_predict()
        pass
        #breakpoint()
    #sketch 


print(f'Loss avg: {np.mean(losses)}')
print(f'PCK 1px avg: {np.mean(pck1s)}')
print(f'PCK 3px avg: {np.mean(pck3s)}')
print(f'PCK 5px avg: {np.mean(pck5s)}')
print(f'AEPE avg: {np.mean(aepes)}')

breakpoint()