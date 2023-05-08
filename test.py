import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset import KittiDataset
from utils import plot_predictions, plot_real

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

model.load_state_dict(torch.load('./saved/t1e400_bid39.pth'))
model.eval()

for batchid, (img, w, corrs) in enumerate(loader):
    
    img = img.to(device)
    corrs = corrs.to(device)

    query = corrs[:,0,:,:]
    target = corrs[:,1,:,:]

    pred = model(img, query)['pred_corrs']

    plot_real(w,query,pred)
    
    plot_predictions(img, query, pred, target, 'example_1', 'plot_test')
    #sketch 
    print(batchid)
