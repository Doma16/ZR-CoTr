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
IMG_SIZE = 256


device = torch.device('cpu')

dataset = KittiDataset(root = '../dataset', split='test')
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = COTR()
model = model.to(device)


for batchid, (img, _, corrs) in enumerate(loader):
    
    img = img.to(device)
    corrs = corrs.to(device)
    
    query = corrs[:, 0, :, :]
    target = corrs[:, 1, :, :]
    
    pred = model(img, query)['pred_corrs']
    
    breakpoint()
    
    print(batchid)