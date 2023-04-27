import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset import KittiDataset
from utils import get_query, plot_predictions

from torchvision.utils import save_image
from torchvision.utils import make_grid

from models.cotr import COTR

import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF

BATCH_SIZE = 1
IMG_SIZE = 256


device = torch.device('cpu')

dataset = KittiDataset(root = '../dataset', split='test')
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = COTR()
model = model.to(device)

model.load_state_dict(torch.load('./saved/plus200epoch.pth'))
model.eval()

for batchid, (img, _, query) in enumerate(loader):
    
    img = img.to(device)
    #corrs = corrs.to(device)

    pred = model(img, query)['pred_corrs']

    plot_predictions(img, query, pred, query, 'example_1', 'plot_test')
    #sketch
    
    tstq = query.detach().numpy()
    tstp = pred.detach().numpy()
    
    tstq = tstq * IMG_SIZE
    tstp = tstp * IMG_SIZE
    
    
    breakpoint()
    
    print(batchid)
