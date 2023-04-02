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

import torchvision.transforms.functional as TF

BATCH_SIZE = 1
IMG_SIZE = 256


device = torch.device('cpu')

dataset = KittiDataset(root = '../dataset', split='test')
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = COTR()
model = model.to(device)

model.load_state_dict(torch.load('./saved/bid190.pth'))


for batchid, (img, _, _) in enumerate(loader):
    
    img = img.to(device)
    #corrs = corrs.to(device)
    
    #query = corrs[:, 0, :, :]
    #target = corrs[:, 1, :, :]
    
    query = [[[0.9525, 0.3467],
              [0.9710, 0.3573],
              [0.8873, 0.9838],
              [0.7874, 0.3432]]]

    query = np.array(query).astype(np.float32)
    query = TF.to_tensor(query)
    query = query.reshape(1,4,2)

    breakpoint()
    
    pred = model(img, query)['pred_corrs']
    
    breakpoint()
    
    print(batchid)