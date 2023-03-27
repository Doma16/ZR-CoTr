import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from models.backbone import BackBone
from dataset import KittiDataset
from utils import get_query

from torchvision.utils import save_image
from torchvision.utils import make_grid

import numpy as np

import matplotlib.pyplot as plt

BATCH_SIZE = 1

# for Adam: beta1 = 0.9, beta2 = 0.98 , smallE = 10e-9

backboneEncoder = BackBone().eval()

dataset = KittiDataset(root='../dataset')
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#need transforms

for batchid, (img1, img2, dmap) in enumerate(loader):
    query = get_query(dmap, 1)



    y = backboneEncoder(img1)

    #preview = y.detach().numpy().squeeze(0)
    #preview = np.mean(preview, axis=0)
    preview = y.view(256, 1, 16, 16)
    print(preview.shape)
    breakpoint()
    save_image(preview, f'./pics/{batchid}.png')

    #plt.imshow(preview)
    #print(preview.shape)
    
    print(y.shape)

    breakpoint()
    print(batchid)
    1

breakpoint()