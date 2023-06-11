import torch
import numpy as np
import imageio
import cv2

import matplotlib.pyplot as plt

from models.cotr import COTR
from utils import plot_predictions, plot_real
from inference.simple_engine import simple_engine


from dataset import KittiDataset, MiddleBury
from torch.utils.data import DataLoader

PATH_MODEL = './saved/ep400_bid39.pth'

EMB_DIM = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
RETURN_INTERMEDIATE = True
DROPOUT = 0.1
NLAYERS = 3

NUM_KP = 100
BATCH_SIZE = 1

if __name__ == '__main__':

    model = COTR(
        emd_dim=EMB_DIM,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        return_intermediate=RETURN_INTERMEDIATE,
        dropout=DROPOUT,
        nlayers=NLAYERS
    )
    model.load_state_dict(torch.load(PATH_MODEL))
    model = model.eval()

    dataset = KittiDataset(root= '../dataset', transforms='original', split='val', num_kp=NUM_KP)
    #dataset = MiddleBury(root='../dataset', transforms = 'original', num_kp=NUM_KP)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    engine = simple_engine(model)

    for batchid, (img, w, corrs) in enumerate(loader):
        
        # DO SOMETHING WITH ENGINE
        dmap = w
        dmap = dmap.numpy()[0]

        b, c, h, w = img.shape

        q_list = []
        for i in range(h):
            queries = []
            for j in range(w//2):
                queries.append([j, i])
            queries = np.array(queries)
            q_list.append(queries)

        query = np.concatenate(q_list)

        ind = np.argwhere(dmap > 0)

        ind2 = ind.copy()
        ind2[:, 1] = np.round(ind2[:, 1] - dmap[dmap>0]).astype(np.int32)
        ind[:, [0,1]] = ind[:, [1,0]]
        ind2[:, [0,1]] = ind2[:, [1,0]]

        #mask_queries out of picture
        mask_q = (ind2[:, 0] > 0) & (ind2[:, 0] < w//2)
        ind = ind[mask_q]
        ind2 = ind2[mask_q]

        query = ind
        mask = np.random.choice(query.shape[0], NUM_KP)
        query = query[mask,:]
        ind2 = ind2[mask, :]

        query_pred = engine.original_image_predict(img, query)
        plot_real(img, query_pred[:,:2], query_pred[:,2:], query_pred[:,2:])
        breakpoint()
        print(batchid, img.shape)
        pass