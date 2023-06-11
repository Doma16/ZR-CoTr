import torch
import numpy as np
import cv2
import imageio

import matplotlib.pyplot as plt

from models.cotr import COTR
from inference.simple_engine import simple_engine

PATH_MODEL = './saved/ep400_bid39_zoom.pth'

def main():
    model = COTR(
        emd_dim=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        return_intermediate=True,
        dropout=0.1,
        nlayers=3
    )
    model.load_state_dict(torch.load(PATH_MODEL))
    #grab image tile it and try it
    engine = simple_engine(model)
    img1 = imageio.imread('./pics/imgs/000020_10.png', pilmode='RGB')
    img2 = imageio.imread('./pics/imgs/000020_11.png', pilmode='RGB')
    engine.interpolation_disparity_predict(img1, img2)


if __name__ == '__main__':
    main()