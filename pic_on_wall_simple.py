import torch
import numpy as np
import imageio
import cv2

import matplotlib.pyplot as plt

from models.cotr import COTR
from inference.simple_engine import simple_engine

PATH_MODEL = './saved/ep400_bid39.pth'

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

    img_a = imageio.imread('./pics/imgs/000020_10.png', pilmode='RGB')
    img_b = imageio.imread('./pics/imgs/000020_11.png', pilmode='RGB')
    rep_img = imageio.imread('./pics/imgs/Meisje_met_de_parel.jpg', pilmode='RGB')
    rep_mask = np.ones(rep_img.shape[:2])
    #plt.imshow(img_a)
    #plt.show()
    lu_corner = [490, 218]
    ru_corner = [580, 217]
    lb_corner = [280, 353]
    rb_corner = [654, 357]
    query = np.array([lu_corner,ru_corner,lb_corner, rb_corner]).astype(np.float32)
    rep_h, rep_w, _ = rep_img.shape
    rep_coord = np.array([[0,0], [rep_w,0], [0,rep_h], [rep_w, rep_h]]).astype(np.float32)
    
    engine = simple_engine(model)
    pred = engine.predict(img_a, img_b, query)
    '''
    engine = SparseEngine(model, 32, mode='stretching')
    corrs = engine.cotr_corr_multiscale(img_a, img_b, np.linspace(0.5, 0.0625, 4), 1, queries_a=query, force=True)
    '''
    breakpoint()
    T = cv2.getPerspectiveTransform(rep_coord, pred)#corrs[:, 2:].astype(np.float32))
    vmask = cv2.warpPerspective(rep_mask, T, (img_b.shape[1], img_b.shape[0])) > 0
    warped = cv2.warpPerspective(rep_img, T, (img_b.shape[1], img_b.shape[0]))
    out = warped * vmask[..., None] + img_b * (~vmask[..., None])

    f, axarr = plt.subplots(1, 3)
    #axarr[0].imshow(rep_img)
    #axarr[0].title.set_text('Virtual Paint')
    #axarr[0].axis('off')
    axarr[0].imshow(img_a)
    axarr[0].title.set_text('Annotated Frame')
    axarr[0].axis('off')
    axarr[1].imshow(img_b)
    axarr[1].title.set_text('Target Frame')
    axarr[1].axis('off')
    axarr[2].imshow(out)
    axarr[2].title.set_text('Overlay')
    axarr[2].axis('off')
    plt.show()

if __name__ == '__main__':
    main()