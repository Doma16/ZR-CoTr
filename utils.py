import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import cv2

IMAGE_SIZE = 256

def get_query(dmap, n_queries=1):
    query_points = []
    indicies = dmap > 0
    b, c, h, w = dmap.shape
    assert c == 1
    assert b == 1
        
    '''
    Treba nekako samplat pointove po nekom omjeru iz svakog kuta slike
    , ako postoje naravno. npr. 120x30 kvadratici po slici cca. 10x10 puta
    '''
    
    breakpoint()
    return query_points

def two_images_side_by_side(img1, img2):
    assert img1.shape == img2.shape
    assert img1.dtype == img2.dtype
    h, w, c = img1.shape
    
    canvas = np.zeros((h, 2 * w, c), dtype=img1.dtype)
    canvas[:, 0 * w:1 * w, :] = img1
    canvas[:, 1 * w:2 * w, :] = img2

    return canvas


def plot_predictions(img, query, pred, target, b_id, file):
    img = img.cpu().detach().numpy()
    query = query.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    b, c, h, w = img.shape
    ow = w // 2
    img = img[0].transpose(1,2,0)
    query = query[0]
    pred = pred[0]
    target = target[0]

    if query.shape[0] > 10:
        query = query[:10,:]
        pred = pred[:10,:]
        target = target[:10, :]

    img = cv2.UMat(img)

    # We will only plot pred > 0
    pred_mask = pred > 0
    for i, p in enumerate(pred):
        if pred_mask[i].all() == True:
            x1 = int(query[i][0] * ow)
            y1 = int(query[i][1] * h)
            x2 = int(pred[i][0] * ow + ow) 
            y2 = int(pred[i][1] * h)
            xt = int(target[i][0] * ow + ow)
            yt = int(target[i][1] * h)
            cv2.line(img, (x1,y1), (x2,y2), (0,100,0), 2)
            cv2.line(img, (x1,y1), (xt, yt), (100,0,0), 2)

    img = img.get()
    plt.imshow(img)
    plt.savefig(f'./{file}/{b_id}.png')


def PCK_N(img, query, pred, target, save_name=None, file=None): # 1px 3px 5px (percentage of correct keypoints)
    pass

def AEPE(img, query, pred, target, save_name=None, file=None): # average end point error
    pass

def F1(img, query, pred, target, save_name=None, file=None): # TODO
    pass