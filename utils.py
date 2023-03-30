import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import numpy as np

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