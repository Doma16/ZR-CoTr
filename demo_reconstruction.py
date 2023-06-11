
import numpy as np
import torch
import imageio
import open3d as o3d

import utils
from models import COTR

def triangulate_rays_to_pcd(center_a, dir_a, center_b, dir_b):
    A = center_a
    a = dir_a / np.linalg.norm(dir_a, axis=1, keepdims=True)
    B = center_b
    b = dir_b / np.linalg.norm(dir_b, axis=1, keepdims=True)
    c = B - A
    D = A + a * ((-np.sum(a * b, axis=1) * np.sum(b * c, axis=1) + np.sum(a * c, axis=1) * np.sum(b * b, axis=1)) / (np.sum(a * a, axis=1) * np.sum(b * b, axis=1) - np.sum(a * b, axis=1) * np.sum(a * b, axis=1)))[..., None]
    return D

def main():
    model = COTR()
    model = model.cuda()
    
    model = model.eval()