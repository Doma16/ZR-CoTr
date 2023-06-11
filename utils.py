import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL

from collections import namedtuple

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

def two_images_vertical(img1, img2):
    assert img1.shape == img2.shape
    assert img1.dtype == img2.dtype
    h, w, c = img1.shape
    
    canvas = np.zeros((h*2, w, c), dtype=img1.dtype)
    canvas[0*h:1*h, :, :] = img1
    canvas[1*h:2*h, :, :] = img2

    return canvas

def plot_real(img, query, pred, real):

    img = img.cpu().detach().numpy()

    img = np.copy(img)
    query = np.copy(query)
    pred = np.copy(pred)
    real = np.copy(real)

    b,c,h,w = img.shape
    img = img[0]

    img = img.transpose(1,2,0)
    img = two_images_vertical(img[:,:w//2,:], img[:,w//2:,:])

    num_show = 30

    if query.shape[0] > num_show:
        query = query[:num_show, :]
        pred = pred[:num_show, :]
        real = real[:num_show, :]

    query[:, 0] = np.round(query[:, 0])
    pred[:, 0] = np.round(pred[:, 0])
    real[:, 0] = np.round(real[:, 0])
    
    query[:, 1] = np.round(query[:, 1])
    pred[:, 1] = np.round(pred[:, 1]+h)
    real[:, 1] = np.round(real[:, 1]+h)

    col = [0.0,0.6,0.1]

    lw = 1
    alpha = 1

    x_q = query[:, 0]
    y_q = query[:, 1]

    x_p = pred[:, 0]
    y_p = pred[:, 1]

    x_r = real[:, 0]
    y_r = real[:, 1]

    plt.imshow(img)

    X_true = np.stack([x_q, x_p])
    Y_true = np.stack([y_q, y_p])

    X_real = np.stack([x_q, x_r])
    Y_real = np.stack([y_q, y_r])

    '''
    plt.plot(
        X_real, Y_real,
        alpha=alpha,
        linestyle='-',
        linewidth=lw,
        aa=False,
        color=[0.6, 0.1, 0],
    )
    plt.scatter(X_real, Y_real)
    '''
    
    plt.plot(
        X_true, Y_true,
        alpha=alpha,
        linestyle='-',
        linewidth=lw,
        aa=False,
        color=col,
    )
    plt.scatter(X_true, Y_true)

    plt.show()



def plot_predictions(img, query, pred, target, b_id, file):
    
    img = img.cpu().detach().numpy()
    query = query.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    img = np.copy(img)
    query = np.copy(query)
    pred = np.copy(pred)
    target = np.copy(target)

    b, c, h, w = img.shape
    img = img[0].transpose(1,2,0)
    query = query[0]
    pred = pred[0]
    target = target[0]

    num_show = 20

    if query.shape[0] > num_show:
        query = query[:num_show,:]
        pred = pred[:num_show,:]
        target = target[:num_show, :]

    query[:, 0] = np.round(query[:, 0] * w)
    pred[:, 0] = np.round(pred[:, 0] * w)
    target[:, 0] = np.round(target[:, 0] * w)

    query[:, 1] = np.round(query[:, 1] * h)
    pred[:, 1] = np.round(pred[:, 1] * h)
    target[:, 1] = np.round(target[:, 1] * h)

    cols = [
        [0.0, 0.67, 0.0],
        [0.9, 0.1, 0.0],
    ]

    lw = 0.5
    alpha = 1

    x_q = query[:, 0]
    y_q = query[:, 1]

    x_t = target[:, 0]
    y_t = target[:, 1]

    x_p = pred[:, 0]
    y_p = pred[:, 1]

    plt.imshow(img)

    X_true = np.stack([x_q, x_t])
    Y_true = np.stack([y_q, y_t])

    plt.plot(
        X_true, Y_true,
        alpha=alpha,
        linestyle='-',
        linewidth=lw,
        aa=False,
        color=cols[1],
    )

    X_pred = np.stack([x_q, x_p])
    Y_pred = np.stack([y_q, y_p])

    plt.plot(
        X_pred, Y_pred,
        alpha=alpha,
        linestyle='-',
        linewidth=lw,
        aa=False,
        color=cols[0]
    )

    X = np.stack([x_q, x_p, x_t])
    Y = np.stack([y_q, y_p, y_t])
    plt.scatter(X, Y)

    plt.savefig(f'./{file}/{b_id}.png')
    plt.clf()


def PCK_N(img, query, pred, target, threshold=1): # example for: 1px 3px 5px (percentage of correct keypoints)
    b_size,c,h,w = img.shape

    a = np.copy( pred.cpu().detach().numpy() )
    b = np.copy( target.cpu().detach().numpy() )

    a[:,:,0] = np.round(a[:,:,0] * w)
    a[:,:,1] = np.round(a[:,:,1] * h)
    b[:,:,0] = np.round(b[:,:,0] * w)
    b[:,:,1] = np.round(b[:,:,1] * h)

    distance_xy = np.abs(a - b)
    assert len(distance_xy.shape) == 3
    
    distance = np.sqrt( np.square(distance_xy[:,:,0]) + np.square(distance_xy[:,:,1]) )
   
    b_size, num_kp, _ = pred.shape
    in_ = np.count_nonzero(distance <= threshold)

    pck = in_ / (b_size*num_kp)
    return pck

def AEPE(img, query, pred, target): # average end point error
    b_size,c,h,w = img.shape

    a = np.copy( pred.cpu().detach().numpy() )
    b = np.copy( target.cpu().detach().numpy() )
    
    a[:,:,0] = np.round(a[:,:,0] * w)
    a[:,:,1] = np.round(a[:,:,1] * h)
    b[:,:,0] = np.round(b[:,:,0] * w)
    b[:,:,1] = np.round(b[:,:,1] * h)

    distance_xy = np.abs(a - b)
    distance = np.sqrt( np.square(distance_xy[:,:,0]) + np.square(distance_xy[:,:,1]) )

    avg = np.mean(distance)
    return avg

def F1(img, query, pred, target): # TODO
    pass

def transform_images(img1, img2, size):
    img1 = np.array(img1)
    img2 = np.array(img2)

    new_size = (size, size)
    img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_LINEAR)

    imgR = two_images_side_by_side(img1, img2)

    imgR = TF.to_tensor(imgR)

    return imgR

def transform_query(query, ow, oh):
    query[:,0] = query[:,0] / (2*ow)
    query[:,1] = query[:,1] / (oh)
    query = query.reshape(1, query.shape[0], query.shape[1])
    return query

ImagePatch = namedtuple('ImagePatch', ['patch', 'x', 'y', 'w', 'h', 'ow', 'oh'])
Point3D = namedtuple('Point3D', ['id', 'arr_idx', 'image_ids'])
Point2D = namedtuple('Point2D', ['id_3d', 'xy'])

def float_image_resize(img, shape, interp=PIL.Image.BILINEAR):
    missing_channel = False
    if len(img.shape) == 2:
        missing_channel = True
        img = img[..., None]
    layers = []
    img = img.transpose(2,0,1)
    for l in img:
        l = np.array(PIL.Image.fromarray(l).resize(shape[::-1], resample=interp))
        assert l.shape[:2] == shape
        layers.append(l)
    if missing_channel:
        return np.stack(layers, axis=-1)[..., 0]
    else:
        return np.stack(layers, axis=-1)
    
def is_nan(x):
    return x != x

def has_nan(x):
    if x is None:
        return False
    return is_nan(x).any()

def confirm(question='OK to continue?'):
    answer = ''
    while answer not in ['y', 'n']:
        answer = input(question + '[y/n]').lower()
    return answer == 'y'

def torch_img_to_np_img(torch_img):
    '''convert a torch image to matplotlib-able numpy image
    torch use Channels x Height x Width
    numpy use Height x Width x Channels
    Arguments:
        torch_img {[type]} -- [description]
    '''
    assert isinstance(torch_img, torch.Tensor), 'cannot process data type: {0}'.format(type(torch_img))
    if len(torch_img.shape) == 4 and (torch_img.shape[1] == 3 or torch_img.shape[1] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (0, 2, 3, 1))
    if len(torch_img.shape) == 3 and (torch_img.shape[0] == 3 or torch_img.shape[0] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (1, 2, 0))
    elif len(torch_img.shape) == 2:
        return torch_img.detach().cpu().numpy()
    else:
        raise ValueError('cannot process this image')

def np_img_to_torch_img(np_img):
    """convert a numpy image to torch image
    numpy use Height x Width x Channels
    torch use Channels x Height x Width

    Arguments:
        np_img {[type]} -- [description]
    """
    assert isinstance(np_img, np.ndarray), 'cannot process data type: {0}'.format(type(np_img))
    if len(np_img.shape) == 4 and (np_img.shape[3] == 3 or np_img.shape[3] == 1):
        return torch.from_numpy(np.transpose(np_img, (0, 3, 1, 2)))
    if len(np_img.shape) == 3 and (np_img.shape[2] == 3 or np_img.shape[2] == 1):
        return torch.from_numpy(np.transpose(np_img, (2, 0, 1)))
    elif len(np_img.shape) == 2:
        return torch.from_numpy(np_img)
    else:
        raise ValueError('cannot process this image with shape: {0}'.format(np_img.shape))
