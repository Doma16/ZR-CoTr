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

def two_images_vertical(img1, img2):
    assert img1.shape == img2.shape
    assert img1.dtype == img2.dtype
    h, w, c = img1.shape
    
    canvas = np.zeros((h*2, w, c), dtype=img1.dtype)
    canvas[0*h:1*h, :, :] = img1
    canvas[1*h:2*h, :, :] = img2

    return canvas

def plot_real(img, query, pred):

    img = img.cpu().detach().numpy()
    query = query.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()

    img = np.copy(img)
    query = np.copy(query)
    pred = np.copy(pred)

    b,h,w,c = img.shape
    img = img[0]
    query = query[0]
    pred = pred[0]

    num_show = 10

    if query.shape[0] > num_show:
        query = query[:num_show, :]
        pred = pred[:num_show, :]

    query[:, 0] = np.round(query[:, 0] * 2*w)
    pred[:, 0] = np.round((pred[:, 0]-0.5) * 2*w)
    
    query[:, 1] = np.round(query[:, 1] * h//2)
    pred[:, 1] = np.round(pred[:, 1] * h//2 + h//2)

    col = [0.0,0.6,0.1]

    lw = 1
    alpha = 1

    x_q = query[:, 0]
    y_q = query[:, 1]

    x_p = pred[:, 0]
    y_p = pred[:, 1]

    plt.imshow(img)

    X_true = np.stack([x_q, x_p])
    Y_true = np.stack([y_q, y_p])

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

    num_show = 10

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

def test():
    pass

if __name__ == '__main__':
    test()