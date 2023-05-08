import torch
import cv2
import numpy as np

from models.cotr import COTR
from utils import transform_images, transform_query, plot_predictions

IMG_SIZE = 256

class simple_engine():
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, img1, img2, query):
        oh1, ow1, oc1 = img1.shape
        oh2, ow2, oc2 = img2.shape

        breakpoint() # have to resize images ?
        
        img, query = self.preprocess(img1,img2, query, ow1, oh1)
        img = img.unsqueeze(0)
        query = torch.tensor(query)
        
        pred = self.model(img, query)['pred_corrs']
        plot_predictions(img, query, pred=pred,  target=pred, b_id='testingtest', file='plot_test')        

        pred = pred.detach().numpy()
        pred = pred.reshape(pred.shape[1:])

        pred[:,0] = pred[:,0] - 0.5
        pred[:,0] *= (2*ow2)
        pred[:,1] *= oh2
        pred = np.round(pred)

        return pred
    
    def preprocess(self, img1, img2, query, ow1, oh1):
        imgR = transform_images(img1, img2, IMG_SIZE)
        queryR = transform_query(query, ow1, oh1)
        return imgR, queryR
    
