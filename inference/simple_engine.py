import torch
import cv2
import numpy as np

from models.cotr import COTR
from utils import transform_images, transform_query, plot_predictions, two_images_side_by_side

IMG_SIZE = 256

class simple_engine():
    def __init__(self, model):
        self.model = model.eval()

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
    
    def cotr_flow_exhaustive(self, img_a, img_b):
        def one_pass(model, img_a, img_b):
            device = next(model.parameters()).device
            img = transform_images(img_a, img_b, IMG_SIZE)
            img = img.to(device)
            print('check shape:', img.shape)
            breakpoint()

            q_list = []
            for i in range(IMG_SIZE):
                queries = []
                for j in range(IMG_SIZE * 2):
                    queries.append([j / (IMG_SIZE * 2), i / IMG_SIZE])
                queries = np.array(queries)
                q_list.append(queries)

            out_list = []
            for q in q_list:
                queries = torch.from_numpy(q)[None].float().to(device)
                out = model.forward(img, queries)['pred_corrs'].detach().cpu().numpy()[0]
                out_list.append(out)
            out_list = np.array(out_list)
            in_grid = torch.from_numpy(np.array(q_list)).float()[None] * 2 - 1
            out_grid = torch.from_numpy(out_list).float()[None] * 2 - 1
            cycle_grid = torch.nn.functional.grid_sample(out_grid.permute(0, 3, 1, 2), out_grid).permute(0, 2, 3, 1)
            confidence = torch.norm(cycle_grid[0, ...] - in_grid[0, ...], dim=-1)
            corr = out_grid[0].clone()
            corr[:, :IMG_SIZE, 0] = corr[:, :IMG_SIZE, 0] * 2 - 1
            corr[:, IMG_SIZE:, 0] = corr[: IMG_SIZE:, 0] * 2 + 1
            corr = torch.cat([corr, confidence[..., None]], dim=-1).numpy()
            return corr[:, :IMG_SIZE, :], corr[:, IMG_SIZE:, :]
    
        corrs_a = []
        corrs_b = []
        breakpoint()


