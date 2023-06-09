import torch
import cv2
import numpy as np
import math

import matplotlib.pyplot as plt

from models.cotr import COTR
from utils import transform_images, transform_query, plot_predictions, two_images_side_by_side
import torchvision.transforms.functional as TF

IMG_SIZE = 256

class simple_engine():
    def __init__(self, model):
        self.model = model.eval()

    def predict(self, img1, img2, query):
        oh1, ow1, oc1 = img1.shape
        oh2, ow2, oc2 = img2.shape

        breakpoint() # have to resize images ?
        
        orginal_query = np.copy(query)

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
        breakpoint()

        n = pred.shape[0]
        h2,w2, c2 = img2.shape
        for i in range(n):
            x,y = pred[i]        

            from_x = int(max(0, x-256))
            to_x = int(min(w2, x+256))
            from_y = int(max(0, y-256)) 
            to_y = int(min(h2, y+256))

            plt.imshow(img2[from_y:to_y, from_x:to_x, :])
            plt.show()

        return pred
    
    def preprocess(self, img1, img2, query, ow1, oh1):
        imgR = transform_images(img1, img2, IMG_SIZE)
        queryR = transform_query(query, ow1, oh1)
        return imgR, queryR

    def original_image_predict(self,img, query):
        # assume img is torch and concatenated already
        edge_threshold = 8 #px
        #b, c, h, w = img.shape

        img = img.numpy()[0].transpose(1,2,0)

        h, w, _ = img.shape

        img1 = img[:,:w//2,:]
        img2 = img[:, w//2:, :]
        
        h1,w1,_ = img1.shape
        h2,w2,_ = img2.shape

        min_x = query[:, 0].min()
        max_x = query[:, 0].max()

        min_y = query[:, 1].min()
        max_y = query[:, 1].max()

        w = max_x - min_x
        h = max_y - min_y

        ###

        n_w = w*1. / ((IMG_SIZE-edge_threshold)/2)
        n_w = math.ceil(n_w)

        ws_c = np.linspace(0, w, n_w+2)[1:-1]

        ws_s = [x-IMG_SIZE//2 for x in ws_c]
        ws_e = [x+IMG_SIZE//2 for x in ws_c]

        w_se = []
        for s,e in zip(ws_s, ws_e):
            if s < 0:
                diff = -s
                e = e + diff
                s = 0
            if e > w:
                diff = e - w
                s = s - diff
                e = w
            s = round(s+min_x)
            e = round(e+min_x)
            w_se.append((s,e))
        
        n_h = h*1. / ((IMG_SIZE-edge_threshold)/2)
        n_h = math.ceil(n_h)

        hs_c = np.linspace(0, h, n_h+2)[1:-1]

        hs_s = [y - IMG_SIZE//2 for y in hs_c]
        hs_e = [y + IMG_SIZE//2 for y in hs_c]

        h_se = []
        for s,e in zip(hs_s, hs_e):
            if s < 0:
                diff = -s
                e = e + diff
                s = 0
            if e > h:
                diff = e - h
                s = s - diff
                e = h
            s = round(s+min_y)
            e = round(e+min_y)
            h_se.append((s,e))
        
        ###

        # for img2 
        w2_se = [] # (s - IMG_SIZE//2, e + IMG_SIZE//2) for s,e in w_se ]
        for s,e in w_se:
            s = s - IMG_SIZE//2
            e = e# + IMG_SIZE//2
            if s < 0:
                s = 0
                e = int(1.5*IMG_SIZE)
            if e > w2:
                e = w2
                s = int(e - 1.5*IMG_SIZE)
            w2_se.append((s,e))
        h2_se = h_se.copy()

        # w_se, h_se, w2_se, h2_se - patches on both images to work with

        def patch_pass(model, p1, p2, query):
            device = next(model.parameters()).device
            assert p1.shape[0] == p2.shape[0]
            assert p1.shape[1] == p2.shape[1]
            # transforms
            p = two_images_side_by_side(p1,p2)
            #p = TF.to_tensor(p).float()[None]
            p = TF.normalize(TF.to_tensor(p), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).float()[None]
            p = p.to(device)
            query = torch.from_numpy(query)[None].float().to(device)

            pred = model(p, query)['pred_corrs'].clone().detach()

            plot_predictions(p, query, pred, pred, 'engine_test', 'plot_test')
            
            p_rev = torch.cat([p[..., IMG_SIZE:],p[..., :IMG_SIZE]], axis=-1)
            pred[..., 0] = pred[..., 0] - 0.5 
            cycle = model(p_rev, pred)['pred_corrs'].clone().detach()

            query = query.cpu().numpy()[0]
            pred = pred.cpu().numpy()[0]
            cycle = cycle.cpu().numpy()[0]

            conf = np.linalg.norm(query - cycle, axis=1, keepdims=True)
            return np.concatenate([pred, conf], axis=1)
        
        n = len(w_se)
        m = len(h_se)

        final_list = []
        for j in range(m):
            for i in range(n):
                s1,e1 = w_se[i]
                s2,e2 = w2_se[i]
                sh,eh = h_se[j]

                pred_list = []
                for part in range(int(((e2-s2)/(IMG_SIZE//2))-1)):
                    print(part*(IMG_SIZE//2), part*(IMG_SIZE//2)+IMG_SIZE)
                    img_in1 = img1[sh:eh,s1:e1,:]
                    img_in2 = img2[sh:eh,s2+part*(IMG_SIZE//2):s2+part*(IMG_SIZE//2)+IMG_SIZE,:]
                    query_in = query.copy()
                    mask = (query_in[:, 0] >= s1) & (query_in[:, 1] >= sh) & (query_in[:, 0] <= e1) & (query_in[:, 1] <= eh)
                    query_in =  query_in.astype(np.float32)
                    query_in[:, 0] -= s1
                    query_in[:, 1] -= sh
                    query_in[:, 0] /= (2*(e1 - s1))
                    query_in[:, 1] /= (eh - sh)
                    pred = patch_pass(self.model, img_in1, img_in2, query_in)
                    pred[~mask, 2] = np.inf
                    #pred[:, 0] -= 0.5
                    pred[:, 0] *= 2 * IMG_SIZE
                    pred[:, 0] += (s2 + part*(IMG_SIZE//2))
                    pred[:, 1] *= (eh - sh)
                    pred[:, 1] += sh
                    pred_list.append(pred)


                pred_list = np.stack(pred_list).transpose(1,0,2)
                out = []
                for item in pred_list:
                    out.append(item[np.argmin(item[..., 2], axis=0)])
                out = np.array(out)
                final_list.append(out)
        
        final_list = np.stack(final_list).transpose(1,0,2)
        final_out = []
        for item in final_list:
            final_out.append(item[np.argmin(item[..., 2], axis=0)])
        final_out = np.array(final_out)
        final_out = final_out[:, :2]
        
        return np.round(np.concatenate([query, final_out], axis=1)).astype(np.int)
        

    def simple_predict(self, img1, img2, query):
        oh1, ow1, oc1 = img1.shape
        oh2, ow2, oc2 = img2.shape

        img, query = self.preprocess(img1, img2, query, ow1, oh1)
        img = img.unsqueeze(0)
        query = torch.tensor(query)

        pred = self.model(img, query)['pred_corrs']
        return pred

    def tiling_predict(self, img1, img2, query):
        oh1, ow1, oc1 = img1.shape
        oh2, ow2, oc2 = img2.shape
        pass

    def interpolation_disparity_predict(self, img1, img2=None):
        if img2 is None:
           img = img1 
        else:
            img = transform_images(img1, img2, IMG_SIZE)[None]
        device = next(self.model.parameters()).device
        q_list = []
        for i in range(IMG_SIZE):
            queries = []
            for j in range(IMG_SIZE):
                queries.append([j / (IMG_SIZE * 2), i / IMG_SIZE])
            queries = np.array(queries)
            q_list.append(queries)
        
        out_list = []
        for q in q_list:
            queries = torch.from_numpy(q)[None].float().to(device)
            out = self.model.forward(img, queries)['pred_corrs'].detach().cpu().numpy()[0]
            out_list.append(out)
        out_list = np.array(out_list)


        tmp = np.array(q_list)
        disp =  tmp[:,:,0] - (out_list[:,:,0] - 0.5)

        temp = np.round(disp * 2 * 256)
        breakpoint()
        '''
        temp = temp + np.abs(temp.min())
        msk = temp > 0
        temp[~msk] = 0 
        '''
        
        plt.imshow(temp, cmap='gray')
        plt.show()

        return disp

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
                for j in range(IMG_SIZE):
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

    def patch_pass(self, model, img1, img2):
        device = next(model.parameters()).device
        pass
