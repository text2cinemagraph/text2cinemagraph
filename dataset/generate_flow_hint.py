import glob
import torch
import argparse
import numpy as np
import numpy
from PIL import Image
import os
import argparse
import dill as pickle
import lz4framed
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import pickle, os
import multiprocessing

def generate_flow(flow, mask, n_clusters=5):

    gt_motion = torch.tensor(flow).permute(2,0,1).unsqueeze(0)
    big_motion_alpha = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
    height, width = gt_motion.shape[2], gt_motion.shape[3]

    xs = torch.linspace(0, width - 1, width)
    ys = torch.linspace(0, height - 1, height)
    xs = xs.view(1, 1, width).repeat(1, height, 1)
    ys = ys.view(1, height, 1).repeat(1, 1, width)
    xys = torch.cat((xs, ys), 1).view(2, -1)  # (2,WW)

    max_hint = n_clusters
    np.random.seed(5)
    estimator = KMeans(n_clusters=max_hint)
    hint_y = torch.zeros((max_hint,))
    hint_x = torch.zeros((max_hint,))
    big_motion_xys = xys[:, torch.where(big_motion_alpha.view(1, 1, height * width))[2]]  # 2, M
    X = big_motion_xys.permute(1, 0).cpu().detach().numpy()
    estimator.fit(X)
    labels = estimator.labels_  
    for i in range(max_hint):
        selected_xy = X[labels == i].mean(0)
        hint_y[i] = int(selected_xy[1])
        hint_x[i] = int(selected_xy[0])

    dense_motion = torch.zeros(gt_motion.shape).view(1, 2, -1)
    dense_motion_norm = torch.zeros(gt_motion.shape).view(1, 2, -1)

    sigma = height / max_hint #/ 2
    hint_y = hint_y.long()
    hint_x = hint_x.long()
    for i_hint in range(max_hint):
        dist = ((xys - xys.view(2, height, width)[:, hint_y[i_hint], hint_x[i_hint]].unsqueeze(
            1)) ** 2).sum(0, True).sqrt()  # 1,W*W
        weight = (-(dist / sigma) ** 2).exp().unsqueeze(0)
        dense_motion += weight * gt_motion[:, :, hint_y[i_hint], hint_x[i_hint], ].unsqueeze(2)
        dense_motion_norm += weight

    dense_motion_norm[dense_motion_norm == 0.0] = 1.0  # = torch.clamp(dense_motion_norm,min=1e-8)
    dense_motion = dense_motion / dense_motion_norm
    dense_motion = dense_motion.view(1, 2, height, width) * big_motion_alpha
    hint = dense_motion

    return hint


def generate_mask(flow, method, ratio=10.0):
    mask = np.zeros((flow.shape[0], flow.shape[1]))
    if method == 'zeros':
        mask = ((flow[:,:,0]!=0.0) | (flow[:,:,1]!=0.0))*1.0
    elif method == 'average':
        mean_flow = np.sum(np.square(flow))/(flow.shape[0]*flow.shape[1])
        mask = (np.sum(np.square(flow), axis=2)>mean_flow/ratio)*1.0
    return mask
    
def load_compressed_tensor(filename):
    retval = None
    with open(filename, mode='rb') as file:
        retval = torch.from_numpy(pickle.loads(lz4framed.decompress(file.read())))
    return retval

PRINT_TIME = False
MAX_WORKERS = multiprocessing.cpu_count()

def generate_input_flow(flow_path, save_folder, n_clusters):
    flowname = flow_path.split('/')[-1].replace('_motion.pth','')
    flow = load_compressed_tensor(flow_path).squeeze(0)
    npyflow = np.array(flow.permute(1,2,0))
    flow_mask = generate_mask(npyflow, 'average', 10.0)
    for cluster in range(1, n_clusters+1):
        try:
            input_flow = generate_flow(npyflow, flow_mask, n_clusters=cluster)
        except:
            input_flow = np.zeros(npyflow.shape)
            input_flow = torch.from_numpy(input_flow).permute(2,0,1)
        input_flow_tensor = input_flow[0]
        if not os.path.isdir(os.path.join(save_folder, flowname)):
            os.makedirs(os.path.join(save_folder, flowname), exist_ok=True)
        savename = os.path.join(save_folder, flowname, f'{cluster}_blend.pth')
        torch.save(input_flow_tensor, savename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, help="path to directory")
    parser.add_argument('--save_path', type=str, help="directory to save the dataset")
    parser.add_argument('--n_clusters', type=int, default=5)
    opt = parser.parse_args()

    ###generate input flow for all train motion (.pth) files
    flow_files = glob.glob(opt.dataroot + '/*.pth')
    print(len(flow_files))

    MAX_WORKERS = int(min(MAX_WORKERS, len(flow_files)))

    save_folder = opt.save_path
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    Parallel(n_jobs=MAX_WORKERS, verbose=2)(delayed(generate_input_flow)(flow_path, save_folder, n_clusters=opt.n_clusters) for flow_path in flow_files)