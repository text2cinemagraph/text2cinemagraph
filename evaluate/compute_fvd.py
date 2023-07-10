import numpy as np
import torch
import glob
import our_fvd
from util import open_url
import pdb
import tqdm
from PIL import Image
import decord
decord.bridge.set_bridge('torch')
import os
import argparse

@torch.no_grad()
def compute_our_fvd(videos_fake, videos_real, device='cuda'):
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.

    with open_url(detector_url, verbose=False) as f:
        detector = torch.jit.load(f).eval().to(device)

    videos_fake = videos_fake.permute(0, 4, 1, 2, 3).to(device)
    videos_real = videos_real.permute(0, 4, 1, 2, 3).to(device)
    feats_fake_all = []
    feats_real_all = []
    for i in range(0,videos_fake.shape[0], 8):
        en = min(i+8, videos_fake.shape[0])
        feats_fake = detector(videos_fake[i:en], **detector_kwargs)
        feats_fake_all.append(feats_fake)

        feats_real = detector(videos_real[i:en], **detector_kwargs)
        feats_real_all.append(feats_real)

    feats_fake_all = torch.cat(feats_fake_all, 0)
    feats_real_all = torch.cat(feats_real_all, 0)
    
    return our_fvd.compute_fvd(feats_fake_all.cpu().numpy(), feats_real_all.cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_path",
        type=str,
        help="path to the folder containing generated videos"
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        help="path to the folder containing ground-truth videos"
    )
    parser.add_argument(
        "--type",
        type=str,
        default='fvd_16',
        choices=['fvd_16', 'fvd_60']
    )

    opt = parser.parse_args()
    path_pred = opt.pred_path
    path_gt = opt.gt_path
    files = glob.glob(path_gt+'/*.mp4')
    files.sort()

    tot_lipis = []
    gt_videos = []
    pred_videos = []
    for file in tqdm.tqdm(files):
        gt_vid = decord.VideoReader(file, width=224, height=224)
        if opt.type == 'fvd_16':
            sample_index = list(range(0, len(gt_vid), 1))[::3][:16]
        else:
            sample_index = list(range(0, len(gt_vid), 1))[:60]
        gt_video = gt_vid.get_batch(sample_index)
        gt_videos.append(gt_video.unsqueeze(0))

        name  = file.split('/')[-1].replace('_gt.mp4', '')
        pred_file = os.path.join(path_pred, f'{name}_video.mp4')
        pred_vid = decord.VideoReader(pred_file, width=224, height=224)
        if opt.type == 'fvd_16':
            sample_index = list(range(0, len(pred_vid), 1))[::3][:16]
        else:
            sample_index = list(range(0, len(pred_vid), 1))[:60]
        pred_video = pred_vid.get_batch(sample_index)
        pred_videos.append(pred_video.unsqueeze(0))

    gt_videos = torch.cat(gt_videos, 0)
    pred_videos = torch.cat(pred_videos, 0)

    gt_videos = gt_videos / 127.5 - 1.
    pred_videos = pred_videos / 127.5 - 1.

    print(gt_videos.shape)
    print(pred_videos.shape)

    fvd = compute_our_fvd(pred_videos, gt_videos)
    print('FVD is: ', fvd)