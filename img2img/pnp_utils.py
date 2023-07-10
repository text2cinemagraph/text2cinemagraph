import os, sys
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
from torchvision import transforms as T
from math import sqrt
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image
import torch
from einops import rearrange
from omegaconf import OmegaConf
import json
from tqdm import tqdm
import glob
from torch import einsum
from sklearn.cluster import KMeans,SpectralClustering
from skimage.measure import label
import cv2
# sys.path.append('img2img')
from img2img.run_features_extraction import load_model_from_config

def visualize_and_save_features_pca(feature_maps_fit_data,feature_maps_transform_data, transform_experiments, t, save_dir):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feature_maps_fit_data)
    feature_maps_pca = pca.transform(feature_maps_transform_data.cpu().numpy())  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(len(transform_experiments), -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(transform_experiments):
        pca_img = feature_maps_pca[i]  # (H * W) x 3
        h = w = int(sqrt(pca_img.shape[0]))
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
        pca_img.save(os.path.join(save_dir, f"{experiment}_time_{t}.png"))

def visualize_and_save_features_pca_custom(feature_maps_fit_data,feature_maps_transform_data, save_path):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feature_maps_fit_data)
    feature_maps_pca = pca.transform(feature_maps_transform_data.cpu().numpy())  # N X 3
    h = w = int(sqrt(feature_maps_pca.shape[0]))
    feature_maps_pca = feature_maps_pca.reshape(h, w, 3)
    torch.save(feature_maps_pca, save_path)

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def load_experiments_features(feature_maps_path):
    feature_maps = []
    feature_map = torch.load(os.path.join(feature_maps_path))[1]
    feature_map = feature_map.reshape(feature_map.shape[0], -1).t()  # N X C
    feature_maps.append(feature_map)

    return feature_maps

def load_experiments_self_attn_maps(unet_model, feature_map_path, block, t):
    self_attn_maps = []
    block_idx = int(block.split('_')[-1])
    self_attn_q = torch.load(os.path.join(feature_map_path, f"{block}_self_attn_q_time_{t}.pt"))
    self_attn_k = torch.load(os.path.join(feature_map_path, f"{block}_self_attn_k_time_{t}.pt"))
    if "output_block" in block:
        scale = unet_model.output_blocks[block_idx][1].transformer_blocks[0].attn1.scale
    else:
        scale = unet_model.input_blocks[block_idx][1].transformer_blocks[0].attn1.scale
    sim = einsum('b i d, b j d -> b i j', self_attn_q, self_attn_k) * scale
    self_attn_map = sim.softmax(dim=-1)
    self_attn_map = rearrange(self_attn_map, 'h n m -> n (h m)')
    self_attn_maps.append(self_attn_map)

    return self_attn_maps

def generate_attn_mask(exp_config, model=None):

    exp_path_root = exp_config.twin_extraction.exp_path_root
    model_config = OmegaConf.load(f"{exp_config.twin_extraction.model_config}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if model is None:
        model = load_model_from_config(model_config, exp_config.twin_extraction.ckpt)
        model = model.to(device)
    unet_model = model.model.diffusion_model

    input_folder =os.path.join(exp_path_root, exp_config.twin_extraction.experiment_name, "feature_maps")
    features_all = []
    path = os.path.join(input_folder, 'output_block_11_self_attn_q_time_*.pt')
    maps = glob.glob(path)
    steps = [int(map.split('_')[-1].replace('.pt', '')) for map in maps]
    steps = [step for step in steps if step <= 541]

    for step in steps:
        for i in exp_config.attn_mask.input_blocks:
            block_name = 'input_block_{}'.format(i)
            feature_path = os.path.join(input_folder)
            fit_features = load_experiments_self_attn_maps(unet_model, feature_path, block_name, step)
            fit_features = torch.cat(fit_features, dim=0)
            features_all.append(fit_features.unsqueeze(0))
        for i in exp_config.attn_mask.output_blocks:
            block_name = 'output_block_{}'.format(i)
            feature_path = os.path.join(input_folder)
            fit_features = load_experiments_self_attn_maps(unet_model, feature_path, block_name, step)
            fit_features = torch.cat(fit_features, dim=0)
            features_all.append(fit_features.unsqueeze(0))
        # break
    features_all = torch.cat(features_all, 0)
    features_all = features_all.mean(0)

    res = exp_config.attn_mask.res
    cluster_n = exp_config.attn_mask.n_clusters

    if exp_config.attn_mask.cluster_type == 'kmeans':
        kmeans = KMeans(n_clusters=cluster_n, n_init=5).fit(features_all.cpu().numpy())
        clusters = kmeans.labels_
        clusters = clusters.reshape(res, res)
    elif exp_config.attn_mask.cluster_type == 'spectral':
        feats = features_all.reshape(res**2, 16, res**2).mean(1).cpu().numpy()
        sc = SpectralClustering(cluster_n, affinity='precomputed', n_init=100,
                                assign_labels='kmeans')
        clusters = sc.fit_predict(feats)
        clusters = clusters.reshape(res, res)
    else:
        print('Choose one clustering type between kmeans and spectral!')
        return
    clusters = label(clusters)

    mask = Image.open(os.path.join(exp_path_root, exp_config.twin_extraction.experiment_name, 'mask_odise.png')).convert('L')
    mask = np.array(mask) / 255.
    labels_img = Image.fromarray(clusters.astype(np.int32)).convert('L').resize((512,512), Image.NEAREST)
    labels_img = np.array(labels_img)
    labels_img = labels_img + 1
    labels_uni = np.unique(labels_img)
    label_cnt = {}
    labels_cnt_tot = {}
    labels_img_masked = labels_img * mask
    for label_ in labels_uni:
        label_cnt[label_] = np.sum((labels_img_masked==label_)*1)
        labels_cnt_tot[label_] = np.sum((labels_img==label_)*1)
    mask_f = np.zeros_like(mask)
    labels_f = []
    for label_ in labels_uni:
        if label_cnt[label_] / labels_cnt_tot[label_] >= exp_config.attn_mask.threshold:
            labels_f.append(label_)
            mask_f[labels_img==label_] = 1.
    # mask_f = mask_f * mask
    mask_f = Image.fromarray((mask_f * 255).astype(np.uint8)).convert('L')
    mask_f.save(os.path.join(exp_path_root, exp_config.twin_extraction.experiment_name, 'mask_self_attn.png'))

    mask_f = np.array(mask_f)
    kernel = np.ones((exp_config.attn_mask.erosion_ksize, exp_config.attn_mask.erosion_ksize), np.uint8)
    img_erosion = cv2.erode(mask_f, kernel, iterations=exp_config.attn_mask.erosion_iter)
    erosion = Image.fromarray(img_erosion).convert('L')
    erosion.save(os.path.join(exp_path_root, exp_config.twin_extraction.experiment_name, 'mask_self_attn_erosion.png'))
