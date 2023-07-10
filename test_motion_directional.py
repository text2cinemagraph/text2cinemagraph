import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from models.sync_batchnorm import convert_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from util.flow_to_color import flow2img
import tqdm
import torch.nn.functional as F
import ntpath
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from util.util import load_image
from transformers import CLIPTokenizer
import glob
import numpy as np
from sklearn.cluster import KMeans
import random
import re
from omegaconf import OmegaConf

def find_dir_from_caption(caption):
    substrings = re.findall(r'(?=(in .*?direction))', caption)
    list_dirs = ['left to right',
                 'left to right, upwards',
                 'upwards, left to right',
                 'upwards',
                 'upwards, right to left',
                 'right to left, upwards',
                 'right to left',
                 'right to left, downwards',
                 'downwards, right to left',
                 'downwards',
                 'downwards, left to right',
                 'left to right, downwards']
    if substrings:
        for s in substrings:
            s = s.replace('in ', '').replace(' direction', '')
            if s in list_dirs:
                print(s)
                return s
    else:
        print("Substring not found.")
        return None

def caption2degree(dir):
        if dir == 'left to right':
            if random.randint(0,2) == 1:
                theta = random.uniform(0, 10)
            else:
                theta = random.uniform(350, 360)
        elif dir == 'left to right, upwards':
            theta = random.uniform(10, 45)
        elif dir == 'upwards, left to right':
            theta = random.uniform(45, 80)
        elif dir == 'upwards':
            theta = random.uniform(80, 100)
        elif dir == 'upwards, right to left':
            theta = random.uniform(100, 135)
        elif dir == 'right to left, upwards':
            theta = random.uniform(135, 170)
        elif dir == 'right to left':
            theta = random.uniform(170, 190)
        elif dir == 'right to left, downwards':
            theta = random.uniform(190, 225)
        elif dir == 'downwards, right to left':
            theta = random.uniform(225, 260)
        elif dir == 'downwards':
            theta = random.uniform(260, 280)
        elif dir == 'downwards, left to right':
            theta = random.uniform(280, 315)
        elif dir == 'left to right, downwards':
            theta = random.uniform(315, 350)
        return theta

def predict_avg_hint(caption, flow):
    substr = find_dir_from_caption(caption)
    if substr is None:
        return None
    theta = caption2degree(substr)
    theta_rad = np.pi * theta / 180.
    flow[:,0,:,:] = np.cos(theta_rad)
    flow[:,1,:,:] = -1. * np.sin(theta_rad)

    print(f'theta:{theta}, flow_x:{np.cos(theta_rad)}, flow_y:{-np.sin(theta_rad)}')

    return flow

def generate_hint(img, mask, caption):

    gt_motion = torch.zeros_like(img)[:,:2,:,:].cpu()
    gt_motion = predict_avg_hint(caption, gt_motion)

    if gt_motion is None:
        print('There is no text direction in input prompt')
        return torch.zeros_like(img)[:,:2,:,:].cpu()

    height, width = gt_motion.shape[2], gt_motion.shape[3]
    xs = torch.linspace(0, width - 1, width)
    ys = torch.linspace(0, height - 1, height)
    xs = xs.view(1, 1, width).repeat(1, height, 1)
    ys = ys.view(1, height, 1).repeat(1, 1, width)
    xys = torch.cat((xs, ys), 1).view(2, -1)  # (2,WW)

    big_motion_alpha = mask
    if int(big_motion_alpha.sum().long()) < 5:
        dense_motion = torch.zeros(gt_motion.shape)
    else:
        max_hint = 1
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

def load_model_d(config_path):
    exp_config = OmegaConf.load(config_path)
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    opt.use_prompts = False
    opt.no_instance = exp_config.optical_flow.no_instance
    opt.label_nc = exp_config.optical_flow.label_nc
    opt.input_nc = exp_config.optical_flow.input_nc
    opt.output_nc = exp_config.optical_flow.output_nc
    opt.fineSize = exp_config.optical_flow.fineSize
    opt.batchSize = exp_config.optical_flow.batchSize
    opt.netG = exp_config.optical_flow.netG
    opt.name = exp_config.optical_flow.model_name
    opt.norm = exp_config.optical_flow.norm

    model = create_model(opt)
    if 'sync' in opt.norm:
        model = convert_model(model)
    model.eval()

    return model

def predict_flow_directional(exp_config, model=None):
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    opt.use_prompts = False
    opt.no_instance = exp_config.optical_flow.no_instance
    opt.label_nc = exp_config.optical_flow.label_nc
    opt.input_nc = exp_config.optical_flow.input_nc
    opt.output_nc = exp_config.optical_flow.output_nc
    opt.use_hint = True
    opt.fineSize = exp_config.optical_flow.fineSize
    opt.batchSize = exp_config.optical_flow.batchSize
    opt.netG = exp_config.optical_flow.netG
    opt.name = exp_config.optical_flow.model_name
    opt.norm = exp_config.optical_flow.norm

    input_folder = os.path.join(exp_config.twin_extraction.exp_path_root, exp_config.twin_extraction.experiment_name)
    image_file = os.path.join(input_folder, 'translations', 'sample_0.png')
    image = load_image(image_file, target_width=opt.fineSize,
                        target_height=opt.fineSize).unsqueeze(0)

    mask_file = os.path.join(input_folder, 'mask_self_attn_erosion.png')
    mask = Image.open(mask_file).convert('L').resize((opt.fineSize, opt.fineSize), Image.NEAREST)
    mask = transforms.functional.to_tensor(mask).unsqueeze(0)
    prompt = exp_config.twin_extraction.prompt

    hint = generate_hint(image, mask, prompt)

    image = torch.cat([image, mask, hint], 1)
    data = {'label': image, 'inst': image, 'image': image, 
                    'feat': 0, 'path': [str(image_file)]}

    if model is None:
        model = create_model(opt)
        if 'sync' in opt.norm:
            model = convert_model(model)
        model.eval()

    if opt.use_prompts:
        generated = model.inference(data['label'], data['inst'], data['image'], data['prompt']) 
    else:    
        generated = model.inference(data['label'], data['inst'], data['image'])

    outSize = 512
    generated_scale = [outSize / generated.shape[3], outSize / generated.shape[2]]
    generated = generated * torch.FloatTensor(generated_scale).view(1,2,1,1).to(generated.device)
    generated = F.interpolate(generated, (outSize, outSize))

    if data['label'][0].shape[0] == 6:
        mask = (data['label'][0][3:4,:,:] == 1.) * 1.
        generated = generated.detach().cpu()
        generated_masked = (generated.clone() * mask) + (torch.zeros_like(generated) * (1. - mask))
        visuals = OrderedDict([('synthesized_flow_masked', flow2img(generated_masked.data[0])),
                               ('hint', flow2img(data['label'][0][4:,:,:]))])
    else:
        visuals = OrderedDict([('synthesized_flow_masked', flow2img(generated.data[0]))])
    img_path = data['path'][0]

    ##save flow
    save_dir = input_folder
    flow_name = '%s.pt' % ('Motion')
    save_path = os.path.join(save_dir, flow_name)
    print(save_path)
    torch.save(generated_masked.data.detach().cpu(), save_path)


    for label, image_numpy in visuals.items():
        image_name = '%s.jpg' % (label)
        save_path = os.path.join(save_dir, image_name)
        util.save_image(image_numpy, save_path)

