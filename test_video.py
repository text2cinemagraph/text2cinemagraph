import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.sync_batchnorm import convert_model
from models.models import create_model
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
import numpy as np
from torchvision import transforms
from util.util import load_image
from transformers import CLIPTokenizer
import glob
import moviepy
import moviepy.editor
import pdb

def load_mod_v(exp_config):
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    opt.no_instance = exp_config.video.no_instance
    opt.label_nc = exp_config.video.label_nc
    opt.input_nc = exp_config.video.input_nc
    opt.output_nc = exp_config.video.output_nc
    opt.fineSize = exp_config.video.fineSize
    opt.batchSize = exp_config.video.batchSize
    opt.netG = exp_config.video.netG
    opt.name = exp_config.video.model_name
    opt.speed = exp_config.video.speed
    n_frames = exp_config.video.n_frames

    model = create_model(opt)
    if 'sync' in opt.norm_G:
        model = convert_model(model)
    model.eval()

    return model

def generate_video(exp_config, model=None):
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    opt.no_instance = exp_config.video.no_instance
    opt.label_nc = exp_config.video.label_nc
    opt.input_nc = exp_config.video.input_nc
    opt.output_nc = exp_config.video.output_nc
    opt.fineSize = exp_config.video.fineSize
    opt.batchSize = exp_config.video.batchSize
    opt.netG = exp_config.video.netG
    opt.name = exp_config.video.model_name
    opt.speed = exp_config.video.speed
    n_frames = exp_config.video.n_frames


    input_folder = os.path.join(exp_config.twin_extraction.exp_path_root, exp_config.twin_extraction.experiment_name)
    image_file = os.path.join(input_folder, 'samples', '0.png')
    image = load_image(image_file, target_width=opt.fineSize,
                        target_height=opt.fineSize).unsqueeze(0)
    
    flow = torch.load(os.path.join(input_folder, 'Motion.pt')) * opt.motion_norm
    flow = torch.clamp(flow, -1, 1)
    flow = flow * opt.speed
    print("Mean flow : %.2f"%flow.abs().mean())
    flow_scale = [opt.fineSize / flow.shape[3], opt.fineSize / flow.shape[2]]
    flow = flow * torch.FloatTensor(flow_scale).view(1,2,1,1)
    flow = F.interpolate(flow, (opt.fineSize, opt.fineSize))
    flow = torch.FloatTensor(flow)

    image = torch.cat([image, image, flow], 1)
    data = {'label': image, 'inst': image, 'image': image, 
                    'feat': 0, 'path': [str(image_file)], 'index': []}

    if model is None:
        model = create_model(opt)
        if 'sync' in opt.norm_G:
            model = convert_model(model)
        model.eval()

    outVid = []
    for i in tqdm.tqdm(range(n_frames)):
        data['index'] = torch.tensor([0, i, n_frames-1]).unsqueeze(0)
        generated = model.inference(data['label'], data['inst'], data['image'], index=data['index']) 
        img_gen = util.tensor2im(generated.data[0])
        outVid.append(img_gen)
        ##save flow
    save_path = os.path.join(input_folder, 'video.mp4')
    moviepy.editor.ImageSequenceClip(sequence=[(npyFrame).clip(0.0, 255.0).round().astype(np.uint8) 
                                               for npyFrame in outVid], fps=30).write_videofile(save_path)

if __name__=='__main__':

    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    model = create_model(opt)
    if 'sync' in opt.norm_G:
        model = convert_model(model)
    model.eval()

    N = 60
    for i, data in tqdm.tqdm(enumerate(dataset)):
        if i >= opt.how_many:
            break

        minibatch = 1 
        outVid = []
        for n_f in tqdm.tqdm(range(N)):
            # pdb.set_trace()
            data['index'][1] = torch.tensor([n_f])
            generated = model.inference(data['label'], data['inst'], data['image'], index=data['index']) 
            img_gen = util.tensor2im(generated.data[0])
            outVid.append(img_gen)
            ##save flow

        img_path = data['path'][0]
        save_dir = webpage.get_image_dir()
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        save_path = os.path.join(web_dir, 'images', f'{name}_video.mp4')
        moviepy.editor.ImageSequenceClip(sequence=[(npyFrame).clip(0.0, 255.0).round().astype(np.uint8) 
                                                for npyFrame in outVid], fps=30).write_videofile(save_path)

    webpage.save()
