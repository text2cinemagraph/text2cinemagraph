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
from omegaconf import OmegaConf

def load_model_ud(config_path):
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

    opt.use_prompts = True
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

def predict_flow(exp_config, model=None):
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    opt.use_prompts = True
    opt.no_instance = exp_config.optical_flow.no_instance
    opt.label_nc = exp_config.optical_flow.label_nc
    opt.input_nc = exp_config.optical_flow.input_nc
    opt.output_nc = exp_config.optical_flow.output_nc
    opt.fineSize = exp_config.optical_flow.fineSize
    opt.batchSize = exp_config.optical_flow.batchSize
    opt.netG = exp_config.optical_flow.netG
    opt.name = exp_config.optical_flow.model_name
    opt.norm = exp_config.optical_flow.norm

    if model is None:
        model = create_model(opt)
        if 'sync' in opt.norm:
            model = convert_model(model)
        model.eval()

    input_folder = os.path.join(exp_config.twin_extraction.exp_path_root, exp_config.twin_extraction.experiment_name)
    image_file = os.path.join(input_folder, 'translations', 'sample_0.png')
    image = load_image(image_file, target_width=opt.fineSize,
                        target_height=opt.fineSize).unsqueeze(0)

    mask_file = os.path.join(input_folder, 'mask_self_attn_erosion.png')
    mask = Image.open(mask_file).convert('L').resize((opt.fineSize, opt.fineSize), Image.NEAREST)
    mask = transforms.functional.to_tensor(mask).unsqueeze(0)
    prompt = exp_config.twin_extraction.prompt
    pretrained_model_path = exp_config.optical_flow.tokenizer_path
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    prompt_ids = tokenizer(
        prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]
    image = torch.cat([image, mask], 1)
    data = {'label': image, 'inst': image, 'image': image, 
                    'feat': 0, 'path': [str(image_file)], 'prompt': prompt_ids.unsqueeze(0), 'caption': [str(prompt)]}

    if opt.use_prompts:
        generated = model.inference(data['label'], data['inst'], data['image'], data['prompt']) 
    else:    
        generated = model.inference(data['label'], data['inst'], data['image'])

    outSize = 512
    generated_scale = [outSize / generated.shape[3], outSize / generated.shape[2]]
    generated = generated * torch.FloatTensor(generated_scale).view(1,2,1,1).to(generated.device)
    generated = F.interpolate(generated, (outSize, outSize))

    if data['label'][0].shape[0] == 4:
        mask = (data['label'][0][3:,:,:] == 1.) * 1.
        generated = generated.detach().cpu()
        generated_masked = (generated.clone() * mask) + (torch.zeros_like(generated) * (1. - mask))
        visuals = OrderedDict([('synthesized_flow_masked', flow2img(generated_masked.data[0]))])
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
    if 'sync' in opt.norm:
        model = convert_model(model)
    model.eval()

    for i, data in tqdm.tqdm(enumerate(dataset)):
        if i >= opt.how_many:
            break

        minibatch = 1 
        if opt.use_prompts:
            generated = model.inference(data['label'], data['inst'], data['image'], data['prompt']) 
        else:    
            generated = model.inference(data['label'], data['inst'], data['image'])

        if data['label'][0].shape[0] == 4:
            mask = (data['label'][0][3:4,:,:] == 1.) * 1.
            generated = generated.detach().cpu()
            generated_masked = (generated.clone() * mask) + (torch.zeros_like(generated) * (1. - mask))
            visuals = OrderedDict([('input_image', util.tensor2im(data['label'][0][:3,:,:])),
                            ('mask', util.tensor2im(mask, normalize=False)),
                            ('synthesized_flow_masked', flow2img(generated_masked.data[0])),
                            ('synthesized_flow', flow2img(generated.data[0])),
                            ('real_flow', flow2img(data['image'][0]))])
        else:
            visuals = OrderedDict([('input_image', util.tensor2im(data['label'][0])),
                                ('synthesized_flow', flow2img(generated.data[0])),
                                ('real_flow', flow2img(data['image'][0]))])
        img_path = data['path'][0]

        ##save flow
        save_dir = webpage.get_image_dir()
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        flow_name = '%s_%s.pt' % (name.replace('_input',''), 'Motion')
        save_path = os.path.join(save_dir, flow_name)
        torch.save(generated.data.detach().cpu(), save_path)
        
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()
