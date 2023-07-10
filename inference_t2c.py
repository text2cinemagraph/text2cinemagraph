import argparse, os, sys
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import tqdm
import torchvision
import torch.nn.functional as F
import json
import nltk
import re

from options.test_options import TestOptions

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
            s_ = s.replace('in ', '').replace(' direction', '')
            if s_ in list_dirs:
                print(s)
                return s
    else:
        print("Substring not found.")

def main():
    opt = TestOptions().parse(save=False)
    exp_config = OmegaConf.load(f"{opt.config}")

    if opt.stage == 'all': #run all steps
        from img2img.run_features_extraction import twin_extraction
        from img2img.run_pnp import twin_generation
        from ODISE.demo.demo import odise_mask
        from img2img.pnp_utils import generate_attn_mask
        from test_motion import predict_flow
        from test_motion_directional import predict_flow_directional
        from test_video import generate_video
        # s1
        if opt.use_hint:
            substr = find_dir_from_caption(exp_config.twin_extraction.prompt)
            prompt_o = exp_config.twin_extraction.prompt
            exp_config.twin_extraction.prompt = exp_config.twin_extraction.prompt.replace(f'{substr} ', '')
            print(exp_config.twin_extraction.prompt)
        twin_extraction(exp_config)
        # s2
        exp_config.twin_generation.ckpt = exp_config.twin_extraction.ckpt
        exp_config.twin_generation.experiment_name = exp_config.twin_extraction.experiment_name
        exp_config.twin_generation.exp_path_root = exp_config.twin_extraction.exp_path_root
        if exp_config.twin_extraction.init_img is not None:
            exp_config.twin_generation.init_img = exp_config.twin_extraction.init_img
        if exp_config.twin_generation.prompt is None:
            with open('./dataset/nouns.json', 'r') as f:
                nouns_c = json.load(f)
            sentences = nltk.sent_tokenize(exp_config.twin_extraction.prompt)
            nouns = []
            for sentence in sentences:
                for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
                    if (pos == 'NN' or pos == 'NNS') and word in nouns_c:
                        nouns.append(word)
            exp_config.twin_generation.prompt = ' '.join(nouns) + ", nature, bright, realistic, photography, 4k, 8k"
        twin_generation(exp_config)
        #s3
        exp_config.odise.input = [os.path.join(exp_config.twin_extraction.exp_path_root, exp_config.twin_extraction.experiment_name, 'translations/sample_0.png')]
        exp_config.odise.output = os.path.join(exp_config.twin_extraction.exp_path_root, exp_config.twin_extraction.experiment_name, 'mask_odise.png')
        exp_config.odise.label = []
        odise_mask(exp_config)
        # s4
        generate_attn_mask(exp_config)
        #s5
        exp_config.twin_extraction.prompt = prompt_o
        if not opt.use_hint:
            predict_flow(exp_config)
        else:
            predict_flow_directional(exp_config)
        #s6
        generate_video(exp_config)

    elif opt.stage == 's1': #twin image extraction
        from img2img.run_features_extraction import twin_extraction

        if opt.use_hint:
            substr = find_dir_from_caption(exp_config.twin_extraction.prompt)
            exp_config.twin_extraction.prompt = exp_config.twin_extraction.prompt.replace(f'{substr} ', '')
            print(exp_config.twin_extraction.prompt)
        twin_extraction(exp_config)

    elif opt.stage == 's2': #twin image generation
        from img2img.run_pnp import twin_generation

        exp_config.twin_generation.ckpt = exp_config.twin_extraction.ckpt
        
        exp_config.twin_generation.experiment_name = exp_config.twin_extraction.experiment_name
        exp_config.twin_generation.exp_path_root = exp_config.twin_extraction.exp_path_root
        if exp_config.twin_extraction.init_img is not None:
            exp_config.twin_generation.init_img = exp_config.twin_extraction.init_img
        
        if exp_config.twin_generation.prompt is None:
            with open('./dataset/nouns.json', 'r') as f:
                nouns_c = json.load(f)
            sentences = nltk.sent_tokenize(exp_config.twin_extraction.prompt)
            nouns = []
            for sentence in sentences:
                for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
                    if (pos == 'NN' or pos == 'NNS') and word in nouns_c:
                        nouns.append(word)
            exp_config.twin_generation.prompt = ' '.join(nouns) + ", nature, bright, realistic, photography, 4k, 8k"
        twin_generation(exp_config)
    
    elif opt.stage == 's3': #odise mask generation
        from ODISE.demo.demo import odise_mask
        
        exp_config.odise.input = [os.path.join(exp_config.twin_extraction.exp_path_root, exp_config.twin_extraction.experiment_name, 'translations/sample_0.png')]
        exp_config.odise.output = os.path.join(exp_config.twin_extraction.exp_path_root, exp_config.twin_extraction.experiment_name, 'mask_odise.png')
        exp_config.odise.label = []
        odise_mask(exp_config)

    elif opt.stage == 's4': #self-attention mask generation
        from img2img.pnp_utils import generate_attn_mask

        generate_attn_mask(exp_config)

    elif opt.stage == 's5': #flow prediction
        from test_motion import predict_flow
        from test_motion_directional import predict_flow_directional

        if not opt.use_hint:
            predict_flow(exp_config)
        else:
            predict_flow_directional(exp_config)

    elif opt.stage == 's6': #video generation
        from test_video import generate_video

        generate_video(exp_config)

if __name__ == "__main__":
    main()