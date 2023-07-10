import os.path
from data.base_dataset import BaseDataset
from PIL import Image
from transformers import CLIPTokenizer

import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from util.util import hflip_flow, load_compressed_tensor, load_image, generate_mask
from pathlib import Path
import torch.nn.functional as F
# import pdb


class MotionPredictionDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot  
        self.image_files = sorted(list(Path(self.root).glob('*.png'))) + sorted(list(Path(self.root).glob('*.jpg')))
        self.randomgen = None
        self.crop_size = (opt.fineSize, opt.fineSize)
        self.rng = np.random.RandomState(0)

        self.use_mask = opt.use_mask
        self.use_hint = opt.use_hint
        self.use_prompts = opt.use_prompts

        if self.use_mask:
            self.mask_path = opt.mask_path
            
        #captions
        if self.use_prompts:
            self.captions_file = opt.captions_file
            self.img2caption = dict()
            lines = open(self.captions_file).readlines()
            for line in lines:
                img_name, caption = line.rstrip('/n').split(':')
                img_name = img_name.split('/')[-1]
                self.img2caption[img_name] = caption
            pretrained_model_path = opt.tokenizer_path
            self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        
        if self.use_hint:
            self.hint_basepath = opt.hints_path
        self.motion_norm = self.opt.motion_norm

    def __len__(self):
        return len(self.image_files) // self.opt.batchSize * self.opt.batchSize

    def augment(self, items):
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            items[0], output_size=self.crop_size)
        cropped = [TF.crop(img, i, j, h, w) for img in items]
        flip = 0
        if self.randomgen.uniform() < 0.5:
            flipped = [hflip_flow(img) for img in cropped]
            flip = 1
        else:
            flipped = cropped

        return flipped, flip

    def __getitem__(self, index):
        image_file = self.image_files[index]
        # The images may be stored as PNGs or JPGs.
        motion_file = Path(str(image_file).replace('.png', '.pth'))
        motion_file = Path(str(motion_file).replace('.jpg', '.pth'))
        motion_file = Path(str(motion_file).replace('_input', '_motion'))

        motion = load_compressed_tensor(motion_file)

        if self.use_mask:
            if self.opt.use_seg_mask:
                mask_name = os.path.basename(str(image_file)).replace('_input.jpg', '_mask.png')
                mask_file = os.path.join(self.mask_path, mask_name)
                mask = Image.open(mask_file).convert('L')
                mask = transforms.functional.to_tensor(mask).unsqueeze(0)

            else:
                mask = generate_mask(motion[0].permute(1,2,0).numpy(), method='average') * 255
                mask = Image.fromarray(mask.astype(np.uint8)).convert('L').resize((64,64))
                mask = transforms.functional.to_tensor(mask).unsqueeze(0)
                h, w = motion.shape[-2:]
                mask = F.interpolate(mask, size=(h, w), mode='nearest') 

        if self.opt.isTrain:
            image = load_image(image_file).unsqueeze(0)
        else:
            image = load_image(image_file, target_width=self.opt.fineSize,
                                target_height=self.opt.fineSize).unsqueeze(0)

        if self.use_hint:
            id = os.path.basename(str(image_file)).replace('_input.jpg', '')
            idx_hint = self.rng.randint(1,6)
            hint_path = os.path.join(self.hint_basepath, id, f'{idx_hint}_blend.pth')
            hint = torch.load(hint_path).unsqueeze(0)
            hint = hint.to(image.dtype)

        if not self.opt.no_tanh:
            motion = motion / self.motion_norm
            if self.use_hint:
                hint = hint / self.motion_norm

        if self.opt.isTrain:
            if self.use_hint:
                cropped, flip = self.augment([image, motion, mask, hint])
                image, motion, mask, hint = cropped
            else:
                cropped, flip = self.augment([image, motion, mask])
                image, motion, mask = cropped
        else:
            motion_scale = [self.opt.fineSize/motion.shape[3], self.opt.fineSize/motion.shape[2]]
            motion = F.interpolate(motion, size=(self.opt.fineSize, self.opt.fineSize), mode='bilinear',align_corners=False)
            motion = motion * torch.FloatTensor(motion_scale).view(1,2,1,1)
            if self.use_hint:
                hint = F.interpolate(hint, size=(self.opt.fineSize, self.opt.fineSize), mode='bilinear',align_corners=False)
                hint = hint * torch.FloatTensor(motion_scale).view(1,2,1,1)
            if self.use_mask:    
                mask = F.interpolate(mask, size=(self.opt.fineSize, self.opt.fineSize), mode='nearest')

        assert(image.shape[-2:] == motion.shape[-2:])
        assert(image.dim() == motion.dim())

        ###captions
        if self.use_prompts:
            img_name = str(image_file).split('/')[-1].replace('_input.jpg','')
            prompt = self.img2caption[img_name]
            prompt_ids = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids[0]

        #concatenate mask and rgb image
        if self.use_mask and self.use_hint:
            image = torch.cat([image, mask, hint], 1)
        elif self.use_mask:
            image = torch.cat([image, mask], 1)
        elif self.use_hint:
            image = torch.cat([image, hint], 1)
        
        input_dict = {'label': image[0], 'inst': 0, 'image': motion[0], 
                      'feat': 0, 'path': [str(image_file)],}
        if self.use_prompts:
            input_dict['prompt'] = prompt_ids
            input_dict['caption'] = [str(prompt)]
        if self.use_hint:
            input_dict['hint'] = hint[0]

        return input_dict

    # So that each worker gets a unique random seed, so different workers aren't picking the same random values.
    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        self.randomgen = np.random.RandomState(seed=worker_info.seed % (2 ** 31 - 1))
    
    def name(self):
        return 'MotionPredictionDataset'


class FramePredictionDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot  
        self.image_files = sorted(list(Path(self.root).glob('*.png'))) + sorted(list(Path(self.root).glob('*.jpg')))
        self.randomgen = None
        self.crop_size = (opt.fineSize, opt.fineSize)
        self.rng = np.random.RandomState(0)
        self.frames_basepath = opt.frames_basepath
        self.motion_path = opt.motion_basepath
        # self.motion_norm = self.opt.motion_norm

    def __len__(self):
        return len(self.image_files) // self.opt.batchSize * self.opt.batchSize

    def augment(self, items):
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            items[0], output_size=self.crop_size)
        cropped = [TF.crop(img, i, j, h, w) for img in items]
        flip = 0
        if self.randomgen.uniform() < 0.5:
            flipped = [hflip_flow(img) for img in cropped]
            flip = 1
        else:
            flipped = cropped

        return flipped, flip

    def __getitem__(self, index):
        image_file = self.image_files[index]
        # The images may be stored as PNGs or JPGs.
        N = 60
        if self.opt.isTrain:
            if self.opt.tr_stage == 'stage1':
                motion_file = Path(str(image_file).replace('.png', '.pth'))
                motion_file = Path(str(motion_file).replace('.jpg', '.pth'))
                motion_file = Path(str(motion_file).replace('_input', '_motion'))
                motion = load_compressed_tensor(motion_file)
            else:
                img_name = str(image_file).split('/')[-1].replace('_input.jpg','')
                motion_file = os.path.join(self.motion_path, f'{img_name}_Motion.pt')
                motion = torch.load(motion_file)
                motion = motion * self.opt.motion_norm

            start_index = self.rng.randint(0,N//3)
            end_index = self.rng.randint(N//3*2,N)
            middle_index = self.rng.randint(start_index, end_index)

            img_name = str(image_file).split('/')[-1].replace('_input.jpg','')
            start_img_path = os.path.join(self.frames_basepath, img_name + "_gt", "images", f'{start_index}.png')
            start_image = load_image(start_img_path).unsqueeze(0)
            middle_img_path = os.path.join(self.frames_basepath, img_name + "_gt", "images", f'{middle_index}.png')
            middle_image = load_image(middle_img_path).unsqueeze(0)
            end_img_path = os.path.join(self.frames_basepath, img_name + "_gt", "images", f'{end_index}.png')
            end_image = load_image(end_img_path).unsqueeze(0)

        else:
            start_index = middle_index = 0
            end_index = N-1
            start_image = end_image = middle_image = load_image(image_file, target_width=self.opt.fineSize,
                                                                 target_height=self.opt.fineSize).unsqueeze(0)
            img_name = str(image_file).split('/')[-1].replace('_input.jpg','')
            motion_file = os.path.join(self.motion_path, f'{img_name}_Motion.pt')
            motion = torch.load(motion_file)
            motion = motion * self.opt.motion_norm
            motion = torch.clamp(motion, -1, 1)
            motion = motion * self.opt.speed

        if self.opt.isTrain:
            cropped, flip = self.augment([start_image, middle_image, end_image, motion])
            start_image, middle_image, end_image, motion = cropped
        else:
            motion_scale = [self.opt.fineSize/motion.shape[3], self.opt.fineSize/motion.shape[2]]
            motion = F.interpolate(motion, size=(self.opt.fineSize, self.opt.fineSize), mode='bilinear',align_corners=False)
            motion = motion * torch.FloatTensor(motion_scale).view(1,2,1,1)

        assert(start_image.shape[-2:] == motion.shape[-2:])
        assert(start_image.dim() == motion.dim())

        #concatenate mask and rgb image
        image = torch.cat([start_image, end_image, motion], 1)
        
        input_dict = {'label': image[0], 'inst': 0, 'image': middle_image[0], 
                      'feat': 0, 'path': [str(image_file)], 'index': [start_index, middle_index,end_index]}

        return input_dict

    # So that each worker gets a unique random seed, so different workers aren't picking the same random values.
    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        self.randomgen = np.random.RandomState(seed=worker_info.seed % (2 ** 31 - 1))
    
    def name(self):
        return 'FramePredictionDataset'