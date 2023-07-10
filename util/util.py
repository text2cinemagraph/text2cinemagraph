from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os
import lz4framed
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pdb

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def load_compressed_tensor(filename):
    retval = None
    with open(filename, mode='rb') as file:
        retval = torch.from_numpy(pickle.loads(lz4framed.decompress(file.read())))
    return retval

def load_image(filepath, target_width=None, target_height=None):
    """
    Loads an image.
    Optionally, scales the image proportionally so the width matches target_width.

    :param filepath: The path to the image file.
    :param target_width: The target width of the output tensor.
    :return: The loaded image tensor
    """
    im = Image.open(filepath)
    oh, ow = np.array(im).shape[:2]
    if target_width is not None and ow != target_width:
        w = target_width
        if target_height is not None:
            h = target_height
        else:
            h = int(target_width * oh / ow)
        im = im.resize((h, w), Image.BICUBIC)

    im = (torchvision.transforms.functional.to_tensor(im) - 0.5) * 2.
    # oh, ow = im.shape[-2:]
    # if target_width is None or ow == target_width:
    #     return im
    # w = target_width
    # h = int(target_width * oh / ow)
    # if target_width > im.shape[-1]:
    #     return nn.functional.interpolate(im, size=(h, w), mode='bicubic')
    # else:
    #     return nn.functional.interpolate(im, size=(h, w), mode='area')
    return im

def load_mask(filepath, target_width=None, image_shape=None):
    """
    Loads an image.
    Optionally, scales the image proportionally so the width matches target_width.

    :param filepath: The path to the image file.
    :param target_width: The target width of the output tensor.
    :return: The loaded image tensor
    """
    im = Image.open(filepath).convert('L')
    if image_shape is not None:
        h, w = image_shape
        im = im.resize((w, h), Image.NEAREST)
    im = torchvision.transforms.functional.to_tensor(im)
    oh, ow = im.shape[-2:]
    if target_width is None or ow == target_width:
        return im
    w = target_width
    h = int(target_width * oh / ow)
    if target_width > im.shape[-1]:
        return nn.functional.interpolate(im, size=(h, w), mode='nearest')
    else:
        return nn.functional.interpolate(im, size=(h, w), mode='nearest')
    
def scale_flow(flow, shape):
    """
    Resize flow field, and scale the flow values accordingly

    Expects Bx2xHxW as input. Returns Bx2xHxW.
    """
    assert (flow.dim() == 4)
    original_height, original_width = flow.shape[-2:]
    height, width = shape
    flow = F.interpolate(
        input=flow.clone(),
        size=(height, width), mode='bilinear', align_corners=False)
    flow[:, 0, :, :] *= float(width) / float(original_width)
    flow[:, 1, :, :] *= float(height) / float(original_height)
    return flow

def hflip_flow(flow):
    """
    Horizontal flipping, but if the image is 2-channeled, it is assumed to be an optical flow field, and the X component
     of the flow field will also be reversed. This is to ensure that the flow remains consistent with the scene, for
     instance if the input images and flow field are all horizontally flipped.

    Expects Bx2xHxW as input. Returns Bx2xHxW.
    """
    assert (flow.dim() == 4)
    flow_flipped = flow.flip(dims=[-1])
    if flow.shape[1] == 2:
        flow_flipped[:, 0] *= -1
    return flow_flipped

def generate_mask(flow, method, ratio=10.0):
    mask = np.zeros((flow.shape[0], flow.shape[1]))
    if method == 'zeros':
        mask = ((flow[:,:,0]!=0.0) | (flow[:,:,1]!=0.0))*1.0
    elif method == 'average':
        mean_flow = np.sum(np.square(flow))/(flow.shape[0]*flow.shape[1])
        mask = (np.sum(np.square(flow), axis=2)>mean_flow/ratio)*1.0
    return mask

# def euler_integration(motion, destination_frame, return_all_frames=False):
#     """
#     This function is provided by Aleksander Hołyński <holynski@cs.washington.edu>
#     Repeatedly integrates the Eulerian motion field to get the displacement map to a future frame.

#     :param motion: The Eulerian motion field to be integrated.
#     :param destination_frame: The number of times the motion field should be integrated.
#     :param return_all_frames: Return the displacement maps to all intermediate frames, not only the last frame.
#     :return: The displacement map resulting from repeated integration of the motion field.
#     """

#     assert (motion.dim() == 4)
#     b, c, height, width = motion.shape
#     assert (b == 1), 'Function only implemented for batch = 1'
#     assert (c == 2), f'Input motion field should be Bx2xHxW. Given tensor is: {motion.shape}'

#     y, x = torch.meshgrid(
#         [torch.linspace(0, height - 1, height, device='cuda'),
#          torch.linspace(0, width - 1, width, device='cuda')])
#     coord = torch.stack([x, y], dim=0).long()

#     destination_coords = coord.clone().float()
#     if return_all_frames:
#         displacements = torch.zeros(destination_frame + 1, 2, height, width, device='cuda')
#         visible_pixels = torch.ones(b + 1, 1, height, width, device='cuda')
#     else:
#         displacements = torch.zeros(1, 2, height, width, device='cuda')
#         visible_pixels = torch.ones(1, 1, height, width, device='cuda')
#     invalid_mask = torch.zeros(1, height, width, device='cuda').bool()
#     for frame_id in range(1, destination_frame + 1):
#         destination_coords = destination_coords + motion[0][:, torch.round(destination_coords[1]).long(),
#                                                   torch.round(destination_coords[0]).long()]
#         out_of_bounds_x = torch.logical_or(destination_coords[0] > (width - 1), destination_coords[0] < 0)
#         out_of_bounds_y = torch.logical_or(destination_coords[1] > (height - 1), destination_coords[1] < 0)
#         invalid_mask = torch.logical_or(out_of_bounds_x.unsqueeze(0), invalid_mask)
#         invalid_mask = torch.logical_or(out_of_bounds_y.unsqueeze(0), invalid_mask)

#         # Set the displacement of invalid pixels to zero, to avoid out-of-bounds access errors
#         destination_coords[invalid_mask.expand_as(destination_coords)] = coord[
#             invalid_mask.expand_as(destination_coords)].float()
#         if return_all_frames:
#             displacements[frame_id] = (destination_coords - coord.float()).unsqueeze(0)
#             # Set the displacements for invalid pixels to be out of bounds.
#             displacements[frame_id][invalid_mask] = torch.max(height, width) + 1
#             visible_pixels[frame_id] = torch.logical_not(invalid_mask.clone()).float().unsqueeze(0)
#         else:
#             displacements = (destination_coords - coord.float()).unsqueeze(0)
#             visible_pixels = torch.logical_not(invalid_mask.clone()).float().unsqueeze(0)
#             displacements[invalid_mask.unsqueeze(0).repeat(1,2,1,1)] = torch.max(torch.Tensor([height, width])) + 1
#     return displacements,visible_pixels

# class EulerIntegration(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, motion, destination_frame, return_all_frames=False,show_visible_pixels=False):
#         displacements = torch.zeros(motion.shape).to(motion.device)
#         visible_pixels = torch.zeros(motion.shape[0], 1, motion.shape[2], motion.shape[3])
#         for b in range(motion.shape[0]):
#             displacements[b:b+1], visible_pixels[b:b+1] = euler_integration(motion[b:b+1], destination_frame[b])

#         if show_visible_pixels:
#             return displacements, visible_pixels
#         else:
#             return displacements

def euler_integration(motion, destination_frame, return_all_frames=False):
    """
    This function is provided by Aleksander Hołyński <holynski@cs.washington.edu>
    Repeatedly integrates the Eulerian motion field to get the displacement map to a future frame.

    :param motion: The Eulerian motion field to be integrated.
    :param destination_frame: The number of times the motion field should be integrated.
    :param return_all_frames: Return the displacement maps to all intermediate frames, not only the last frame.
    :return: The displacement map resulting from repeated integration of the motion field.
    """

    assert (motion.dim() == 4)
    b, c, height, width = motion.shape
    assert (b == 1), 'Function only implemented for batch = 1'
    assert (c == 2), f'Input motion field should be Bx2xHxW. Given tensor is: {motion.shape}'

    y, x = torch.meshgrid(
        [torch.linspace(0, height - 1, height, device='cuda'),
         torch.linspace(0, width - 1, width, device='cuda')])
    coord = torch.stack([x, y], dim=0).long()

    destination_coords = coord.clone().float()
    if return_all_frames:
        displacements = torch.zeros(destination_frame + 1, 2, height, width, device='cuda')
        visible_pixels = torch.ones(b + 1, 1, height, width, device='cuda')
    else:
        displacements = torch.zeros(1, 2, height, width, device='cuda')
        visible_pixels = torch.ones(1, 1, height, width, device='cuda')
    invalid_mask = torch.zeros(1, height, width, device='cuda').bool()
    for frame_id in range(1, destination_frame + 1):
        destination_coords = destination_coords + motion[0][:, torch.round(destination_coords[1]).long(),
                                                  torch.round(destination_coords[0]).long()]
        out_of_bounds_x = torch.logical_or(destination_coords[0] > (width - 1), destination_coords[0] < 0)
        out_of_bounds_y = torch.logical_or(destination_coords[1] > (height - 1), destination_coords[1] < 0)
        invalid_mask = torch.logical_or(out_of_bounds_x.unsqueeze(0), invalid_mask)
        invalid_mask = torch.logical_or(out_of_bounds_y.unsqueeze(0), invalid_mask)

        # Set the displacement of invalid pixels to zero, to avoid out-of-bounds access errors
        destination_coords[invalid_mask.expand_as(destination_coords)] = coord[
            invalid_mask.expand_as(destination_coords)].float()
        if return_all_frames:
            displacements[frame_id] = (destination_coords - coord.float()).unsqueeze(0)
            # Set the displacements for invalid pixels to be out of bounds.
            displacements[frame_id][invalid_mask] = torch.max(height, width) + 1
            visible_pixels[frame_id] = torch.logical_not(invalid_mask.clone()).float().unsqueeze(0)
        else:
            displacements = (destination_coords - coord.float()).unsqueeze(0)
            visible_pixels = torch.logical_not(invalid_mask.clone()).float().unsqueeze(0)
            displacements[invalid_mask.unsqueeze(0).repeat(1,2,1,1)] = torch.max(torch.Tensor([height, width])) + 1
    return displacements,visible_pixels

class EulerIntegration(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
    def forward(self, motion, destination_frame, return_all_frames=False,show_visible_pixels=False):
        displacements = torch.zeros(motion.shape).to(motion.device)
        visible_pixels = torch.zeros(motion.shape[0], 1, motion.shape[2], motion.shape[3])
        for b in range(motion.shape[0]):
            displacements[b:b+1], visible_pixels[b:b+1] = euler_integration(motion[b:b+1], destination_frame[b])

        if show_visible_pixels:
            return displacements, visible_pixels
        else:
            return displacements