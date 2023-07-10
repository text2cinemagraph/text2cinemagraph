import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .normalization import LinearNoiseLayer, PartialLinearNoiseLayer
from . import attention
from .configs import get_resnet_arch
from util.util import EulerIntegration
from . import softsplat
from .partialconv2d import PartialConv2d
import pdb

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def modify_sd2pix2pixHD_keys(sd):

    keys = list(sd.keys())
    sd2pix2pixHD_w = {}
    sd2pix2pixHD_w['model.diffusion_model.input_blocks.4.1'] = 'attn_input_block_1'
    sd2pix2pixHD_w['model.diffusion_model.input_blocks.5.1'] = 'attn_input_block_2'
    sd2pix2pixHD_w['model.diffusion_model.input_blocks.7.1'] = 'attn_input_block_3'
    sd2pix2pixHD_w['model.diffusion_model.input_blocks.8.1'] = 'attn_input_block_4'
    sd2pix2pixHD_w['model.diffusion_model.middle_block.1'] = 'attn_middle_block'
    sd2pix2pixHD_w['model.diffusion_model.output_blocks.4.1'] = 'attn_output_block_4'
    sd2pix2pixHD_w['model.diffusion_model.output_blocks.5.1'] = 'attn_output_block_3'
    sd2pix2pixHD_w['model.diffusion_model.output_blocks.6.1'] = 'attn_output_block_2'
    sd2pix2pixHD_w['model.diffusion_model.output_blocks.7.1'] = 'attn_output_block_1'

    final_keys = []
    final_keys_ori = []
    for key in keys:
        if not ('norm1' in key or 'attn1' in key):
            for key_ in sd2pix2pixHD_w.keys():
                if key_ in key:
                    final_keys.append(key.replace(key_, sd2pix2pixHD_w[key_]))
                    final_keys_ori.append(key)
                    break
    ignore_keys = [key for key in keys if key not in final_keys_ori]
    for k in keys:
        for ik in ignore_keys:
            if k.startswith(ik):
                try:
                    del sd[k]
                except:
                    print("key {} not in state_dict.".format(k))
    sd_new = {}
    for key, key_ in zip(final_keys_ori, final_keys):
        sd_new[key_] = sd[key]
    return sd_new

def define_G(input_nc, output_nc, ngf, netG_type, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[], no_tanh=True, opt=None):    

    if netG_type == 'spadexattnunetsd':
        netG = SPADEXAttnUnet4MotionSD(input_nc, output_nc, norm=norm, no_tanh=no_tanh)
    elif netG_type == 'spadeunet':
        netG = SPADEUnet4MaskMotion(input_nc, output_nc, no_tanh=no_tanh)
    elif netG_type == 'spadeunet4softmaxsplating':
        netG = SPADEUnet4SoftmaxSplating(opt, model_type='resnet_256W8UpDown64_de_resnet_pconv2_nonorm')
    else:
        raise('generator not implemented!')

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)

    if netG_type == 'spadexattnunetsd':
        sd_path = './img2img/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt'
        sd = torch.load(sd_path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        sd_ = modify_sd2pix2pixHD_keys(sd)
        _, _ = netG.load_state_dict(sd_, strict=False)
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())   
            netG.cuda(gpu_ids[0])

        #freeze k and q matrices
        for x in netG.named_parameters():
            if 'attn2.to_k' in x[0] or 'attn2.to_q' in x[0]:
                x[1].requires_grad = False

    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class EnePointErrorWrapper(nn.Module):
    def forward(self, pred_motion, gt_motion):
        if pred_motion.shape[1] == 3:
            new_pred_motion = pred_motion[:,:2,:,:] * pred_motion[:,2:3,:,:]
        else:
            new_pred_motion = pred_motion
        if gt_motion.shape[1] == 3:
            new_gt_motion = gt_motion[:,:2,:,:] * gt_motion[:,2:3,:,:]
        else:
            new_gt_motion = gt_motion

        err = torch.norm(new_pred_motion - new_gt_motion, 2, 1)
        err = err.mean()
        return err

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

##############################################################################
# Generator
##############################################################################

class SPADE(nn.Module):
    def __init__(self, norm_layer, norm_nc, label_nc):
        super().__init__()
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        ks = 3
        nhidden = 128
        pw = ks // 2
        self.param_free_norm = norm_layer(norm_nc)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        if segmap.shape[1] == 1: # Mask
            segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest',align_corners=False)
        elif segmap.shape[1] == 3: # RGB
            segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear',align_corners=False)
        elif segmap.shape[1] == 4: #RGB,Mask
            segmap = torch.cat([
                F.interpolate(segmap[:,:3,...], size=x.size()[2:], mode='bilinear',align_corners=False),
                F.interpolate(segmap[:,3:4,...], size=x.size()[2:], mode='nearest'),
            ],1)
        elif segmap.shape[1] == 6: #RGB,Mask,Flow
            segmap = torch.cat([
                F.interpolate(segmap[:,:3,...], size=x.size()[2:], mode='bilinear',align_corners=False),
                F.interpolate(segmap[:,3:4,...], size=x.size()[2:], mode='nearest'),
                F.interpolate(segmap[:,4:6,...], size=x.size()[2:], mode='bilinear',align_corners=False),
            ],1)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
    
def get_conv_layer_v1(norm, use_3D=False):
    if "spectral" in norm:
        if use_3D:
            conv_layer_base = lambda in_c, out_c, k, s, p: nn.utils.spectral_norm(
                nn.Conv3d(in_c, out_c, kernel_size=k, padding=p, stride=s)
            )
        else:
            conv_layer_base = lambda in_c, out_c, k, s, p: nn.utils.spectral_norm(
                nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s)
            )
    else:
        if use_3D:
            conv_layer_base = lambda in_c, out_c, k, s, p: nn.Conv3d(
                in_c, out_c, kernel_size=k, padding=p, stride=s
            )
        else:
            conv_layer_base = lambda in_c, out_c, k, s, p: nn.Conv2d(
                in_c, out_c, kernel_size=k, padding=p, stride=s
            )

    return conv_layer_base


def spectral_conv_function(in_c, out_c, k, p, s, dilation=1, bias=True):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s, dilation=dilation, bias=bias)
    )

def conv_function(in_c, out_c, k, p, s, dilation=1 ,bias=True):
    return nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s, dilation=dilation, bias=bias)

def get_conv_layer(opt):
    if "spectral" in opt.norm_G:
        conv_layer_base = spectral_conv_function
    else:
        conv_layer_base = conv_function

    return conv_layer_base

def spectral_pconv_function(in_c, out_c, k, p, s, dilation=1):
    return nn.utils.spectral_norm(
        PartialConv2d(in_c, out_c, kernel_size=k, padding=p, stride=s, dilation=dilation,
                  bias=True, multi_channel=True, return_mask=True)
    )


def pconv_function(in_c, out_c, k, p, s, dilation=1):
    return PartialConv2d(in_c, out_c, kernel_size=k, padding=p, stride=s, dilation=dilation,
                  bias=True, multi_channel=True, return_mask=True)

def get_pconv_layer(opt):
    if "spectral" in opt.norm_G:
        conv_layer_base = spectral_pconv_function
    else:
        conv_layer_base = pconv_function
    return conv_layer_base


class SPADEUnet4MaskMotion(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(
        self,
        channels_in=3,
        channels_out=2,
        num_filters=32,
        use_3D=False,
        norm='spectral_instance',
        up_mode="bilinear",
        no_tanh=True,
    ):
        super(SPADEUnet4MaskMotion, self).__init__()

        conv_layer = get_conv_layer_v1(norm, use_3D=use_3D)

        self.conv1 = conv_layer(channels_in, num_filters, 4, 2, 1)
        self.conv2 = conv_layer(num_filters, num_filters * 2, 4, 2, 1)
        self.conv3 = conv_layer(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = conv_layer(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.conv5 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv6 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv7 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)

        #self.up = nn.Upsample(scale_factor=2, mode="nearest")
        #self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_nearest = nn.Upsample(scale_factor=2, mode="nearest")
        self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False)

        self.dconv1 = conv_layer(num_filters * 8, num_filters * 8, 3, 1, 1)
        self.dconv2 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv3 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv4 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv5 = conv_layer(num_filters * 8 * 2, num_filters * 4, 3, 1, 1)
        self.dconv6 = conv_layer(num_filters * 4 * 2, num_filters * 2, 3, 1, 1)
        self.dconv7 = conv_layer(num_filters * 2 * 2, num_filters, 3, 1, 1)
        self.dconv8 = conv_layer(num_filters * 2, channels_out, 3, 1, 1)

        if norm == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm == "spectral_instance":
            norm_layer = nn.InstanceNorm2d
        elif norm == "spectral_batch":
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = nn.InstanceNorm2d

        self.spade_layer = SPADE(norm_layer, num_filters, channels_in)#self.batch_norm = norm_layer(num_filters)
        self.spade_layer2_0 = SPADE(norm_layer, num_filters*2, channels_in)#self.batch_norm2_0 = norm_layer(num_filters * 2)
        self.spade_layer2_1 = SPADE(norm_layer, num_filters*2, channels_in)#self.batch_norm2_1 = norm_layer(num_filters * 2)
        self.spade_layer4_0 = SPADE(norm_layer, num_filters*4, channels_in)#self.batch_norm4_0 = norm_layer(num_filters * 4)
        self.spade_layer4_1 = SPADE(norm_layer, num_filters*4, channels_in)#self.batch_norm4_1 = norm_layer(num_filters * 4)
        self.spade_layer8_0 = SPADE(norm_layer, num_filters*8, channels_in)#self.batch_norm8_0 = norm_layer(num_filters * 8)
        self.spade_layer8_1 = SPADE(norm_layer, num_filters*8, channels_in)#self.batch_norm8_1 = norm_layer(num_filters * 8)
        self.spade_layer8_2 = SPADE(norm_layer, num_filters*8, channels_in)#self.batch_norm8_2 = norm_layer(num_filters * 8)
        self.spade_layer8_3 = SPADE(norm_layer, num_filters*8, channels_in)#self.batch_norm8_3 = norm_layer(num_filters * 8)
        self.spade_layer8_4 = SPADE(norm_layer, num_filters*8, channels_in)#self.batch_norm8_4 = norm_layer(num_filters * 8)
        self.spade_layer8_5 = SPADE(norm_layer, num_filters*8, channels_in)#self.batch_norm8_5 = norm_layer(num_filters * 8)
        self.spade_layer8_6 = SPADE(norm_layer, num_filters*8, channels_in)#self.batch_norm8_6 = norm_layer(num_filters * 8)
        self.spade_layer8_7 = SPADE(norm_layer, num_filters*8, channels_in)#self.batch_norm8_7 = norm_layer(num_filters * 8)


        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.no_tanh = no_tanh

    def forward(self, input):
        # Encoder
        e1 = self.conv1(input)
        e2 = self.spade_layer2_0(self.conv2(self.leaky_relu(e1)), input)
        e3 = self.spade_layer4_0(self.conv3(self.leaky_relu(e2)), input)
        e4 = self.spade_layer8_0(self.conv4(self.leaky_relu(e3)), input)
        e5 = self.spade_layer8_1(self.conv5(self.leaky_relu(e4)), input)
        e6 = self.spade_layer8_2(self.conv6(self.leaky_relu(e5)), input)
        e7 = self.spade_layer8_3(self.conv7(self.leaky_relu(e6)), input)
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        e8 = self.relu(e8)
        e8 = torch.cat( [self.up(e8[:, :3, ...]),self.up_nearest(e8[:, 3:4, ...]),self.up(e8[:, 4:, ...])], 1)
        d1_ = self.spade_layer8_4(self.dconv1(e8), input)
        d1_ = torch.cat([self.up(d1_[:, :3, ...]), self.up_nearest(d1_[:, 3:4, ...]), self.up(d1_[:, 4:, ...])], 1)
        e7 = torch.cat([self.up(e7[:, :3, ...]), self.up_nearest(e7[:, 3:4, ...]), self.up(e7[:, 4:, ...])], 1)
        d1 = torch.cat((d1_, e7), 1)
        d1 = self.relu(d1)
        d2_ = self.spade_layer8_5(self.dconv2(d1), input)
        d2_ = torch.cat([self.up(d2_[:, :3, ...]), self.up_nearest(d2_[:, 3:4, ...]), self.up(d2_[:, 4:, ...])], 1)
        e6 = torch.cat([self.up(e6[:, :3, ...]), self.up_nearest(e6[:, 3:4, ...]), self.up(e6[:, 4:, ...])], 1)
        d2 = torch.cat((d2_, e6), 1)
        d2 = self.relu(d2)
        d3_ = self.spade_layer8_6(self.dconv3(d2), input)
        d3_ = torch.cat([self.up(d3_[:, :3, ...]), self.up_nearest(d3_[:, 3:4, ...]), self.up(d3_[:, 4:, ...])], 1)
        e5 = torch.cat([self.up(e5[:, :3, ...]), self.up_nearest(e5[:, 3:4, ...]), self.up(e5[:, 4:, ...])], 1)
        d3 = torch.cat((d3_, e5), 1)
        d3 = self.relu(d3)
        d4_ = self.spade_layer8_7(self.dconv4(d3), input)
        d4_ = torch.cat([self.up(d4_[:, :3, ...]), self.up_nearest(d4_[:, 3:4, ...]), self.up(d4_[:, 4:, ...])], 1)
        e4 = torch.cat([self.up(e4[:, :3, ...]), self.up_nearest(e4[:, 3:4, ...]), self.up(e4[:, 4:, ...])], 1)
        d4 = torch.cat((d4_, e4), 1)
        d4 = self.relu(d4)
        d5_ = self.spade_layer4_1(self.dconv5(d4), input)
        d5_ = torch.cat([self.up(d5_[:, :3, ...]), self.up_nearest(d5_[:, 3:4, ...]), self.up(d5_[:, 4:, ...])], 1)
        e3 = torch.cat([self.up(e3[:, :3, ...]), self.up_nearest(e3[:, 3:4, ...]), self.up(e3[:, 4:, ...])], 1)
        d5 = torch.cat((d5_, e3), 1)
        d5 = self.relu(d5)
        d6_ = self.spade_layer2_1(self.dconv6(d5), input)
        d6_ = torch.cat([self.up(d6_[:, :3, ...]), self.up_nearest(d6_[:, 3:4, ...]), self.up(d6_[:, 4:, ...])], 1)
        e2 = torch.cat([self.up(e2[:, :3, ...]), self.up_nearest(e2[:, 3:4, ...]), self.up(e2[:, 4:, ...])], 1)
        d6 = torch.cat((d6_, e2), 1)
        d6 = self.relu(d6)
        d7_ = self.spade_layer(self.dconv7(d6), input)
        d7_ = torch.cat([self.up(d7_[:, :3, ...]), self.up_nearest(d7_[:, 3:4, ...]), self.up(d7_[:, 4:, ...])], 1)
        e1 = torch.cat([self.up(e1[:, :3, ...]), self.up_nearest(e1[:, 3:4, ...]), self.up(e1[:, 4:, ...])], 1)
        d7 = torch.cat((d7_, e1), 1)
        d7 = self.relu(d7)
        d8 = self.dconv8(d7)
        if not self.no_tanh:
            d8 = self.tanh(d8)
        return d8
    
    
class SPADEXAttnUnet4MotionSD(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(
        self,
        channels_in=3,
        channels_out=2,
        num_filters=320,
        use_3D=False,
        norm=None,
        n_heads=8,
        context_dim=768,
        no_tanh=True,
        up_mode="bilinear",
    ):
        super(SPADEXAttnUnet4MotionSD, self).__init__()

        conv_layer = get_conv_layer_v1(norm, use_3D=use_3D)

        self.conv1 = conv_layer(channels_in, num_filters, 3, 1, 1) #256
        self.conv2 = conv_layer(num_filters, num_filters, 3, 1, 1) #128
        self.conv3 = conv_layer(num_filters, num_filters * 2, 3, 1, 1) #64
        self.conv4 = conv_layer(num_filters * 2, num_filters * 2, 3, 1, 1) ### input_blocks.4.1 #32
        self.conv5 = conv_layer(num_filters * 2, num_filters * 2, 3, 1, 1) ### input_blocks.5.1 #32
        self.conv6 = conv_layer(num_filters * 2, num_filters * 4, 3, 1, 1) ### input_blocks.7.1 #16
        self.conv7 = conv_layer(num_filters * 4, num_filters * 4, 3, 1, 1) ### input_blocks.8.1 #16
        self.conv8 = conv_layer(num_filters * 4, num_filters * 4, 3, 1, 1) ### middle block #16

        self.avgpool1 = nn.AvgPool2d(2, stride=2)
        self.avgpool2 = nn.AvgPool2d(2, stride=2)
        self.avgpool3 = nn.AvgPool2d(2, stride=2)
        self.avgpool4 = nn.AvgPool2d(2, stride=2)
        self.avgpool6 = nn.AvgPool2d(2, stride=2)
        self.avgpool8 = nn.AvgPool2d(2, stride=2)

        
        self.attn_input_block_1 = attention.SpatialTransformer(num_filters * 2, n_heads, num_filters//n_heads * 2, context_dim=context_dim)
        self.attn_input_block_2 = attention.SpatialTransformer(num_filters * 2, n_heads, num_filters//n_heads * 2, context_dim=context_dim)
        self.attn_input_block_3 = attention.SpatialTransformer(num_filters * 4, n_heads, num_filters//n_heads * 4, context_dim=context_dim)
        self.attn_input_block_4 = attention.SpatialTransformer(num_filters * 4, n_heads, num_filters//n_heads * 4, context_dim=context_dim)
        self.attn_middle_block = attention.SpatialTransformer(num_filters * 4, n_heads, num_filters//n_heads * 4, context_dim=context_dim)
        self.attn_output_block_4 = attention.SpatialTransformer(num_filters * 4, n_heads, num_filters//n_heads * 4, context_dim=context_dim)
        self.attn_output_block_3 = attention.SpatialTransformer(num_filters * 4, n_heads, num_filters//n_heads * 4, context_dim=context_dim)
        self.attn_output_block_2 = attention.SpatialTransformer(num_filters * 2, n_heads, num_filters//n_heads * 2, context_dim=context_dim)
        self.attn_output_block_1 = attention.SpatialTransformer(num_filters * 2, n_heads, num_filters//n_heads * 2, context_dim=context_dim)

        self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False)

        self.dconv1 = conv_layer(num_filters * 4, num_filters * 4, 3, 1, 1) ### output_blocks.4.1 #16
        self.dconv2 = conv_layer(num_filters * 4 * 2, num_filters * 4, 3, 1, 1) ### output_blocks.5.1 #16
        self.dconv3 = conv_layer(num_filters * 4 * 2, num_filters * 2, 3, 1, 1) ### output_blocks.6.1 #32
        self.dconv4 = conv_layer(num_filters * 2 * 2, num_filters * 2, 3, 1, 1) ### output_blocks.7.1 #32
        self.dconv5 = conv_layer(num_filters * 2 * 2, num_filters * 2, 3, 1, 1) #64
        self.dconv6 = conv_layer(num_filters * 2 * 2, num_filters, 3, 1, 1) #128
        self.dconv7 = conv_layer(num_filters * 2, num_filters, 3, 1, 1) #256
        self.dconv8 = conv_layer(num_filters * 2, channels_out, 3, 1, 1) #512

        norm_layer = nn.InstanceNorm2d

        self.spade_layer = SPADE(norm_layer, num_filters, channels_in)#self.batch_norm = norm_layer(num_filters)
        self.spade_layer2_0 = SPADE(norm_layer, num_filters, channels_in)#self.batch_norm2_0 = norm_layer(num_filters * 2)
        self.spade_layer2_1 = SPADE(norm_layer, num_filters, channels_in)#self.batch_norm2_1 = norm_layer(num_filters * 2)
        self.spade_layer4_0 = SPADE(norm_layer, num_filters*2, channels_in)#self.batch_norm4_0 = norm_layer(num_filters * 4)
        self.spade_layer4_1 = SPADE(norm_layer, num_filters*2, channels_in)#self.batch_norm4_1 = norm_layer(num_filters * 4)
        self.spade_layer8_0 = SPADE(norm_layer, num_filters*2, channels_in)#self.batch_norm8_0 = norm_layer(num_filters * 8)
        self.spade_layer8_1 = SPADE(norm_layer, num_filters*2, channels_in)#self.batch_norm8_1 = norm_layer(num_filters * 8)
        self.spade_layer8_2 = SPADE(norm_layer, num_filters*4, channels_in)#self.batch_norm8_2 = norm_layer(num_filters * 8)
        self.spade_layer8_3 = SPADE(norm_layer, num_filters*4, channels_in)#self.batch_norm8_3 = norm_layer(num_filters * 8)
        self.spade_layer8_4 = SPADE(norm_layer, num_filters*4, channels_in)#self.batch_norm8_4 = norm_layer(num_filters * 8)
        self.spade_layer8_5 = SPADE(norm_layer, num_filters*4, channels_in)#self.batch_norm8_5 = norm_layer(num_filters * 8)
        self.spade_layer8_6 = SPADE(norm_layer, num_filters*2, channels_in)#self.batch_norm8_6 = norm_layer(num_filters * 8)
        self.spade_layer8_7 = SPADE(norm_layer, num_filters*2, channels_in)#self.batch_norm8_7 = norm_layer(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.no_tanh = no_tanh

    def forward(self, input, context=None, qk=None):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.avgpool1(self.conv1(input)) #256
        e2 = self.spade_layer2_0(self.avgpool2(self.conv2(self.leaky_relu(e1))), input) #128
        e3 = self.spade_layer4_0(self.avgpool3(self.conv3(self.leaky_relu(e2))), input) #64
        e4 = self.spade_layer8_0(self.avgpool4(self.conv4(self.leaky_relu(e3))), input) #32
        e4 = self.attn_input_block_1(e4, context) #32
        e5 = self.spade_layer8_1(self.conv5(self.leaky_relu(e4)), input) #32
        e5 = self.attn_input_block_2(e5, context) #32
        e6 = self.spade_layer8_2(self.avgpool6(self.conv6(self.leaky_relu(e5))), input) #1
        e6 = self.attn_input_block_3(e6, context) #16
        e7 = self.spade_layer8_3(self.conv7(self.leaky_relu(e6)), input) #16
        e7 = self.attn_input_block_4(e7, context) #16
        # No batch norm on output of Encoder
        e8 = self.avgpool8(self.conv8(self.leaky_relu(e7))) #16
        e8 = self.attn_middle_block(e8, context) #16

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 1 x 1
        d1_ = self.spade_layer8_4(self.dconv1(self.up(self.relu(e8))), input) #16
        d1_ = self.attn_output_block_4(d1_, context) #16
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.spade_layer8_5(self.dconv2(self.relu(d1)), input) #16
        d2_ = self.attn_output_block_3(d2_, context) #16
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.spade_layer8_6(self.dconv3(self.up(self.relu(d2))), input) #32
        d3_ = self.attn_output_block_2(d3_, context) #32
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.spade_layer8_7(self.dconv4(self.relu(d3)), input) #32
        d4_ = self.attn_output_block_1(d4_, context) #32
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.spade_layer4_1(self.dconv5(self.up(self.relu(d4))), input) #64
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.spade_layer2_1(self.dconv6(self.up(self.relu(d5))), input) #128
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.spade_layer(self.dconv7(self.up(self.relu(d6))), input) #256
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.up(self.relu(d7))) #512
        if not self.no_tanh:
            d8 = self.tanh(d8)
            
        return d8

class Identity(nn.Module):
    def forward(self, input):
        return input


class ResNet_Block(nn.Module):
    def __init__(self, in_c, in_o, opt, downsample=None):
        super().__init__()
        bn_noise1 = LinearNoiseLayer(opt, output_sz=in_c)
        bn_noise2 = LinearNoiseLayer(opt, output_sz=in_o)

        conv_layer = get_conv_layer(opt)

        conv_aa = conv_layer(in_c, in_o, 3, 1, 1)
        conv_ab = conv_layer(in_o, in_o, 3, 1, 1)

        conv_b = conv_layer(in_c, in_o, 1, 0, 1)

        if downsample == "Down":
            norm_downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif downsample == "Up":
            norm_downsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        elif downsample:
            norm_downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            norm_downsample = Identity()

        self.ch_a = nn.Sequential(
            bn_noise1,
            nn.ReLU(),
            conv_aa,
            bn_noise2,
            nn.ReLU(),
            conv_ab,
            norm_downsample,
        )
        if downsample or (in_c != in_o):
            self.ch_b = nn.Sequential(conv_b, norm_downsample)
        else:
            self.ch_b = Identity()

    def forward(self, x):
        x_a = self.ch_a(x)
        x_b = self.ch_b(x)

        return x_a + x_b


class ResNet_Block_Pconv2(nn.Module):
    def __init__(self, in_c, in_o, opt, downsample=None, ks=3, activation='Relu', padding=1, stride=1,  dilation=1):
        super().__init__()
        if "pbn" in opt.pconv:
            self.bn_noise1 = PartialLinearNoiseLayer(opt, output_sz=in_c)
            self.bn_noise2 = PartialLinearNoiseLayer(opt, output_sz=in_o)
            self.pconv_opt = 'pbn'
        else:
            self.bn_noise1 = LinearNoiseLayer(opt, output_sz=in_c)
            self.bn_noise2 = LinearNoiseLayer(opt, output_sz=in_o)
            self.pconv_opt = 'bn'

        pconv_layer = get_pconv_layer(opt)
        conv_layer = get_conv_layer(opt)


        self.conv_aa = pconv_layer(in_c, in_o, ks, padding, stride, dilation)
        self.conv_ab = pconv_layer(in_o, in_o, ks, padding, stride, dilation)

        if "woresbias" in opt.pconv:
            self.conv_b = conv_layer(in_c, in_o, 1, 0, 1, bias=False)
        else:
            self.conv_b = conv_layer(in_c, in_o, 1, 0, 1)
        if downsample == "Down":
            self.norm_downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.norm_downsample_mask = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif downsample == "Up":
            self.norm_downsample = nn.Upsample(scale_factor=2, mode="bilinear",align_corners=False)
            self.norm_downsample_mask = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            self.norm_downsample = Identity()
            self.norm_downsample_mask = Identity()
        self.downsample = downsample
        self.in_c = in_c
        self.in_o = in_o
        self.debug_opt = opt.pconv
        if activation == "Relu":
            self.activation = nn.ReLU()
        elif activation == "LRelu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "PRelu":
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x, mask):
        if "debug" in self.debug_opt:
            print("mask ratio:{}.".format(mask.sum()/(mask.shape[0]*mask.shape[1]*mask.shape[2]*mask.shape[3])))
        # x: [1, 64, 256, 256]
        # mask: [1, 64, 256, 256]
        # x_a
        if "pbn" in self.pconv_opt:
            x_a, mask_a = self.bn_noise1(x, mask) # [1, 64, 256, 256]
        else:
            x_a = self.bn_noise1(x)
            mask_a = mask

        x_a = self.activation(x_a)
        x_a, mask_a = self.conv_aa(x_a, mask_a)

        if "pbn" in self.pconv_opt:
            x_a, mask_a = self.bn_noise2(x_a, mask_a)
        else:
            x_a = self.bn_noise2(x_a)

        x_a = self.activation(x_a)
        x_a, mask_a = self.conv_ab(x_a, mask_a)
        x_a = self.norm_downsample(x_a)
        mask_a = self.norm_downsample_mask(mask_a)
        # x_b
        if self.downsample or (self.in_c != self.in_o):
            x_b = self.conv_b(x)
            x_b = self.norm_downsample(x_b)
        else:
            x_b = Identity()(x)
        return x_a + x_b, mask_a


class ResNetEncoder_with_Z(nn.Module):
    """ Modified implementation of the BigGAN model.
    """

    def __init__(
        self,
        opt,
        channels_in=3,
        channels_out=64,
        downsample=True,
        model_type=None,
    ):
        super().__init__()
        if not model_type:
            arch = get_resnet_arch(opt.refine_model_type, opt, channels_in)
            #opt.refine_model_type == 'resnet_256W8UpDown64'
        else:
            arch = get_resnet_arch(model_type, opt, channels_in)

        gblocks = []
        for l_id in range(1, len(arch["layers_enc"])-1):
            gblock = ResNet_Block(
                arch["layers_enc"][l_id - 1],
                arch["layers_enc"][l_id],
                opt,
                (downsample and arch["downsample"][l_id - 1]),
            )
            gblocks += [gblock]

        gblock = ResNet_Block(
            arch["layers_enc"][len(arch["layers_enc"]) - 2],
            arch["layers_enc"][len(arch["layers_enc"]) - 1] + 1,
            opt,
            (downsample and arch["downsample"][len(arch["layers_enc"]) - 2]),
        )
        gblocks += [gblock]

        self.gblocks = nn.Sequential(*gblocks)
        self.opt = opt

    def forward(self, x):
        output = self.gblocks(x)
        return output[:, :-1,...], output[:,-1:,...]


class ResNetDecoderPconv2(nn.Module):
    """ Modified implementation of the BigGAN model. """

    def __init__(self, opt, channels_in=64, channels_out=3, use_tanh=False):
        super().__init__()

        arch = get_resnet_arch(opt.refine_model_type, opt)

        eblocks = []
        for l_id in range(1, len(arch["layers_dec"])):
            eblock = ResNet_Block_Pconv2(
                arch["layers_dec"][l_id - 1],#opt.ngf,
                arch["layers_dec"][l_id],#opt.ngf*2
                opt,
                arch["upsample"][l_id - 1],#"Down"
                arch["ks_dec"][l_id - 1] if "ks_dec" in arch else 3,
                arch["activation"][l_id - 1] if "activation" in arch else None,
            )
            eblocks += [eblock]
        self.eblocks = nn.Sequential(*eblocks)

        self.pconv_setting = opt.pconv

    def forward(self, x):
        mask = (x!=0).float()
        if "mask1" in self.pconv_setting:
            mask[:] = 1.0
        x_a, mask_a = self.eblocks[0](x, mask)
        for eblock in self.eblocks[1:]:
            x_a, mask_a = eblock(x_a, mask_a)
        return x_a


class SPADEUnet4SoftmaxSplating(nn.Module):
    def __init__(self, opt, model_type=None):
        super().__init__()

        self.opt = opt
        self.W = opt.fineSize
        self.motion_norm = opt.motion_norm

        self.encoder = ResNetEncoder_with_Z(opt)

        self.euler_integration = EulerIntegration(opt)

        self.softsplater = softsplat.ModuleSoftsplat('summation')

        self.projector = ResNetDecoderPconv2(opt, channels_in=64, channels_out=3, use_tanh=False)


    def forward(self, input, index=None):
        """ Forward pass of a view synthesis model with a voxel latent field.
        """
        # Input values
        start_img = input[:,:3,...] # B, 3, W, W
        end_img = input[:,3:6,...]
        motion = input[:,6:,...]

        bs = start_img.shape[0]
        
        if isinstance(index, list):
            start_index = index[0]
            middle_index = index[1]
            end_index = index[2]
        elif len(index.shape) > 1:
            start_index = index[:, 0]
            middle_index = index[:, 1]
            end_index = index[:, 2]
        else:
            start_index = index[0]
            middle_index = index[1]
            end_index = index[2]

        start_fs, Z_f = self.encoder(start_img)
        end_fs, Z_p = self.encoder(end_img)

        # Regreesed Motion
        flow = motion.cuda()
        flow_f = self.euler_integration(flow, middle_index.long() - start_index.long())
        flow_p = self.euler_integration(-flow, end_index.long() + 1 - middle_index.long())
        #linear
        #flow_f = flow * (middle_index.long() - start_index.long() - 1).view(2,1,1,1)
        #flow_p = -flow * ( end_index.long() + 1 - middle_index.long() - 1).view(2,1,1,1)

        alpha = (1.0 - (middle_index.float() - start_index.float()).float() / (
                                end_index.float() - start_index.float() + 1.0).float()).view(bs, 1, 1, 1).cuda()
        # forward
        Z_f = Z_f.view(bs, 1, self.W, self.W)

        Z_f_norm = Z_f - Z_f.max()
        tenInput_f = torch.cat([start_fs * Z_f_norm.exp() * alpha, Z_f_norm.exp() * alpha], 1)  # B, 65, W, W

        gen_fs_f = self.softsplater(tenInput=tenInput_f,
                                    tenFlow=flow_f,
                                    tenMetric=start_fs.new_ones((start_fs.shape[0], 1, start_fs.shape[2],
                                                                    start_fs.shape[3])))  # B, 65, W, W
        tenNormalize = gen_fs_f[:, -1:, :, :]  # B, 1, W, W
        gen_fs_f = gen_fs_f[:, :-1, :, :]
        gen_fs = gen_fs_f

        # backward
        Z_p = Z_p.view(bs, 1, self.W, self.W)
        Z_p_norm = Z_p - Z_p.max()
        tenInput_p =torch.cat([end_fs * Z_p_norm.exp() * (1 - alpha), Z_p_norm.exp() * (1 - alpha)], 1)

        gen_fs_p = self.softsplater(tenInput=tenInput_p,
                                    tenFlow=flow_p,
                                    tenMetric=start_fs.new_ones(
                                            (start_fs.shape[0], 1, start_fs.shape[2], start_fs.shape[3])))
        tenNormalize += gen_fs_p[:, -1:, :, :]
        gen_fs_p = gen_fs_p[:, :-1, :, :]
        gen_fs += gen_fs_p

        tenNormalize = torch.clamp(tenNormalize, min=1e-8)
        gen_fs = gen_fs / tenNormalize

        gen_img = self.projector(gen_fs.contiguous())  # B,3,W,W, [-1,1]

        gen_img = nn.Tanh()(gen_img)

        return gen_img

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
