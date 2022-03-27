import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from networks.blocks import Conv2d_cd, ResNetBlock, Conv2dBlock
import numpy as np

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, mode='self'):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        
        self.mode = mode
        reduction = 8
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//reduction , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//reduction , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.D = np.sqrt(in_dim//reduction)
        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x, y=None):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        if self.mode == 'self':
            proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
            energy =  torch.bmm(proj_query,proj_key) # transpose check
            attention = self.softmax(energy / self.D) # BX (N) X (N) 
            proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        else:
            proj_key = self.key_conv(y).view(m_batchsize, -1, width*height)
            energy =  torch.bmm(proj_query,proj_key) # transpose check
            attention = self.softmax(energy / self.D) # BX (N) X (N) 
            proj_value = self.value_conv(y).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        
        if self.mode == 'self':
            out = self.gamma*out + x
        else:
            out = torch.cat([out, x],dim=1)
        return out


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap, mode='hdr'): 
        # Nx12x8x16x16
        device = bilateral_grid.get_device()
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([hg, wg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, 'bilinear', align_corners=True) # Nx12xHxW
        return coeff.squeeze(2)

class ApplyCoeffs(nn.Module):
    def __init__(self, use_norm=False):
        super(ApplyCoeffs, self).__init__()
        self.use_norm = use_norm

    def denormalize(self, x, isMask=False):
        if isMask:
            mean = 0
            std=1
        else:
            mean = torch.zeros_like(x)
            mean[:,0,:,:] = .485
            mean[:,1,:,:] = .456
            mean[:,2,:,:] = .406
            std = torch.zeros_like(x)
            std[:,0,:,:] = 0.229
            std[:,1,:,:] = 0.224
            std[:,2,:,:] = 0.225
        x = (x*std + mean) #*255
        return x # change the range into [0,1]
    
    def norm(self, x):
        mean = torch.zeros_like(x)
        mean[:,0,:,:] = .485
        mean[:,1,:,:] = .456
        mean[:,2,:,:] = .406
        std = torch.zeros_like(x)
        std[:,0,:,:] = 0.229
        std[:,1,:,:] = 0.224
        std[:,2,:,:] = 0.225
        x = (x - mean) / std #*255
        return x

    def forward(self, coeff, full_res_input):
        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''
        full_res_input = self.denormalize(full_res_input)
        # coeff[:,:,:20] = coeff[:,:,50:70]
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]
        
        # return self.norm(torch.cat([R, G, B], dim=1))
        if self.use_norm:
            return self.norm(torch.cat([R, G, B], dim=1))
        else:
            return torch.cat([R, G, B], dim=1)

class GuideNN(nn.Module):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = Conv2dBlock(3, 16, ks=1, st=1, padding=0, norm='bn')
        self.conv2 = Conv2dBlock(16, 1, ks=1, st=1, padding=0, norm='none', activation='tanh') #nn.Tanh, nn.Sigmoid

    def forward(self, x):
        return self.conv2(self.conv1(x))#.squeeze(1)

class Coeffs(nn.Module):
    def __init__(self, nin=4, nout=3, params=None):
        super(Coeffs, self).__init__()
        self.params = params
        self.nin = nin 
        self.nout = nout
        
        lb = params['luma_bins']
        cm = params['channel_multiplier']
        sb = params['spatial_bin']
        bn = params['batch_norm']
        
        theta = params['theta']
        nsize = params['net_input_size']
        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(nsize/sb))
        self.splat_features = nn.ModuleList()
        self.lp_features = nn.ModuleList()
        prev_ch = 3 #3
        # Downsample
        for i in range(n_layers_splat):
            use_bn = bn if i > 0 else False
            self.splat_features.append(Conv2d_cd(prev_ch, cm*(2**i)*lb, 3, 1, 1, use_bn=use_bn, actv='relu', theta=theta))
            self.splat_features.append(nn.MaxPool2d(2,2,0))
            prev_ch = splat_ch = cm*(2**i)*lb
        # ResNet Blocks
        self.res_blks = nn.ModuleList()
        for i in range(3):
            self.res_blks.append(ResNetBlock(prev_ch, prev_ch))
        #Self-attention
        self.sa = SelfAttention(prev_ch)
        
        self.conv_out = nn.Sequential(*[
            Conv2dBlock(prev_ch, 8*cm*lb, ks=3, st=1, padding=1, norm='bn'),
            Conv2dBlock(8*cm*lb, lb*nin*nout, ks=1, st=1, padding=0, norm='none', activation='none')
        ])
        
        # predicton
        self.conv_out = Conv2dBlock(8*cm*lb, lb*nout*nin, ks=1, st=1, padding=0, norm='none', activation='none')

   
    def forward(self, lowres_input):
        params = self.params
        bs = lowres_input.shape[0]
        lb = params['luma_bins']
        cm = params['channel_multiplier']
        sb = params['spatial_bin']

        x = lowres_input
        for layer in self.splat_features:
            x = layer(x)   
            
        for layer in self.res_blks:
            x = layer(x)
        
        x = self.sa(x)
        x = self.conv_out(x) # 1,96,16,16
        
        s = x.shape
        y = torch.stack(torch.split(x, self.nin*self.nout, 1),2) # B x Coefs x Luma x Spatial x Spatial -> (B, 12,8,16,16)
        return y


class HDRPointwiseNN(nn.Module):

    def __init__(self, opt):
        super(HDRPointwiseNN, self).__init__()
        params = {'luma_bins':opt.luma_bins, 'channel_multiplier':opt.channel_multiplier, 'spatial_bin':opt.spatial_bin, 
                'batch_norm':opt.batch_norm, 'net_input_size':opt.net_input_size, 'theta':opt.theta}
        self.coeffs = Coeffs(params=params)
        self.guide = GuideNN(params=params)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

        self.mean = [.485, .456, .406]
        self.std = [.229, .224, .225]
        self.max_val = [(1-m)/s for m,s in zip(self.mean, self.std)]
        self.min_val = [(0-m)/s for m,s in zip(self.mean, self.std)]

    def clip(self, x):
        y = x.new(x.size())
        for i in range(3):
            y[:,i,:,:] = torch.clamp(x[:,i,:,:], min=self.min_val[i], max=self.max_val[i])
        return y

    def norm(self, x):
        mean = torch.zeros_like(x)
        mean[:,0,:,:] = .485
        mean[:,1,:,:] = .456
        mean[:,2,:,:] = .406
        std = torch.zeros_like(x)
        std[:,0,:,:] = 0.229
        std[:,1,:,:] = 0.224
        std[:,2,:,:] = 0.225
        x = (x - mean) / std #*255
        return x

    def forward(self, lowres, fullres):
        coeffs = self.coeffs(lowres)
        guide = self.guide(fullres)
        slice_coeffs = self.slice(coeffs, guide)
        illu_out = self.apply_coeffs(slice_coeffs, fullres).sigmoid()
        
        out = self.clip(illu_out)
        return out, guide