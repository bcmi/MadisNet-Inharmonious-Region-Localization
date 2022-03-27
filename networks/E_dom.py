import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import torchvision.models as models
from networks.blocks import PartialConv2d



class DomainEncoder(nn.Module):
    def __init__(self, style_dim):
        super(DomainEncoder, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.actv = nn.ReLU(True)
        self.maxpooling = nn.MaxPool2d(2,2,0,ceil_mode=False)
        # encoder 1
        self.conv0 = PartialConv2d(vgg16.features[0])
        self.conv1 = PartialConv2d(vgg16.features[2])
        # encoder 2
        self.conv2 = PartialConv2d(vgg16.features[5])
        self.conv3 = PartialConv2d(vgg16.features[7])
        # encoder 3
        self.conv4 = PartialConv2d(vgg16.features[10])
        self.conv5 = PartialConv2d(vgg16.features[12])
        self.conv6 = PartialConv2d(vgg16.features[14])
         # fix the encoder
        for i in range(7):
            for param in getattr(self, 'conv{:d}'.format(i)).parameters():
                param.requires_grad = False
        # adaptor
        self.adaptor1 = nn.Conv2d(self.conv1.out_channels, style_dim, kernel_size=1, stride=1, bias=False)
        self.adaptor1 = PartialConv2d(self.adaptor1)

        self.adaptor2 = nn.Conv2d(self.conv3.out_channels, style_dim, kernel_size=1, stride=1, bias=False)
        self.adaptor2 = PartialConv2d(self.adaptor2)

        self.adaptor3 = nn.Conv2d(self.conv6.out_channels, style_dim, kernel_size=1, stride=1, bias=False)
        self.adaptor3 = PartialConv2d(self.adaptor3)

        self.weight = nn.Parameter(torch.ones((3,), dtype=torch.float32),  requires_grad=True)
            
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
    
    def train(self, mode=True):
        self.adaptor1.train()
        self.adaptor2.train()
        self.adaptor3.train()
        self.weight.requires_grad = True
       
       

    def forward(self, input, mask, eps=1e-8):
        """Standard forward."""
        xb = input
        mb = mask

        # Encoder
        xb, mb = self.conv0(xb, mb)
        xb = self.actv(xb)
        xb, mb = self.conv1(xb, mb)
        xb = self.actv(xb)
        x1b = self.maxpooling(xb)
        m1b = self.maxpooling(mb)

        xb, mb = self.conv2(x1b, m1b)
        xb = self.actv(xb)
        xb, mb = self.conv3(xb, mb)
        xb = self.actv(xb)
        x2b = self.maxpooling(xb)
        m2b = self.maxpooling(mb)

        xb, mb = self.conv4(x2b, m2b)
        xb = self.actv(xb)
        xb, mb = self.conv5(xb, mb)
        xb = self.actv(xb)
        xb, mb = self.conv6(xb, mb)
        xb = self.actv(xb)
        x3b = self.maxpooling(xb)
        m3b = self.maxpooling(mb)

        # Domain code
        w = self.weight.sigmoid()
        w = w / (w.sum() + eps)
        x1b,_ = self.adaptor1(x1b, m1b)
        x1b = self.avg_pooling(x1b)
        x2b,_ = self.adaptor2(x2b, m2b)
        x2b = self.avg_pooling(x2b)
        x3b,_ = self.adaptor3(x3b, m3b)
        x3b = self.avg_pooling(x3b)
        s = w[0]*x1b + w[1]*x2b + w[2]*x3b
        
        return s



