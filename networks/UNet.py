import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
from networks.blocks import BasicBlock, Bottleneck


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, skip_channels, out_channels, upsample=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if upsample:
            self.up = nn.Sequential(*[
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, skip_channels, 3,1,1)
            ])
        else:
            self.up = nn.Conv2d(in_channels, skip_channels, 3,1,1)

        self.conv = DoubleConv(skip_channels*2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1,x2), dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, n_class=1, nf=64, n_downs=4, backbone='resnet34'):
        super(UNet, self).__init__()
        if backbone == 'none':
            self.in_conv = DoubleConv(in_ch, nf)
        elif 'resnet' in backbone:
            self.in_conv = nn.Sequential(*[
                nn.Conv2d(in_ch, nf, 3,1,1,bias=False),
                nn.BatchNorm2d(nf),
                nn.ReLU(True)
            ])
        

        self.downs = nn.ModuleDict()
        self.ups = nn.ModuleDict()
        self.n_downs = n_downs
        extra_dim = 512
        if 'resnet' in backbone:
            
            if backbone == 'resnet34':
                resnet = models.resnet34(True)
                dims = [64,64,128,256,512] 
                if n_downs-4>0: dims += [extra_dim]*(n_downs-4)
            elif backbone == 'resnet50':
                resnet = models.resnet50(True)
                dims = [64,64*4,128*4,256*4,512*4]
                if n_downs-4>0: dims += [extra_dim]*(n_downs-4)
        elif backbone == 'none':
            dims = []
            for i in range(min(4+1, n_downs+1)):
                dims.append(nf*(2**i))
            if n_downs-4>0:
                if n_downs-4>0: dims += [extra_dim]*(n_downs-4)
        # Build encoder
        for i in range(n_downs):
            if backbone == 'none':
                self.downs['d{}'.format(i)] = Down(dims[i], dims[i+1])
            elif 'resnet' in backbone:
                if i < 4:
                    self.downs['d{}'.format(i)] = getattr(resnet, 'layer{}'.format(i+1))
                elif i == 4:
                    self.downs['d{}'.format(i)] = nn.Sequential(*[
                        nn.MaxPool2d(3,2,1), 
                        BasicBlock(dims[-1], extra_dim), 
                        BasicBlock(extra_dim, extra_dim),
                        BasicBlock(extra_dim, extra_dim)
                        ])
                else:
                    self.downs['d{}'.format(i)] = nn.Sequential(*[
                        nn.MaxPool2d(3,2,1), 
                        BasicBlock(extra_dim, extra_dim), 
                        BasicBlock(extra_dim, extra_dim),
                        BasicBlock(extra_dim, extra_dim)
                        ])
        # Build Decoder
        for i in range(n_downs):
            if i == n_downs - 1:
                self.ups['u{}'.format(i)] = Up(dims[i+1], dims[i+1], dims[i], False)
            else:
                self.ups['u{}'.format(i)] = Up(dims[i+1], dims[i+1], dims[i])

        self.bottleneck = DoubleConv(dims[-1], dims[-1])
        self.out_conv = nn.Sequential(*[
            nn.Conv2d(nf, n_class, 1,1,0)
        ])

    def forward(self, x):
        hx = self.in_conv(x)
        enc_xs = []
        for i in range(self.n_downs):
            hx = self.downs['d{}'.format(i)](hx)
            enc_xs.append(hx)
        hx = self.bottleneck(hx)
        for i in range(self.n_downs):
            idx = self.n_downs - i - 1
            hx = self.ups['u{}'.format(idx)](hx, enc_xs[idx])
        
        logits = self.out_conv(hx)
        logits  = F.interpolate(logits , x.shape[2:][::-1], mode='bilinear', align_corners=True)
        logits  = logits.sigmoid()
        return {"mask":[logits]}
    