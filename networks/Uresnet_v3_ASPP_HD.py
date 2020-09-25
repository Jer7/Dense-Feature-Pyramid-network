import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools

import sys, os

# import ../db.py


from libs import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

# from Unet_files.unet_parts import up, outconv

def conv3x3(in_planes, out_planes, stride=1, bias = False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out

class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, features, inner_features=256, out_features=512, dilations=(6, 12, 18)):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   InPlaceABNSync(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )
        
    def forward(self, x):

        _, _, h, w = x.size()

        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle

class Edge_Module(nn.Module):
    #[256,512,1024] are number of channels of conv2, conv3 and conv4 output of resnet
    def __init__(self,in_fea=[256,512,1024], mid_fea=256, out_fea=2):
        super(Edge_Module, self).__init__()
        
        self.conv1 =  nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
            ) 
        self.conv2 =  nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
            )  
        self.conv3 =  nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.reduce_depth =  nn.Sequential(
            nn.Conv2d(mid_fea*3, 512, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(512)
        )
        self.conv4 = nn.Conv2d(mid_fea,out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea*3,out_fea, kernel_size=1, padding=0, dilation=1, bias=True)


    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()
        #conv 1x1 on resnet:layer2
        edge1_fea = self.conv1(x1)
        #conv3x3 on edge1_fea 
        edge1 = self.conv4(edge1_fea)
        #conv 1x1 on resnet:layer3
        edge2_fea = self.conv2(x2)
        #conv3x3 on edge2_fea 
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)        
        
        #upsample edge2_fea to resnet:layer2 size
        edge2_fea =  F.interpolate(edge2_fea, size=(h, w), mode='bilinear',align_corners=True) 
        #upsample edge3_fea resnet:layer2 size
        edge3_fea =  F.interpolate(edge3_fea, size=(h, w), mode='bilinear',align_corners=True) 
        #upsample edge2 resnet:layer2 size
        edge2 =  F.interpolate(edge2, size=(h, w), mode='bilinear',align_corners=True)
        #-------------
        edge3 =  F.interpolate(edge3, size=(h, w), mode='bilinear',align_corners=True) 
        #concat edge for retropropa
        edge = torch.cat([edge1, edge2, edge3], dim=1)
        #concat edge features for fuse it with parsed features. edge_fea depth = 256+256+256
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        #reduce depth of edge
        # edge_fea = self.reduce_depth(edge_fea)

        edge = self.conv5(edge)
         
        return edge, edge_fea

class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            # conv 3x3 + reduce channels to out_features
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            )

    def _make_stage(self, features, out_features, size):
        #pool input to (size, size)
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        # conv 1x1 to reduce channels
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        #interpolate up/down sample according to the given size. Here it upsample the output of the pyramid pooling to input size (conv5)
        #Prior has all pooling output + conv5 output
        priors = [ F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        # bottleneck: conv3x3 with padding + reduce channels to out_features
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class Decoder_Module(nn.Module):

    def __init__(self, num_classes):
        super(Decoder_Module, self).__init__()
        #xt/psp features has depth 256
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
            )
        #xl/conv2 feature has depth 256 --> 48
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(48)
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )
        # self.RC1 = Residual_Covolution(512, 1024, num_classes)
        # self.RC2 = Residual_Covolution(512, 1024, num_classes)
        # self.RC3 = Residual_Covolution(512, 1024, num_classes)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    # xt = psp module output////// xl: resnet layer2 output
    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        #upsample
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        
        # xt/psp + xl/conv2 = 256 + 48
        x = torch.cat([xt, xl], dim=1)
        #x depth 256
        x = self.conv4(x)
        #conv1x1 to nb of class for retropropa
        seg = self.conv5(x)
        return seg, x 



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()

        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #Just chaneg the stride if you want to change the H,W of layer
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #dilation is the number space between the values of a kernel (atrous)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1,1,1))

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        # shape the bypassing input channel to the same as the output of the res block (downsampling)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        # generate_multi_grid returns 1 when multi_grid is not tuple (1)
        # generate_multi_grid return grids[index%len(grids)] when multi_grid is tuple (1,1,1)
        #  I dont get the use of generate_multi_grid.
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        #Create 1 residual block
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        # update the nb of channels of the new bypassing input channel 
        self.inplanes = planes * block.expansion
        #Create the other residual blocks
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        #conv1
        # x = self.prefix(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        
        x = self.relu2(self.bn2(self.conv2(x)))
        
        x = self.relu3(self.bn3(self.conv3(x)))
        
        #conv2
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class up(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            InPlaceABNSync(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            InPlaceABNSync(out_ch)
        )
    def forward(self, x1, x2):
        _, _, h, w = x2.size()
        #upsample
        x1 = F.interpolate(x1, size=(h, w), mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class catcat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(catcat, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            InPlaceABNSync(out_ch)
        )
        self.conv3 = nn.Sequential(
            conv3x3(out_ch,out_ch),
            InPlaceABNSync(out_ch)
        )
    def forward(self, x1, x2):
        _, _, h, w = x2.size()
        x1 = self.conv1(x1)
        x1 = F.interpolate(x1, size=(h, w), mode='bilinear', align_corners=True)
        x1 = self.conv3(x1)
        x = torch.cat([x2, x1], dim=1)
        return x

class up_HD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_HD, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            InPlaceABNSync(out_ch)
        )
        self.conv3 = nn.Sequential(
            conv3x3(out_ch,out_ch),
            InPlaceABNSync(out_ch)
        )
    def forward(self, x, y):
        _, _, h, w = y.size()
        x = self.conv1(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        x = self.conv3(x)
        return x

class up_ASPP(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, inner_features, out_ch):
        super(up_ASPP, self).__init__()
        self.conv = nn.Sequential(
            ASPPModule(in_ch, inner_features, out_ch),
            InPlaceABNSync(out_ch)
        )
    def forward(self, x1, x2):
        _, _, h, w = x2.size()
        #upsample
        x1 = F.interpolate(x1, size=(h, w), mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x



class Uresnet(nn.Module):
    def __init__(self, resnet, num_classes):
        super(Uresnet, self).__init__()

        # Take pretrained resnet, except AvgPool and FC
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1 
        self.relu1 = resnet.relu1
        self.conv2 = resnet.conv2
        self.bn2 = resnet.bn2 
        self.relu2 = resnet.relu2
        self.conv3 = resnet.conv3
        self.bn3 = resnet.bn3 
        self.relu3 = resnet.relu3

        # self.prefix = orig_resnet.prefix
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # #make layer3=layer4=layer5= input/8
        # for n, m in self.layer3.named_modules():
        #     if 'conv2' in n:
        #         m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        #     elif 'downsample.0' in n:
        #         m.stride = (1, 1)
        # for n, m in self.layer4.named_modules():
        #     if 'conv2' in n:
        #         m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
        #     elif 'downsample.0' in n:
        #         m.stride = (1, 1)

        self.layer5 = ASPPModule(2048,256,512)

        # self.stabilization = nn.Sequential(
        #     nn.Conv2d(512*4, 512, kernel_size=1, padding=0, dilation=1, bias=False),
        #     InPlaceABNSync(512),
        #     nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            
        # )
        self.stabilization = nn.Sequential(
            InPlaceABNSync(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            
        )

        # self.c1 = nn.Conv2d(128, 32, kernel_size=1, padding=0, dilation=1, bias=False)
        # self.c2 = nn.Conv2d(256, 64, kernel_size=1, padding=0, dilation=1, bias=False)
        # self.c3 = nn.Conv2d(512, 128, kernel_size=1, padding=0, dilation=1, bias=False)
        # self.c4 = nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False)


        # self.c1 = nn.Sequential(
        #     nn.Conv2d(128, 32, kernel_size=1, padding=0, dilation=1, bias=False),
        #     # conv3x3(32, 32),
        #     InPlaceABNSync(32) 
        # )   
        self.c1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, padding=0, dilation=1, bias=False),
            # ASPPModule(128,16,32),
            # InPlaceABNSync(32),
            # conv3x3(32, 32),
            InPlaceABNSync(32)
        )
        # self.c1 = nn.Sequential(
        #     conv3x3(128, 128, stride=2),
        #     InPlaceABNSync(128),
        #     conv3x3(128, 128, stride=2),
        #     InPlaceABNSync(128)
        # )
        self.c2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, padding=0, dilation=1, bias=False),
            # ASPPModule(256,32,64),
            # InPlaceABNSync(64),
            # conv3x3(64, 64),
            InPlaceABNSync(64)
        )
        # self.c2 = nn.Sequential(
            # nn.Conv2d(256, 128, kernel_size=1, padding=0, dilation=1, bias=False),
        #     # # conv3x3(64, 64),
        #     # InPlaceABNSync(64)
        #     conv3x3(256, 128, stride=2),
        #     InPlaceABNSync(128)
        # )
        self.c3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, padding=0, dilation=1, bias=False),
            # ASPPModule(512,64,128),
            # InPlaceABNSync(128),
            # conv3x3(128, 128),
            InPlaceABNSync(128)
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            # ASPPModule(1024,128,256),
            # InPlaceABNSync(256),
            # conv3x3(256, 256),
            InPlaceABNSync(256)
        )
            
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 32)
        # self.up4 = up(64, 64)
        # self.outc = nn.Conv2d(64, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

        
        # self.up1 = up_ASPP(in_ch=512, inner_features=64 ,out_ch=128)
        # self.up2 = up_ASPP(in_ch=256, inner_features=32 ,out_ch=64)
        # self.up3 = up_ASPP(in_ch=128, inner_features=16 ,out_ch=32)
        # self.up4 = up(64, 64)

        #CHANGE PRED
        # self.layer5_reduce = nn.Sequential(
        #     nn.Conv2d(256, 32, kernel_size=1, padding=0, dilation=1, bias=True)
        # ) 
        self.up_HD5 = up_HD(256,32)
        self.up_HD4 = up_HD(128,32)
        self.up_HD3 = up_HD(64,32)
        self.up_HD2 = up_HD(32,32)

        # self.layer4_reduce = nn.Sequential(
        #     nn.Conv2d(128, 32, kernel_size=1, padding=0, dilation=1, bias=True)
        # ) 
        # self.layer3_reduce = nn.Sequential(
        #     nn.Conv2d(64, 32, kernel_size=1, padding=0, dilation=1, bias=True)
        # ) 
        # self.layer2_pred = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=1, padding=0, dilation=1, bias=True)
        # ) 
        # self.general_pred = nn.Sequential(
        #     conv3x3(32,32),
        #     InPlaceABNSync(32),
        #     nn.Conv2d(32, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        # ) 
        self.layer5_pred3x3 = nn.Sequential(
            conv3x3(32,32),
            InPlaceABNSync(32)
        )
        self.aux_pred = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(16),
            conv3x3(16,16,bias = True),
            InPlaceABNSync(16),
            nn.Conv2d(16, num_classes, kernel_size=1, padding=0, dilation=1, bias=True) 
        )
        
        #ORIGINAL
        # self.layer5_pred = nn.Sequential(
        #     nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        # ) 
        # self.layer4_pred = nn.Sequential(
        #     nn.Conv2d(128, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        # ) 
        # self.layer3_pred = nn.Sequential(
        #     nn.Conv2d(64, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        # ) 
        # self.layer2_pred = nn.Sequential(
        #     nn.Conv2d(32, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        # ) 

        # self.layer5_cat = nn.Sequential(
        #     nn.Conv2d(256, 32, kernel_size=1),
        #     InPlaceABNSync(32),
        #     conv3x3(out_ch,out_ch),
        #     InPlaceABNSync(out_ch)

        # )
        # self.layer4_cat = catcat(128,32)
        # self.layer3_cat = catcat(64,32)
        # self.layer2_cat = catcat(32,32)
        #USELESS when size layer1 = size layer2
        # self.layer1_cat = catcat(32,32)
        self.double3x3 = nn.Sequential(
            # nn.Conv2d(160, 64, 3, padding=1),
            conv3x3(160,64),
            InPlaceABNSync(64),
            # nn.Conv2d(64, 64, 3, padding=1),
            conv3x3(64,64),
            InPlaceABNSync(64)
        )
        #self.reduce = self.outc =  nn.Conv2d(64, 16, kernel_size=1, padding=0, dilation=1, bias=False)
        self.reduce =  nn.Conv2d(64, 16, kernel_size=1, padding=0, dilation=1, bias=False)
        self.outc =  nn.Sequential(
            conv3x3(16,16),
            InPlaceABNSync(16),
            nn.Conv2d(16, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )


    def forward(self, x):
        _, _,h_gt,w_gt = x.size()
        #conv1
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.relu2(self.bn2(self.conv2(x1)))
        x1 = self.relu3(self.bn3(self.conv3(x1)))
        _, _,h,w = x1.size()
        
        # conv2
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)
        #conv3
        x3 = self.layer2(x2)
        #conv4
        x4 = self.layer3(x3)
        #conv5
        x5 = self.layer4(x4)

        aspp = self.layer5(x5)
        x = self.stabilization(aspp)

        prep_5 = self.up_HD5(x,x1)
        # parsing = self.layer5_reduce(x)
        # parsing = F.interpolate(parsing , size=(h, w), mode='bilinear', align_corners=True)
        # parsing = self.layer5_pred3x3(parsing)
        # final = parsing
        parsing = F.interpolate(prep_5, size=(h_gt, w_gt), mode='bilinear', align_corners=True)
        parsing = self.aux_pred(parsing)
        
        

        x1 = self.c1(x1)
        x2 = self.c2(x2)
        # x2 = torch.cat([x1,x2], dim=1)
        x3 = self.c3(x3)
        x4 = self.c4(x4)

        x = self.up1(x, x4)
        prep_4 = self.up_HD4(x, x1)
        
        parsing1 = F.interpolate(prep_4, size=(h_gt, w_gt), mode='bilinear', align_corners=True)
        parsing1 = self.aux_pred(parsing1)
        
        final = torch.cat([prep_4, prep_5], dim=1)
        # final = self.layer4_cat(x, prep_5)
        # parsing1 = self.layer4_reduce(x)
        # parsing1 = F.interpolate(parsing1 , size=(h, w), mode='bilinear', align_corners=True)
        # parsing1 = self.general_pred(parsing1)

        x = self.up2(x, x3)
        prep_3 = self.up_HD3(x, x1)
        parsing2 = F.interpolate(prep_3, size=(h_gt, w_gt), mode='bilinear', align_corners=True)
        parsing2 = self.aux_pred(parsing2)
        
        final = torch.cat([prep_3, final], dim=1)

        # final = self.layer3_cat(x, final)
        # parsing2 = self.layer3_reduce(x)
        # parsing2 = F.interpolate(parsing2 , size=(h, w), mode='bilinear', align_corners=True)
        # parsing2 = self.general_pred(parsing2)

        x = self.up3(x, x2)
        prep_2 = self.up_HD2(x, x1)
        parsing3 = F.interpolate(prep_2, size=(h_gt, w_gt), mode='bilinear', align_corners=True)
        parsing3 = self.aux_pred(parsing3)
        final = torch.cat([prep_2, final], dim=1)

        # final = self.layer2_cat(x, final)
        # # parsing3 = self.layer2_reduce(x)
        # parsing3 = F.interpolate(x , size=(h, w), mode='bilinear', align_corners=True)
        # parsing3 = self.general_pred(parsing3)

        # final = self.layer1_cat(x1, final)
        final = torch.cat([final, x1], dim=1)

        x = self.double3x3(final)
        x = self.reduce(x)
        x = F.interpolate(x, size=(h_gt, w_gt), mode='bilinear', align_corners=True)
        x = self.outc(x)
        

        return [[parsing,parsing1,parsing2,parsing3,x]]


def Res_Deeplab(num_classes=21):
    #resnet101
    
    orig_resnet = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    # dilated_resnet = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)
    arch_net = Uresnet(orig_resnet, num_classes)
    #resnet50
    # model = ResNet(Bottleneck,[3, 4, 6, 3], num_classes)
    
    return arch_net

