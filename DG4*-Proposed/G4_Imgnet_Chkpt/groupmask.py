

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import math

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
    
    
class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)

def group_mask(groups, alph_value, inC, outC, kernel_size):

    mask = torch.ones((inC,outC,kernel_size,kernel_size))
    g_size = inC // groups

    for g_x in range(groups):
        for g_y in range(groups):
            for o_x in range(g_size):
                for o_y in range(g_size):
                    if g_x == g_y:
                        mask[g_x*g_size + o_x, g_y*g_size + o_y] = 1.0
                    else:
                        mask[g_x*g_size + o_x, g_y*g_size + o_y] = alph_value

    return Variable(mask, requires_grad=False)



class MaskConv2d(nn.Module):
    def __init__(self, inC, outC, kernel_size, groups=1):
        super(MaskConv2d, self).__init__()

        self.groups = groups
        #print("groups in maskconv:", groups)
        self.kernel_size = kernel_size
        self.inC = inC
        self.outC = outC
        self.weight = torch.nn.Parameter(data=torch.Tensor(inC, outC, kernel_size, kernel_size), requires_grad=True)
        self.bias = torch.nn.Parameter(data=torch.Tensor(outC))
        stdev = 1.0/math.sqrt(inC*kernel_size*kernel_size)
        self.weight.data.uniform_(-stdev, stdev)
        self.bias.data.uniform_(-stdev, stdev)
        #nn.init.kaiming_normal(self.weight.data, mode='fan_out')

    def forward(self, inp, alpha):
        mask = group_mask(self.groups, alpha, self.inC, self.outC, self.kernel_size)
        masked_wt = self.weight.mul(mask.cuda())
        return torch.nn.functional.conv2d(inp, masked_wt, bias=self.bias)



class conv_module (nn.Module):
    def __init__(self, chann, dropprob, dilated, groups=1):        
        super().__init__()

        #self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)
        #self.conv1R_1 = nn.Conv2d(chann, chann, kernel_size=1, bias=True)

        #self.bn1_1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.groups = groups

        #print("groups in conv_module:", groups)

        self.conv3x3_1 = nn.Conv2d(chann, chann, kernel_size=3, stride=(1, 1), padding=(1, 1), groups=chann)

        self.bn1_1 = nn.BatchNorm2d(chann, eps=1e-03)

        #self.conv1L_1 = nn.Conv2d(chann, chann, kernel_size=1, bias=True, groups=8)

        self.conv1L_1 = MaskConv2d(chann, chann, kernel_size=1, groups=groups)

        self.bn2_1 = nn.BatchNorm2d(chann, eps=1e-03)



        #self.conv1R_2 = nn.Conv2d(chann, chann, kernel_size=1, bias=True)

        #self.bn1_2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x3_2 = nn.Conv2d(chann, chann, kernel_size=3, stride=(1,1), padding=(1*dilated,1*dilated), dilation=(dilated,dilated), bias=True, groups=chann)

        # self.conv3x3_2 = nn.Conv2d(chann, chann, kernel_size=3, stride=1, bias=True, groups=chann)

        self.bn1_2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv1L_2 = MaskConv2d(chann, chann, kernel_size=1, groups = groups)

        self.bn2_2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input, alpha = 1.0):

        output = self.conv3x3_1(input)
        #output = self.conv3x1_1(input)
        output = self.bn1_1(output)
        output = F.relu(output)
        output = self.conv1L_1(output, alpha)
        output = self.bn2_1(output)
        #output = channel_shuffle(output, self.groups)
        output = self.conv3x3_2(output)
        output = self.bn1_2(output)
        output = F.relu(output)
        output = self.conv1L_2(output, alpha)
        output = self.bn2_2(output)
        #output = F.relu(output)

        #        output = self.conv3x3_2(output)
        #        output = self.bn1_2(output)
	   # output = F.relu(output)
        #        output = self.conv1L_2(output)
        #        output = self.bn2_2(output)
        #        #output = F.relu(output)


        if (self.dropout.p != 0):
            #output += input
            output = self.dropout(output)
        
        return F.relu(output + input)    #+input = identity (residual connection)
        #return F.relu(output)


class Encoder(nn.Module):
    def __init__(self, num_classes, groups):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        print("groups in enc:", groups)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(conv_module(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(conv_module(128, 0.3, 2, groups=groups))
            self.layers.append(conv_module(128, 0.3, 4, groups=groups))
            self.layers.append(conv_module(128, 0.3, 8, groups=groups))
            self.layers.append(conv_module(128, 0.3, 16, groups=groups))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, alpha, predict=False):
        output = self.initial_block(input)

        print("---alpha in enc:", alpha)

        for layer in self.layers:
            if layer is conv_module:
                output = self.layer(output, alpha)
            else:
                output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

#ERFNet
#MobileNet

class Net (nn.Module):
    def __init__(self, num_classes, groups=1, encoder=None):  #use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes, groups=groups)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False, alpha = 1.0):
        if only_encode:
            return self.encoder.forward(input, alpha, predict=True)
        else:
            output = self.encoder(input, alpha)    #predict=False by default
            return self.decoder.forward(output)

