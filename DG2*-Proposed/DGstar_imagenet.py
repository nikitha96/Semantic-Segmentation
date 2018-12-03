
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

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
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
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
        
        return F.relu(output+input)  

class conv_module (nn.Module):
    def __init__(self, chann, dropprob, dilated, groups):        
        super().__init__()

        #print("groups in conv_groups", groups)

        self.groups = groups

        self.conv3x3_1 = nn.Conv2d(chann, chann, kernel_size=3, stride=(1, 1), padding=(1, 1), groups=chann)

        self.bn1_1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv1L_1 = nn.Conv2d(chann, chann, kernel_size=1, bias=True, groups=groups)

        self.bn2_1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x3_2 = nn.Conv2d(chann, chann, kernel_size=3, stride=(1,1), padding=(1*dilated,1*dilated), dilation=(dilated,dilated), bias=True, groups=chann)

        self.bn1_2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv1L_2 = nn.Conv2d(chann, chann, kernel_size=1, bias=True, groups=groups)

        self.bn2_2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x3_1(input)
        #output = self.conv3x1_1(input)
        output = self.bn1_1(output)
        output = F.relu(output)

        output = self.conv1L_1(output)
        output = self.bn2_1(output)
        output = F.relu(output)
        output = channel_shuffle(output, self.groups)
        output = self.conv3x3_2(output)
        output = self.bn1_2(output)
        output = F.relu(output)
        output = self.conv1L_2(output)
        output = self.bn2_2(output)
        
        if (self.dropout.p != 0):
            #output += input
            output = self.dropout(output)
        
        return F.relu(output + input)    #+input = identity (residual connection)
        #return F.relu(output)

class Encoder(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
            self.layers.append(conv_module(64, 0.1, 1, groups))  
        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(conv_module(128, 0.1, 2, groups))
            self.layers.append(conv_module(128, 0.1, 4, groups))
            self.layers.append(conv_module(128, 0.1, 8, groups))
            self.layers.append(conv_module(128, 0.1, 16, groups))


    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class Features(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.encoder = Encoder(groups)
        self.extralayer1 = nn.MaxPool2d(2, stride=2)
        self.extralayer2 = nn.AvgPool2d(14,1,0)

    def forward(self, input):
        output = self.encoder(input)
        output = self.extralayer1(output)
        output = self.extralayer2(output)
        return output

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(128, num_classes)

    def forward(self, input):
        output = input.view(input.size(0), 128) #first is batch_size
        output = self.linear(output)
        return output

class Model_Net(nn.Module):
    def __init__(self, num_classes, groups):  #use encoder to pass pretrained encoder
        super().__init__()

        self.features = Features(groups)
        self.classifier = Classifier(num_classes)

    def forward(self, input):
        output = self.features(input)
        output = self.classifier(output)
        return output
