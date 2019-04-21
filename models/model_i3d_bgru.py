# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import math, os, numpy as np, torch, torch.nn as nn
from torch.nn import ReplicationPad3d
from torch.autograd import Variable
from .i3d_utils import Unit3Dpy, Mixed, MaxPool3dTFPadding

def _validate(modelOutput, labels):
    maxvalues, maxindices = torch.max(modelOutput.data, 1)
    count = 0
    for i in range(0, labels.squeeze(1).size(0)):
        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1

    return count, maxindices

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        out, _ = self.gru(x, h0)                # batch_size x 7 x 1024
        out = self.fc(out)                      # batch_size x 7 x 500
        out = torch.mean(out, 1)                # batch_size x 500
        return out

class I3D_BGRU(nn.Module):
    def __init__(self, inputDim=528, hiddenDim=1024, nClasses=500):
        super(I3D_BGRU, self).__init__()
        in_channels = 1
        self.loss_history_train, self.loss_history_val = [], []
        
        # 1st conv-pool
        self.conv3d_1a_7x7 = Unit3Dpy(out_channels=64, in_channels=in_channels, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding='SAME')
        # self.conv3d_1a = Unit3Dpy(out_channels=64, in_channels=1, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding='SAME')
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        
        # 2nd conv-conv-pool
        self.conv3d_2b_1x1 = Unit3Dpy(out_channels=64, in_channels=64, kernel_size=(1, 1, 1), padding='SAME')
        self.conv3d_2c_3x3 = Unit3Dpy(out_channels=192, in_channels=64, kernel_size=(3, 3, 3), padding='SAME')
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])
        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        # self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])
        # self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

        # Mixed 5
        # self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        # self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = torch.nn.AvgPool3d((1, 7, 7), (1, 1, 1))
        # self.dropout = torch.nn.Dropout(dropout_prob)

        # self.conv3d_final = Unit3Dpy(in_channels=832, out_channels=self.num_classes, kernel_size=(1, 1, 1), activation=None, use_bias=True, use_bn=False)
        # self.conv3d_final = nn.Conv3d(832, self.num_classes, kernel_size=(1, 1, 1), bias=True)
        # self.softmax = torch.nn.Softmax(1)

        # self._initialize_weights()

        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.nLayers = 2
        self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses)

        self.validator = _validate
        self._initialize_weights()

    def forward(self, x):
        # inputs:       batch_size x 3 x 29 x 112 x 112
        
        out = self.conv3d_1a_7x7(x)           # batch_size x 64 x 14 x 56 x 56
        out = self.maxPool3d_2a_3x3(out)        # batch_size x 64 x 14 x 28 x 28
        out = self.conv3d_2b_1x1(out)           # batch_size x 64 x 14 x 28 x 28
        out = self.conv3d_2c_3x3(out)           # batch_size x 192 x 14 x 28 x 28
        out = self.maxPool3d_3a_3x3(out)        # batch_size x 192 x 14 x 14 x 14
        out = self.mixed_3b(out)                # batch_size x 256 x 14 x 14 x 14     
        out = self.mixed_3c(out)                # batch_size x 480 x 14 x 14 x 14
        out = self.maxPool3d_4a_3x3(out)        # batch_size x 480 x 7 x 7 x 7
        out = self.mixed_4b(out)                # batch_size x 512 x 7 x 7 x 7
        out = self.mixed_4c(out)                # batch_size x 512 x 7 x 7 x 7
        out = self.mixed_4d(out)                # batch_size x 512 x 7 x 7 x 7
        out = self.mixed_4e(out)                # batch_size x 528 x 7 x 7 x 7
        out = self.avg_pool(out)                # batch_size x 528 x 7 x 1 x 1
        
        out = out.squeeze(3)                    # batch_size x 528 x 7 x 1
        out = out.squeeze(3)                    # batch_size x 528 x 7

        out = out.transpose(1, 2)               # batch_size x 7 x 528
        out = self.gru(out)                     # batch_size x 500   
        return out

    def loss(self):
        return nn.CrossEntropyLoss()

    def validator_function(self):
        return self.validator

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()