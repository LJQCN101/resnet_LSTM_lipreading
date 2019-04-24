# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import re, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from .ConvFrontend import ConvFrontend
from .ResNetBBC import ResNetBBC
from .LSTMBackend import LSTMBackend
from .ConvBackend import ConvBackend
from xinshuo_miscellaneous import print_log

class C3D_CONV_BLSTM(nn.Module):
    def __init__(self, args, input_dims=256, hidden_dims=256, num_lstm=2):
        super(C3D_CONV_BLSTM, self).__init__()
        channel = args.channel
        print_log('channel is %d' % channel, log=args.logfile)

        self.frontend = ConvFrontend(channel)
        self.resnet = ResNetBBC(input_dims, args.batch_size)
        self.lstm = LSTMBackend(input_dims=input_dims, hidden_dims=hidden_dims, 
            num_lstm=num_lstm, num_classes=args.num_classes, num_frames=args.num_frames)

        def freeze(m): m.requires_grad = False

        # self.frontend.apply(freeze)
        # self.resnet.apply(freeze)

        #function to initialize the weights and biases of each module. Matches the
        #classname with a regular expression to determine the type of the module, then
        #initializes the weights for it.
        def weights_init(m):
            classname = m.__class__.__name__
            if re.search("Conv[123]d", classname):
                m.weight.data.normal_(0.0, 0.02)
            elif re.search("BatchNorm[123]d", classname):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)
            elif re.search("Linear", classname):
                m.bias.data.fill_(0)

        #Apply weight initialization to every module in the model.
        self.apply(weights_init)

    def forward(self, input):
        # inputs:       36 x 1 x 29 x 112 x 112

        out = self.frontend(input)          # 36 x 64 x 29 x 28 x 28
        out = self.resnet(out)              # 36 x 29 x 256
        out = self.lstm(out)                # 36 x 29 x 500

        return out

    def loss(self):

        return self.lstm.loss

    def validator_function(self):

        return self.lstm.validator