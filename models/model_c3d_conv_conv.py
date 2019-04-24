# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import re, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from .ConvFrontend import ConvFrontend
from .ResNetBBC import ResNetBBC
from .LSTMBackend import LSTMBackend
from .ConvBackend import ConvBackend
from xinshuo_miscellaneous import print_log

class C3D_CONV_CONV(nn.Module):
    def __init__(self, args):
        super(C3D_CONV_CONV, self).__init__()
        channel = args.channel
        print_log('channel is %d' % channel, log=args.logfile)

        self.frontend = ConvFrontend(channel)
        self.resnet = ResNetBBC(input_dims, args.batch_size)
        self.backend = ConvBackend(args.num_classes)


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

        out = self.frontend(input)
        out = self.resnet(out)
        out = self.backend(out)

        return out

    def loss(self):
        
        return self.backend.loss

    def validator_function(self):

        return self.backend.validator
