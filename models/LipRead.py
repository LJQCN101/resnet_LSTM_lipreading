# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import re, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from .ConvFrontend import ConvFrontend
from .ResNetBBC import ResNetBBC
from .LSTMBackend import LSTMBackend
from .ConvBackend import ConvBackend
from xinshuo_miscellaneous import print_log

class LipRead(nn.Module):
    def __init__(self, options):
        super(LipRead, self).__init__()
        self.frontend = ConvFrontend()
        self.resnet = ResNetBBC(options)
        self.backend = ConvBackend(options)
        self.lstm = LSTMBackend(options)
        self.type = options["model"]["type"]

        def freeze(m):
            m.requires_grad=False

        if (options["model"]["type"] == "LSTM-init"):
            print_log('with freezing frontend', log=options["general"]["logfile"])
            self.frontend.apply(freeze)
            self.resnet.apply(freeze)
        else: print_log('no freezing', log=options["general"]["logfile"])


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

        if self.type == "temp-conv":
            out = self.frontend(input)
            out = self.resnet(out)
            out = self.backend(out)

        if self.type == "LSTM" or self.type == "LSTM-init":
            out = self.frontend(input)          # 36 x 64 x 29 x 28 x 28
            # print(out.size())
            out = self.resnet(out)              # 36 x 29 x 256
            # print(out.size())
            out = self.lstm(out)                # 36 x 29 x 500
            # print(out.size())
            # zxc

        return out

    def loss(self):
        if(self.type == "temp-conv"):
            return self.backend.loss

        if(self.type == "LSTM" or self.type == "LSTM-init"):
            return self.lstm.loss

    def validator_function(self):
        if(self.type == "temp-conv"):
            return self.backend.validator

        if(self.type == "LSTM" or self.type == "LSTM-init"):
            return self.lstm.validator