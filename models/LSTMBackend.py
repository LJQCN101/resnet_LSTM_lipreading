import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class NLLSequenceLoss(nn.Module):
    """
    Custom loss function.
    Returns a loss that is the sum of all losses at each time step.
    """
    def __init__(self, num_frames):
        super(NLLSequenceLoss, self).__init__()
        self.criterion = nn.NLLLoss()
        self.num_frames = num_frames

    def forward(self, input, target):
        loss = 0.0
        transposed = input.transpose(0, 1).contiguous()
        for i in range(0, self.num_frames):
            loss += self.criterion(transposed[i], target)

        # loss /= self.num_frames
        return loss

def _validate(modelOutput, labels):
    # modelOutput               # num_batch x 29 x 500
    # labels                    # num_batch x 1
    averageEnergies = torch.sum(modelOutput.data, 1)            # num_batch x 500
    maxvalues, maxindices = torch.max(averageEnergies, 1)
    count = 0
    for i in range(0, labels.squeeze(1).size(0)):
        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1

    return count, maxindices

class LSTMBackend(nn.Module):
    def __init__(self, input_dims=256, hidden_dims=256, num_lstm=2, num_classes=500, num_frames=29):
        super(LSTMBackend, self).__init__()
        self.Module1 = nn.LSTM(input_size=input_dims, hidden_size=hidden_dims,
            num_layers=num_lstm, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_dims * 2, num_classes)
        self.softmax = nn.LogSoftmax(dim=2)
        self.loss = NLLSequenceLoss(num_frames)
        self.validator = _validate

    def forward(self, input):
        # input: batch_size x 29 x 256
        temporalDim = 1
        lstmOutput, _ = self.Module1(input)     # batch_size x 29 x 512
        output = self.fc(lstmOutput)            # batch_size x 29 x 500
        output = self.softmax(output)           # batch_size x 29 x 500
        return output
