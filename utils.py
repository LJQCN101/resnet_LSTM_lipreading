# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import torch, matplotlib.pyplot as plt, torch.utils.data, os, numpy as np, math
from xinshuo_miscellaneous import print_log
from collections import OrderedDict
from torch.optim.optimizer import Optimizer

def reload_model(model, logfile, path=""):
    if not bool(path):
        print_log('train from scratch', log=logfile)
        return model
    else:
        print_log('loadding the model', log=logfile)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path)
        all_weights, finetuned_layer, random_initial_layer = [], [], []
        for key, value in model_dict.items():
            if key in pretrained_dict:
                all_weights.append((key, pretrained_dict[key]))
                finetuned_layer.append(key)
            else:
                all_weights.append((key, value))
                random_initial_layer.append(key)
    
        print_log('==> finetuned layers : {}\n\n'.format(finetuned_layer), log=logfile)
        print_log('==> random initialized layers : {}\n\n'.format(random_initial_layer), log=logfile)
        all_weights = OrderedDict(all_weights)
        model.load_state_dict(all_weights)
        print_log('*** model has been successfully loaded from %s! ***' % path, log=logfile)
        return model

def plot_loss(loss, val_loss, save_dir=None):
    loss = np.array(loss)
    val_loss = np.array(val_loss)
    plt.figure("loss")
    plt.gcf().clear()
    plt.plot(loss[:, 0], label='train')
    plt.plot(val_loss[:, 0], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    save_path = os.path.join(save_dir, "loss.png")
    plt.savefig(save_path)

def plot_accu(accu, val_accu, save_dir=None):
    accu = np.array(accu)
    val_accu = np.array(val_accu)
    plt.figure("accu")
    plt.gcf().clear()
    plt.plot(accu[:, 0], label='train')
    plt.plot(val_accu[:, 0], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    save_path = os.path.join(save_dir, "accuracy.png")
    plt.savefig(save_path)

# def showLR(optimizer):
#     lr = []
#     for param_group in optimizer.param_groups: lr += [param_group['lr']]
#     return lr

# class AdjustLR(object):
#     def __init__(self, optimizer, init_lr, sleep_epochs=5, half=5, verbose=0):
#         super(AdjustLR, self).__init__()
#         self.optimizer = optimizer
#         self.sleep_epochs = sleep_epochs
#         self.half = half
#         self.init_lr = init_lr
#         self.verbose = verbose

#     def step(self, epoch):
#         if epoch >= self.sleep_epochs:
#             for idx, param_group in enumerate(self.optimizer.param_groups):
#                 new_lr = self.init_lr[idx] * math.pow(0.5, (epoch-self.sleep_epochs+1)/float(self.half))
#                 param_group['lr'] = new_lr
#             if self.verbose:
#                 print('>>> reduce learning rate <<<')
