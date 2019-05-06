# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

from __future__ import print_function
import torch, toml, numpy as np, os, argparse, random, matplotlib; matplotlib.use('Agg')
from models import C3D_CONV_BLSTM, C3D_CONV_CONV, I3D, I3D_BLSTM
from training import Trainer
from validation import Validator
from utils import plot_loss, plot_accu, reload_model
from xinshuo_miscellaneous import get_timestring, print_log, is_path_exists
from xinshuo_miscellaneous.pytorch import prepare_seed
from xinshuo_io import mkdir_if_missing


parser = argparse.ArgumentParser(description='Pytorch Video-only BBC-LRW Example')
parser.add_argument('--path', default='', help='path to model')
parser.add_argument('--modelname', default='', help='temporalConv, backendGRU, finetuneGRU')

parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='initial momentum')
parser.add_argument('--weight_decay', default=5e-5, type=float, help='initial weight decay')

parser.add_argument('--batch_size', default=36, type=int, help='mini-batch size (default: 36)')
parser.add_argument('--num_frames', default=29, type=int, help='mini-batch size (default: 36)')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, help='number of start epoch')
parser.add_argument('--end_epoch', default=20, type=int, help='number of total epochs')
parser.add_argument('--statsfrequency', default=2, help='display interval')
parser.add_argument('--num_classes', default=500, type=int, help='the number of classes')
parser.add_argument('--channel', default=1, type=int, help='the number of input channels')

parser.add_argument('--train', action='store_true', help='training mode')
parser.add_argument('--val', action='store_true', help='validation mode')
parser.add_argument('--test', action='store_true', help='testing mode')
parser.add_argument('--vis', action='store_true', help='visualization mode')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()
torch.backends.cudnn.benchmark = True   
prepare_seed(args.seed)


print("Loading options...")
with open('options.toml', 'r') as optionsFile: options = toml.loads(optionsFile.read())
args.save_dir = os.path.join(options["general"]["modelsavedir"], args.modelname + '_' + get_timestring()); mkdir_if_missing(args.save_dir)
args.dataset = options["general"]["dataset"]
args.logfile = os.path.join(args.save_dir, 'log.txt'); args.logfile = open(args.logfile, 'w')
# print_log(options, args.logfile)
print_log(args, args.logfile)
print_log('\n\nsaving to %s' % args.save_dir, log=args.logfile)

print_log('creating the model\n\n', log=args.logfile)
if args.modelname == 'C3D_CONV_BLSTM': model = C3D_CONV_BLSTM(args, input_dims=256, hidden_dims=256, num_lstm=2)
elif args.modelname == 'C3D_CONV_BLSTM_frontfix': model = C3D_CONV_BLSTM(args, input_dims=256, hidden_dims=256, num_lstm=2)
elif args.modelname == 'C3D_CONV_CONV': model = C3D_CONV_CONV(args, input_dims=256)
elif args.modelname == 'I3D_BLSTM': model = I3D_BLSTM()
elif args.modelname == 'I3D': model = I3D()

print_log(model, log=args.logfile)
model = reload_model(model, args.logfile, args.path)     # reload model
model = model.cuda()								# move the model to the GPU.

if args.modelname == 'C3D_CONV_BLSTM_frontfix':
	print_log('\n\nwith freezing frontend', log=args.logfile)
	for param in model.frontend.parameters(): param.requires_grad = False
	for param in model.resnet.parameters(): param.requires_grad = False
else: print_log('\n\nno freezing', log=args.logfile)

print_log('loading data', log=args.logfile)
if args.train: trainer = Trainer(args)
if args.val: validator = Validator(args)

if args.train:
	loss_history_train, loss_history_val = [], []
	accu_history_train, accu_history_val = [], []
	for epoch in range(args.start_epoch, args.end_epoch):
		loss_train, accu_train = trainer.epoch(model, epoch)
		if args.val: loss_val, accu_val = validator.epoch(model, epoch)

		# plot figure
		loss_history_train.append([loss_train]); loss_history_val.append([loss_val])
		accu_history_train.append([accu_train]); accu_history_val.append([accu_val])
		plot_loss(loss_history_train, loss_history_val, save_dir=args.save_dir)
		plot_accu(accu_history_train, accu_history_val, save_dir=args.save_dir)

if args.test:
	tester = Tester(args)
	tester.epoch(model)
	if args.val: validator.epoch(model, epoch=0)


args.logfile.close()




# weird momentum and learning rate, 30 51 58 62 64
# normal momentum and learning rate, 20 48 56 61 64

# channel 3, 13 48 54 61 63 65 69 70 70 71 72
# channel 1, 30 50 58 61 65 66 70 71 71 72 71 74