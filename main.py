# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

from __future__ import print_function
import torch, toml, os
from models import LipRead, I3D
from training import Trainer
from validation import Validator
from utils import plot_loss, plot_accu
from xinshuo_miscellaneous import get_timestring, print_log, is_path_exists
from xinshuo_io import mkdir_if_missing

print("Loading options...")
with open('options.toml', 'r') as optionsFile: options = toml.loads(optionsFile.read())
if options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]: torch.backends.cudnn.benchmark = True 
options["general"]["modelsavedir"] = os.path.join(options["general"]["modelsavedir"], 'trained_model_' + get_timestring()); mkdir_if_missing(options["general"]["modelsavedir"])
options["general"]["logfile"] = open(os.path.join(options["general"]["modelsavedir"], 'log.txt'), 'w')
print_log(options, log=options["general"]["logfile"])
print_log('\n\nsaving to %s' % options["general"]["modelsavedir"], log=options["general"]["logfile"])

print_log('creating the model\n\n', log=options["general"]["logfile"])
model = LipRead(options)
# model = I3D()
print_log(model, log=options["general"]["logfile"])


if options["general"]["loadpretrainedmodel"]: 
	print_log('\n\nloading model', log=options["general"]["logfile"])
	print_log('loading the pretrained model at %s' % options["general"]["pretrainedmodelpath"], log=options["general"]["logfile"])
	assert is_path_exists(options["general"]["pretrainedmodelpath"]), 'the pretrained model does not exists'
	model.load_state_dict(torch.load(options["general"]["pretrainedmodelpath"]))		#Create the model.
else: print_log('\n\ntraining from scratch', log=options["general"]["logfile"])
if options["general"]["usecudnn"]: model = model.cuda(options["general"]["gpuid"])		#Move the model to the GPU.



print_log('loading data', log=options["general"]["logfile"])
if options["training"]["train"]: trainer = Trainer(options)
if options["validation"]["validate"]: validator = Validator(options)
	# validator.epoch(model, epoch=0)

loss_history_train, loss_history_val = [], []
accu_history_train, accu_history_val = [], []
if options["training"]["train"]:
	for epoch in range(options["training"]["startepoch"], options["training"]["endepoch"]):
		loss_train, accu_train = trainer.epoch(model, epoch)
		if options["validation"]["validate"]: 
			loss_val, accu_val = validator.epoch(model, epoch)

		loss_history_train.append([loss_train]); loss_history_val.append([loss_val])
		accu_history_train.append([accu_train]); accu_history_val.append([accu_val])
		plot_loss(loss_history_train, loss_history_val, save_dir=options["general"]["modelsavedir"])
		plot_accu(accu_history_train, accu_history_val, save_dir=options["general"]["modelsavedir"])

	# if options["testing"]["test"]:
	# 	tester = Tester(options)
	# 	tester.epoch(model)

options["general"]["logfile"].close()