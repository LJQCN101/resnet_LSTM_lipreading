# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

from torch.autograd import Variable
import torch, torch.optim as optim, os, math, time
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
from xinshuo_miscellaneous import print_log, convert_secs2time
from xinshuo_io import fileparts

class Trainer():
    def __init__(self, args):
        augment = True
        shuffle = True
        self.channel = args.channel
        self.batchsize = args.batch_size
        self.trainingdataset = LipreadingDataset(args.dataset, "train", augment=augment, channel=self.channel)
        self.trainingdataloader = DataLoader(self.trainingdataset, batch_size=self.batchsize, shuffle=shuffle, num_workers=args.workers, drop_last=True)
        # self.usecudnn = options["general"]["usecudnn"]
        self.statsfrequency = args.statsfrequency
        # self.gpuid = options["general"]["gpuid"]
        self.learningrate = args.lr
        self.weightdecay = args.weight_decay
        self.momentum = args.momentum
        self.log_file = args.logfile
        self.modelsavedir = args.save_dir
        # _, self.savename, _ = fileparts(self.modelsavedir)
        self.num_batches = int(len(self.trainingdataset) / self.batchsize)
        self.num_samples = int(len(self.trainingdataset))
        self.num_frames = args.num_frames
        self.modelname = args.modelname
        print_log('loaded training dataset with %d data' % len(self.trainingdataset), log=self.log_file)
        if augment: print_log('using augmentation', log=self.log_file)
        else: print_log('no data augmentation', log=self.log_file)

    def learningRate(self, epoch):
        decay = math.floor(epoch / 5)        
        return self.learningrate * pow(0.5, decay)

    def epoch(self, model, epoch):
        #set up the loss function.
        criterion = model.loss().cuda()
        optimizer = optim.SGD(model.parameters(), lr=self.learningRate(epoch), momentum=self.momentum, weight_decay=self.weightdecay)
        validator_function = model.validator_function()

        print_log('\n\nlearning args: lr, momentum and weight decay are', log=self.log_file)
        for param_group in optimizer.param_groups:
            print_log(param_group['lr'], log=self.log_file)
            print_log(param_group['momentum'], log=self.log_file)
            print_log(param_group['weight_decay'], log=self.log_file)
        # zxc

        #transfer the model to the GPU.
        # if self.usecudnn: criterion = criterion.cuda()
        # startTime = datetime.now()
        print_log("\n\nStarting training...", log=self.log_file)
        sum_loss_so_far, corrects_so_far, sum_samples_so_far = 0., 0., 0.
        for i_batch, (sample_batched, _) in enumerate(self.trainingdataloader):
            optimizer.zero_grad()
            inputs = Variable(sample_batched['temporalvolume']).cuda()
            labels = Variable(sample_batched['label']).cuda()
            # if(self.usecudnn):
                # inputs = inputs.cuda()
                # labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze(1))
            loss.backward()
            optimizer.step()

            # ave_loss_per_batch = loss.item() / float(self.num_frames)           # TODO only true for lstm model
            ave_loss_per_batch = loss.item() / 7.           # TODO only true for lstm model
            # ave_loss_per_batch = loss.item()

            sum_loss_so_far += ave_loss_per_batch * inputs.size(0)
            corrects_per_batch, predict_index_list = validator_function(outputs, labels)
            corrects_so_far += corrects_per_batch
            sum_samples_so_far += self.batchsize

            if i_batch == 0: since = time.time()
            elif i_batch % self.statsfrequency == 0 or (i_batch == self.num_batches - 1):
                print_log('%s, train, Epoch: %d, %d/%d (%.f%%), Loss: %.4f, Accu: %.4f, EP: %s, ETA: %s' % (self.modelname, epoch, 
                    sum_samples_so_far, self.num_samples, 100. * i_batch / (self.num_batches - 1), 
                    ave_loss_per_batch, corrects_per_batch / float(self.batchsize), 
                    convert_secs2time(time.time()-since), 
                    convert_secs2time((time.time()-since)*(self.num_batches - 1) / i_batch - (time.time()-since))), log=self.log_file)


        ave_loss_per_epoch = sum_loss_so_far / sum_samples_so_far
        ave_accu_per_epoch = corrects_so_far / sum_samples_so_far           # debug: to test the number is the same
        print_log('train, Epoch: {}, Average Loss: {:.4f}, Average Accuracy: {:.4f}'.format(epoch, ave_loss_per_epoch, ave_accu_per_epoch)+'\n', log=self.log_file)


        print_log("Epoch completed, saving state...", log=self.log_file)
        torch.save(model.state_dict(), os.path.join(self.modelsavedir, '%s_epoch%03d.pt' % (self.modelname, epoch)))

        return ave_loss_per_epoch, ave_accu_per_epoch