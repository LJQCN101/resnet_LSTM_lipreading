# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import torch, os, torch.optim as optim
from torch.autograd import Variable
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
from xinshuo_miscellaneous import print_log
from xinshuo_io import fileparts

class Validator():
    def __init__(self, options):
        self.batchsize = options["input"]["batchsize"]
        self.validationdataset = LipreadingDataset(options["general"]["dataset"], "val", False)
        self.validationdataloader = DataLoader(self.validationdataset, batch_size=self.batchsize, 
            shuffle=False, num_workers=options["input"]["numworkers"], drop_last=True)
        self.usecudnn = options["general"]["usecudnn"]
        self.statsfrequency = options["training"]["statsfrequency"]
        self.gpuid = options["general"]["gpuid"]
        self.log_file = options["general"]["logfile"]
        self.savedir = options["general"]["modelsavedir"]
        self.num_batches = int(len(self.validationdataset) / self.batchsize)
        self.num_samples = int(len(self.validationdataset))
        self.num_frames = options["general"]["num_frames"]
        print_log('loaded validation dataset with %d data' % len(self.validationdataset), log=self.log_file)

    def epoch(self, model, epoch):
        print_log("Starting validation...", log=self.log_file)
        criterion = model.loss()
        validator_function = model.validator_function()

        sum_loss_so_far, corrects_so_far, sum_samples_so_far = 0., 0., 0.
        for i_batch, (sample_batched, filename_batch) in enumerate(self.validationdataloader):
            with torch.no_grad():
                inputs = Variable(sample_batched['temporalvolume'])
                labels = sample_batched['label']
                if(self.usecudnn):
                    inputs = inputs.cuda(self.gpuid)
                    labels = labels.cuda(self.gpuid)        # num_batch x 1

                outputs = model(inputs)                      # num_batch x 500 for temp-conv         num_batch x 29 x 500               
                loss = criterion(outputs, labels.squeeze(1))
                
                # ave_loss_per_batch = loss.item() / float(self.num_frames)
                ave_loss_per_batch = loss.item() 
                sum_loss_so_far += ave_loss_per_batch * inputs.size(0)
                corrects_per_batch, predict_index_list = validator_function(outputs, labels)
                corrects_so_far += corrects_per_batch
                sum_samples_so_far += self.batchsize

                # for batch_index in range(self.batchsize):
                    # filename_tmp = filename_batch[batch_index]
                    # _, filename_tmp, _ = fileparts(filename_tmp)
                    # filename_tmp = filename_tmp.split('_')[0]
                    # prediction_tmp = self.validationdataset.label_list[predict_index_list[batch_index]]
                    # print_log('Evaluation: val set, batch index %d/%d, filename: %s, prediction: %s' % (batch_index+1, self.batchsize, filename_tmp, prediction_tmp), log=self.log_file)


                print_log('val, Epoch: %d, batch %d/%d (%.f%%), Loss: %.4f, Accu: %.4f' % (epoch, 
                    sum_samples_so_far, self.num_samples, 100. * i_batch / (self.num_batches - 1), 
                    ave_loss_per_batch, corrects_per_batch / float(self.batchsize)), log=self.log_file)
                    # , self.batchsize*(i_batch+1)), 
                

        ave_loss_per_epoch = sum_loss_so_far / sum_samples_so_far
        ave_accu_per_epoch = corrects_so_far / sum_samples_so_far           # debug: to test the number is the same
        print_log('val, Epoch: {}, Average Loss: {:.4f}, Average Accuracy: {:.4f}'.format(epoch, ave_loss_per_epoch, ave_accu_per_epoch)+'\n', log=self.logfile)

        # accuracy = correct_so_far / len(self.validationdataset)
        # accu_savepath = os.path.join(self.savedir, 'accuracy_epoch%03d.txt' % epoch)
        # print_log('saving the accuracy file to %s' % accu_savepath, log=self.log_file)
        # with open(accu_savepath, "a") as outputfile:
        #     outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}".format(correct_so_far, len(self.validationdataset), accuracy))