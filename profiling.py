from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import time
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import models.lstm as lstmpy
from torchvision import models
import torch as t
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
print(t.version.cuda)
import os, sys
import logging
import argparse

class Profiling(object):
    def __init__(self, model):
        if isinstance(model, torch.nn.Module) is False:
            raise ValueError("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self._parameter_names = {v: k for k, v
                                in model.named_parameters()}
        self._seq_keys = [k for k, v in model.named_parameters()]
        self._backward_seq_keys = []
        self._backward_key_sizes = []
        self._grad_accs = []
        self._handles = {}
        self.hook_done = False
        self._start = time.time()
        self._register_hooks()
        self._is_profiling = False

    def _register_hooks(self):
        for name, p in self.model.named_parameters():
            #p_tmp = p.expand_as(p)
            #grad_acc = p_tmp.grad_fn.next_functions[0][0]
            #grad_acc.register_hook(self._make_hook(name, p))
            #self._grad_accs.append(grad_acc)
            p.register_hook(self._make_hook(name, p))

    def _make_hook(self, name, p):
        def hook(*ignore):
            if not self._is_profiling:
                return
            name = self._parameter_names.get(p)
            if len(self._backward_seq_keys) != len(self._seq_keys):
                self._backward_seq_keys.append(name)
                self._backward_key_sizes.append(p.numel())
            if name not in self._handles:
                self._handles[name] = []
            torch.cuda.synchronize()
            ct = self._timestamp(name)
            #print(self._start, ',', ct, ', diff: ', ct-self._start, name, ', size: ', p.data.numel())
            self._handles[name].append(ct - self._start)
        return hook

    def reset_start(self):
        self._start = time.time()

    def reset(self):
        self._start = time.time()
        self._handles.clear()

    def stop(self):
        self._is_profiling = False

    def start(self):
        self._is_profiling = True
        self._start = time.time()

    def get_backward_seq_keys(self):
        return self._backward_seq_keys

    def get_backward_key_sizes(self):
        return self._backward_key_sizes

    def get_layerwise_times(self):
        num_trials = len(self._handles[self._seq_keys[0]])
        layerwise_times_multipletest = []
        totals = []
        for j in range(num_trials):
            s = 0
            total = 0.0
            layerwise_times = [] # from the last layer to the first layer
            #for i, k in enumerate(self._seq_keys[::-1]):
            for i, k in enumerate(self._backward_seq_keys):
                t = self._handles[k][j]
                #print('name: ', k, ' diff: ', t-s)
                layerwise_times.append(t-s)
                total += (t-s)
                s = total
            layerwise_times_multipletest.append(layerwise_times)
            totals.append(total)
        array = np.array(layerwise_times_multipletest)
        layerwise_times = np.mean(array, axis=0)
        return layerwise_times, np.mean(totals)

    def _timestamp(self, name):
        return time.time()


def benchmark(trainer):
    # Benchmark to achieve the backward time per layer
    p = Profiling(trainer.net)
    # Warmup
    input_shape, output_shape = trainer.get_data_shape()
    warmup = 5 # warmup should be 0 on some GPUs (e.g., P102-100)
    iteration = 50

    for i in range(iteration+warmup):
        data = trainer.data_iter()

        if trainer.dataset == 'an4':
            inputs, labels_cpu, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        else:
            inputs, labels_cpu = data
        if trainer.is_cuda:
            if trainer.dnn == 'lstm' :
                inputs = Variable(inputs.transpose(0, 1).contiguous()).cuda()
                labels = Variable(labels_cpu.transpose(0, 1).contiguous()).cuda()
            else:
                inputs, labels = inputs.cuda(non_blocking=True), labels_cpu.cuda(non_blocking=True)
        else:
            labels = labels_cpu

        if trainer.dnn == 'lstman4':
            out, output_sizes = trainer.net(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
            loss = trainer.criterion(out, labels_cpu, output_sizes, target_sizes)
            torch.cuda.synchronize()
            loss = loss / inputs.size(0)  # average the loss by minibatch
        elif trainer.dnn == 'lstm' :
            hidden = trainer.net.init_hidden()
            hidden = lstmpy.repackage_hidden(hidden)
            #print(inputs.size(), hidden[0].size(), hidden[1].size())
            outputs, hidden = trainer.net(inputs, hidden)
            tt = torch.squeeze(labels.view(-1, trainer.net.batch_size * trainer.net.num_steps))
            loss = trainer.criterion(outputs.view(-1, trainer.net.vocab_size), tt)
            torch.cuda.synchronize()
        else:
            # forward + backward + optimize
            outputs = trainer.net(inputs)
            loss = trainer.criterion(outputs, labels)
            torch.cuda.synchronize()

        if i >= warmup:
            p.start()
        loss.backward()
        if trainer.is_cuda:
            torch.cuda.synchronize()
    layerwise_times, sum_total = p.get_layerwise_times()
    seq_keys = p.get_backward_seq_keys()
    p.stop()
    return seq_keys[::-1], layerwise_times[::-1], p.get_backward_key_sizes()[::-1]


class CommunicationProfiler(object):
    def __init__(self, comm_op, sync_op, sizes=None):
        self.comm_op = comm_op
        self.sync_op = sync_op
        self.sizes = sizes

    def benchmark(self, num_iters=100):
        if self.sizes is None:
            small_sizes = [8*1024*i for i in range(1, 64)] # 1K to 1M
            large_sizes = [] #[1024*1024*i for i in range(8)] # 1M to 512M
            sizes = small_sizes+large_sizes
        else:
            sizes = self.sizes
        warmup = 5
        size = 1024
        tensor = torch.rand(size).float().cuda()
        stime = time.time()
        for i in range(warmup):
            name = 'warmup-%d' % i
            h = self.comm_op(tensor, average=True, name=name)
            self.sync_op(h)
        etime = time.time()
        elapsed_times = []
        for s in sizes:
            tensor = torch.rand(s).float().cuda()
            torch.cuda.synchronize()
            stime = time.time()
            for i in range(num_iters):
                name = 'run-size%d-%d'% (s, i)
                h = self.comm_op(tensor, average=True, name=name)
                self.sync_op(h)
            etime = time.time()
            elapsed_times.append((etime-stime)/num_iters)
        return sizes, elapsed_times


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                        help='use fp16 compression during byteps pushpull')

    parser.add_argument('--same', type=int, default=0,
        help='if using servers with workers on same machines')
    parser.add_argument('--whole-grad', action='store_true', default=False,
        help='set whole grad')

    parser.add_argument('--model', type=str, default='resnet50',
                        help='model to benchmark')
    parser.add_argument('--DMLC-PS', type=str, default='aa',
                        help='model to benchmark')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size')

    parser.add_argument('--num-warmup-batches', type=int, default=10,
                        help='number of warm-up batches that don\'t count towards benchmark')
    parser.add_argument('--num-batches-per-iter', type=int, default=1,
                        help='number of batches per benchmark iteration')
    parser.add_argument('--num-iters', type=int, default=50,
                        help='number of benchmark iterations')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='number of classes')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--profiler', action='store_true', default=False,
                        help='disables profiler')
    parser.add_argument('--partition', type=int, default=None,
                        help='partition size')

    parser.add_argument('--nworkers', type=int, default=1,
                        help='aaa')
    parser.add_argument('--nservers', type=int, default=1,
                        help='aaa')
    parser.add_argument('--worker-id', type=int, default=1,
                        help='aaa')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    strhdlr = logging.StreamHandler()
    logger.addHandler(strhdlr)
    formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
    strhdlr.setFormatter(formatter)

    if args.whole_grad:
        relative_path = './bps_whole_grad'
    else:
        relative_path = './bps_layerwise'
    relative_path += '_single_train_log'

    logfile = os.path.join(relative_path, args.model+'-network'+str(args.DMLC_PS)+'-bs'+str(args.batch_size)+'-iters'+str(args.num_iters)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)

    model = getattr(models, args.model)(num_classes=args.num_classes)
    torch.cuda.set_device(0)
    model.cuda()

    # Set up fake data
    datasets = []
    for _ in range(100):
        data = torch.rand(args.batch_size, 3, 224, 224)
        target = torch.LongTensor(args.batch_size).random_() % 1000
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        datasets.append(data)
    data_index = 0


    # Benchmark to achieve the backward time per layer
    p = Profiling(model)
    # Warmup
    # input_shape, output_shape = trainer.get_data_shape()
    warmup = 5 # warmup should be 0 on some GPUs (e.g., P102-100)
    iteration = args.num_iters
    data_index = 0

    for i in range(iteration+warmup):
        #data = trainer.data_iter()

        times = time.time()
        data = datasets[data_index%len(datasets)]
        logger.info("Iter: %d, Dataload time: %.10f" % (i, time.time() - times))
        times = time.time()
        data_index += 1
        # optimizer.zero_grad()
        output = model(data)
        # logger.info("Forward time: %.10f" % (time.time() - times))
        # times = time.time()
        loss = F.cross_entropy(output, target)
        # print("Main process. begin backward()\n")

        logger.info("Iter: %d, Forward time: %.10f" % (i, time.time() - times))
        times = time.time()
  
        # print("Main process. end backward()\n")
        # optimizer.step()
        # logger.info("Communication and updating time: %.10f" % (time.time() - times))

        # forward + backward + optimize
        # outputs = trainer.net(inputs)
        # loss = trainer.criterion(outputs, labels)
        torch.cuda.synchronize()
        times = time.time()
 
        if i >= warmup:
            p.start()
        loss.backward()
        logger.info("Iter: %d, Backward time: %.10f" % (i, time.time() - times))

        torch.cuda.synchronize()
    layerwise_times, sum_total = p.get_layerwise_times()
    seq_keys = p.get_backward_seq_keys()
    p.stop()
    seq_keys[::-1], layerwise_times[::-1], p.get_backward_key_sizes()[::-1]
    logger.info("seq_keys: %s" % seq_keys[::-1])
    logger.info("layerwise_times: %s" % layerwise_times[::-1])
    logger.info("backward_key_sizes: %s" % p.get_backward_key_sizes()[::-1])
    logger.info("total backward_times in profiling: %f" % np.sum(layerwise_times[::-1]) )












