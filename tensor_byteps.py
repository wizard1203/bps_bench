import torch as t
import time 
import sys
import os 
import pprint
import argparse
import logging

import bpstorch__init__2 as bps
import numpy as np


parser = argparse.ArgumentParser(description='One Tensor Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--same', type=int, default=0,
help='if using servers with workers on same machines')

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


parser.add_argument('--tensor-size', type=int, default=1,
                    help='aaa')


args = parser.parse_args()

bps.init()


t.cuda.set_device(bps.local_rank())

# BytePS: (optional) compression algorithm.
compression = bps.Compression.fp16 if False  else bps.Compression.none


logger = logging.getLogger()

logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

relative_path = './one_tensor_test'
if args.same == 1:
    relative_path += '_same_log'
else:
    relative_path += '_log'


logfile = os.path.join(relative_path, 'one_tensor_test_size'+str(args.tensor_size)+'KB-network'+str(args.DMLC_PS)+'-nworkers'+str(args.nworkers)+'-nservers'+str(args.nservers)+'worker'+str(args.worker_id)+'rank'+str(bps.local_rank())+'.log')


hdlr = logging.FileHandler(logfile)
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.info('configurations: %s', args)



iters = 55
results = {}
size = args.tensor_size

#for size in [8, 16, 32, 64, 128, 256]: 
#    bps.declare('OnlyTensor.'+'size'+str(size))
 
#for size in [8, 16, 32, 64, 128, 256]:
#    tensor = t.rand(1024, 1024, size)
#    tensor = tensor.view(-1)
#    tensor_time = 0
#    for i in range(iters):
#        start = time.time()
#        tensor_compressed, ctx = compression.compress(tensor)
#        handle = bps.byteps_push_pull(tensor_compressed, average=True, name='OnlyTensor.'+'size'+str(size))
#        output = bps.synchronize(handle)
#        del handle
#        #tensor = bps.push_pull(tensor)
#        end = time.time()
#        if i > 10 and i < 51:
#            tensor_time += end - start
#        logger.info("tensor size %d B, iteration: %d, time: %.3f" %(1024*1024*size*4, i, end-start))
#        print("tensor size %d B, iteration: %d, time: %.3f" %(1024*1024*size*4, i, end-start))
#    results[1024*1024*size*4] = tensor_time/40 

#tensor = t.rand(1024, 1024, size)
tensor = t.rand(256, size)
tensor = tensor.view(-1)
tensor.cuda()
tensor_time = 0
bps.declare('OnlyTensor.'+'size'+str(size)+'KB')

iter_times = []
for i in range(iters):
    time.sleep(0.1)
    start = time.time()
    tensor_compressed, ctx = compression.compress(tensor)
    handle = bps.byteps_push_pull(tensor_compressed, average=True, name='OnlyTensor.'+'size'+str(size)+'KB')
    output = bps.synchronize(handle)
    del handle
    #tensor = bps.push_pull(tensor)
    end = time.time()
    if i > 10 and i < 51:
        tensor_time = end - start
        iter_times.append(tensor_time)
    #logger.info("tensor size %d B, iteration: %d, time: %.3f" %(1024*1024*size*4, i, end-start))
    logger.info("tensor size %d B, iteration: %d, time: %.8f" %(256*size*4, i, end-start))
    #print("tensor size %d B, iteration: %d, time: %.3f" %(1024*1024*size*4, i, end-start))
#results[1024*1024*size*4] = tensor_time/40 

print(iter_times)
iter_times_mean = np.mean(iter_times)
iter_times_conf = 1.96 * np.std(iter_times)
logger.info('Rank: %d, Iter time: %.8f +-%.8f' % (bps.local_rank(), iter_times_mean, iter_times_conf))

