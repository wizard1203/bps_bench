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

logfile = os.path.join(relative_path, 'one_tensor_test_size'+str(args.tensor_size)+'worker'+str(args.worker_id)+'rank'+str(bps.local_rank())+'.log')


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

tensor = t.rand(1024, 1024, size)
tensor = tensor.view(-1)
tensor_time = 0
for i in range(iters):
    start = time.time()
    tensor_compressed, ctx = compression.compress(tensor)
    handle = bps.byteps_push_pull(tensor_compressed, average=True, name='OnlyTensor.'+'size'+str(size))
    output = bps.synchronize(handle)
    del handle
    #tensor = bps.push_pull(tensor)
    end = time.time()
    if i > 10 and i < 51:
        tensor_time += end - start
    logger.info("tensor size %d B, iteration: %d, time: %.3f" %(1024*1024*size*4, i, end-start))
    print("tensor size %d B, iteration: %d, time: %.3f" %(1024*1024*size*4, i, end-start))
results[1024*1024*size*4] = tensor_time/40 

pp = pprint.PrettyPrinter(indent=2)
pp.pprint(results)

