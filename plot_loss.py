# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import matplotlib.pyplot as plt
import matplotlib
import copy
matplotlib.use("TkAgg")
import numpy as np
import datetime
import itertools
import utils as u
import minmax_communication_cost
import logs_content
#markers=['.','x','o','v','^','<','>','1','2','3','4','8','s','p','*']
#markers=['.','x','o','v','^','<','>']
markers=[None]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'olive']
#colors = colors[2:7]
#colors = colors[0:4]
colors = colors[0:6]
markeriter = itertools.cycle(markers)
coloriter = itertools.cycle(colors)

#OUTPUTPATH='/media/sf_Shared_Data/tmp/p2psgd'
OUTPUTPATH='./'
max_epochs = 500
# max_epochs_dict = {
#     'mnistCNN':800,
#     'cifar10CNN':800,
#     'resnet20':800
# }
max_epochs_dict = {
    'mnistCNN':100,
    'cifar10CNN':700,
    'resnet20':900
}
EPOCH = True
#FONTSIZE=17
FONTSIZE=14
# fig, ax = plt.subplots(1,1,figsize=(5,3.8))
# ax2 = None

STANDARD_TITLES = {
        'resnet20': 'ResNet-20',
        'vgg16': 'VGG16',
        'alexnet': 'AlexNet',
        'resnet50': 'ResNet-50',
        'lstmptb': 'LSTM-PTB',
        'lstman4': 'LSTM-AN4'
        }

def get_real_title(title):
    return STANDARD_TITLES.get(title, title)

def seconds_between_datetimestring(a, b):
    a = datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
    b = datetime.datetime.strptime(b, '%Y-%m-%d %H:%M:%S')
    delta = b - a 
    return delta.days*86400+delta.seconds
sbd = seconds_between_datetimestring

def get_loss(line, isacc=False):
    if EPOCH:
        #if line.find('Epoch') > 0 and line.find('acc:') > 0:
        valid = line.find('val acc: ') > 0 if isacc else line.find('avg loss: ') > 0
        #if line.find('Epoch') > 0 and line.find('loss:') > 0 and not line.find('acc:')> 0:
        if not valid:
            # valid = line.find('val top-1 acc: ') > 0 if isacc else line.find('average loss: ') > 0
            valid = line.find('val top-1 acc: ') > 0 if isacc else line.find('avg loss: ') > 0    
        if line.find('Epoch') > 0 and valid: 
            if isacc:
                if line.find('val acc: ') >0:
                    items = line.split(' ')
                    #print(items)
                    loss = float(items[-1])
                    # print(loss)
                    t = line.split(' I')[0].split(',')[0]
                    t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                else:
                    items = line.split('acc:')
                    #print(items)
                    loss = float(items[1].split(',')[0].strip())
                    # print(loss)
                    t = line.split(' I')[0].split(',')[0]
                    t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')         
            else:
                items = line.split(' ')
                #print(items)
                loss = float(items[-1])
                # print(loss)
                t = line.split(' I')[0].split(',')[0]
                t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            return loss, t
    else:
        if line.find('average forward') > 0:
            items = line.split('loss:')[1]
            loss = float(items[1].split(',')[0])
            t = line.split(' I')[0].split(',')[0]
            t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            return loss, t
    return None, None

def read_losses_from_log(logfile, isacc=False):
    f = open(logfile)
    print(logfile)
    losses = []
    times = []
    average_delays = []
    lrs = []
    i = 0
    time0 = None
    # max_epochs = max_epochs
    counter = 0
    for line in f.readlines():
        #if line.find('average forward') > 0:
        valid = line.find('val acc: ') > 0 if isacc else line.find('average loss: ') > 0
        if not valid:
            valid = line.find('val top-1 acc: ') > 0 if isacc else line.find('average loss: ') > 0    
        if line.find('Epoch') > 0 and valid:
        #if not time0 and line.find('INFO [  100]') > 0:
            # print(line)
            t = line.split(' I')[0].split(',')[0]
            t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            if not time0:
                time0 = t
        if line.find('lr: ') > 0:
            try:
                lr = float(line.split(',')[-2].split('lr: ')[-1])
                lrs.append(lr)
            except:
                pass
        if line.find('average delay: ') > 0:
            delay = int(line.split(':')[-1])
            average_delays.append(delay)
        loss, t = get_loss(line, isacc)
        # print(loss)
        if loss and t:
            # print(logfile, loss)
            counter += 1
            losses.append(loss)
            times.append(t)
        if counter > max_epochs:
            break

        #if line.find('Epoch') > 0 and line.find('acc:') > 0:
        #    items = line.split(' ')
        #    loss = float(items[-1])
        #    #items = line.split('loss:')[1]
        #    #loss = float(items[1].split(',')[0])

        #    losses.append(loss)
        #    t = line.split(' I')[0].split(',')[0]
        #    t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        #    times.append(t)
    f.close()
    if not EPOCH:
        print('not find epoch')
        average_interval = 10
        times = [times[t*average_interval] for t in range(len(times)/average_interval)]
        losses = [np.mean(losses[t*average_interval:(t+1)*average_interval]) for t in range(len(losses)/average_interval)]
    print(losses[0])
    if len(times) > 0:
        t0 = time0 if time0 else times[0] #times[0]
        for i in range(0, len(times)):
            delta = times[i]- t0
            times[i] = delta.days*86400+delta.seconds
    return losses, times, average_delays, lrs

def read_norm_from_log(logfile):
    f = open(logfile)
    means = []
    stds = []
    for line in f.readlines():
        if line.find('gtopk-dense norm mean') > 0:
            items = line.split(',')
            mean = float(items[-2].split(':')[-1])
            std = float(items[--1].split(':')[-1])
            means.append(mean)
            stds.append(std)
    print('means: ', means)
    print('stds: ', stds)
    return means, stds

def plot_loss(logfile, label, isbandwidth=False, plot_loss=False, isacc=False, title='ResNet-20', scale=None, comm_ratio=None, comm_bandwidth=None, ax=None):
    losses, times, average_delays, lrs = read_losses_from_log(logfile, isacc=isacc)
    norm_means, norm_stds = read_norm_from_log(logfile)
    new_losses = []
    if scale:
        for item in losses:
            for i in range(scale):
                new_losses.append(item)
        losses = new_losses

    print(label,title,":", max(losses))
    # print(losses, times, average_delays, lrs)
    #print('times: ', times)
    #print('Learning rates: ', lrs)
    if len(average_delays) > 0:
        delay = int(np.mean(average_delays))
    else:
        delay = 0
    if delay > 0:
        label = label + ' (delay=%d)' % delay
    #plt.plot(losses, label=label, marker='o')
    #plt.xlabel('Epoch')
    #plt.title('ResNet-20 loss')
    if isacc:
        ax.set_ylabel('Top-1 Validation Accuracy [%]')
    else:
        ax.set_ylabel('Training loss')
    #plt.title('ResNet-50')
    #ax.set_title(get_real_title(title))

    # marker = markeriter.next()
    # color = coloriter.next()
    marker = next(markeriter)
    color = next(coloriter)

    #print('marker: ', marker)
    #ax.plot(losses[0:180], label=label, marker=marker, markerfacecolor='none')
    if comm_ratio:
        if title.lower() == 'resnet-20':
            model_size = 269722
        elif title.lower() == 'mnist-cnn':
            model_size = 6653628.00
        elif title.lower() == 'cifar10-cnn':
            model_size = 7025886.000000
        model_size = 6653628.00
        x = np.arange(1, len(losses)+1)
        if comm_bandwidth:
            if type(comm_bandwidth) == type([]):
                print("???!!!")
            else:
                x = (comm_ratio * model_size * x /(1024 * 1024 * 4)) / (comm_bandwidth/8)
        else:
            x = comm_ratio * model_size * x / (1024 * 1024 * 4)
        ax.set_xlabel('Accumulated communication size [Mbytes]')
    else:
        x = np.arange(len(losses))
        ax.set_xlabel('Epoch')
    if isacc:
        losses = np.array(losses)*100
    ax.plot(x, losses, label=label, marker=marker, linewidth=1.5, markerfacecolor='none', color=color)
    if False and len(norm_means) > 0:
        global ax2
        if ax2 is None:
            ax2 = ax.twinx()
            ax2.set_ylabel('L2-Norm of : gTopK-Dense')
        ax2.plot(norm_means, label=label+' norms', color=color)
    if comm_ratio:
        ax.set_xlabel('Communication size [MB]')
        if comm_bandwidth:
            ax.set_xlabel('Communication time [s]')
    else:
        ax.set_xlabel('Epoch')
    #plt.plot(times, losses, label=label, marker=markeriter.next())
    #plt.xlabel('Time [s]')
    ax.grid(linestyle=':')
    if len(lrs) > 0:
        lr_indexes = [0]
        lr = lrs[0]
        for i in range(len(lrs)):
            clr = lrs[i]
            if lr != clr:
                lr_indexes.append(i)
                lr = clr
        #for i in lr_indexes:
        #    if i < len(losses):
        #        ls = losses[i]
        #        ax.text(i, ls, 'lr=%f'%lrs[i])
    u.update_fontsize(ax, FONTSIZE)

def plot_bandwidth(logfile, label, isbandwidth=False, plot_loss=False, isacc=False, title='ResNet-20', scale=None, comm_ratio=None, comm_bandwidth=None, ax=None):
    # losses, times, average_delays, lrs = read_losses_from_log(logfile, isacc=isacc)
    # norm_means, norm_stds = read_norm_from_log(logfile)
    # new_losses = []
    # if scale:
    #     for item in losses:
    #         for i in range(scale):
    #             new_losses.append(item)
    #     losses = new_losses

    # # print(losses, times, average_delays, lrs)
    # #print('times: ', times)
    # #print('Learning rates: ', lrs)
    # if len(average_delays) > 0:
    #     delay = int(np.mean(average_delays))
    # else:
    #     delay = 0
    # if delay > 0:
    #     label = label + ' (delay=%d)' % delay
    #plt.plot(losses, label=label, marker='o')
    #plt.xlabel('Epoch')
    # #plt.title('ResNet-20 loss')
    # if isacc:
    #     ax.set_ylabel('Top-1 Validation Accuracy')
    # else:
    #     ax.set_ylabel('Training loss')
    ax.set_ylabel('Bandwidth [Mbits/s]')
    #plt.title('ResNet-50')
    # ax.set_title(get_real_title(title))
    # marker = markeriter.next()
    # color = coloriter.next()
    marker = next(markeriter)
    color = next(coloriter)
    #print('marker: ', marker)
    #ax.plot(losses[0:180], label=label, marker=marker, markerfacecolor='none')
    # if comm_ratio:
    #     model_size = 6653628.00
    #     x = np.arange(len(losses))
    #     if comm_bandwidth:
    #         if type(comm_bandwidth) == type([]):
    #             print("???!!!")
    #         else:
    #             x = comm_ratio * model_size * x / comm_bandwidth
    #     else:
    #         x = comm_ratio * model_size * x
    #     ax.set_xlabel('Total communication size')
    # else:
    #     x = np.arange(len(losses))
    #     ax.set_xlabel('Epoch')
    x = np.arange(len(comm_bandwidth))
    ax.set_xlabel('Iteration')
    ax.plot(x, comm_bandwidth, label=label, marker=marker, linewidth=0.5, markerfacecolor='none', color=color)
    if False and len(norm_means) > 0:
        global ax2
        if ax2 is None:
            ax2 = ax.twinx()
            ax2.set_ylabel('L2-Norm of : gTopK-Dense')
        ax2.plot(norm_means, label=label+' norms', color=color)
    #plt.plot(times, losses, label=label, marker=markeriter.next())
    #plt.xlabel('Time [s]')
    ax.grid(linestyle=':')
    # if len(lrs) > 0:
    #     lr_indexes = [0]
    #     lr = lrs[0]
    #     for i in range(len(lrs)):
    #         clr = lrs[i]
    #         if lr != clr:
    #             lr_indexes.append(i)
    #             lr = clr
    #     #for i in lr_indexes:
    #     #    if i < len(losses):
    #     #        ls = losses[i]
    #     #        ax.text(i, ls, 'lr=%f'%lrs[i])
    u.update_fontsize(ax, FONTSIZE)


def plot_loss_with_host(hostn, nworkers, hostprefix, baseline=False):
    if not baseline or nworkers == 64:
        port = 5922
    else:
        port = 5945
    for i in range(hostn, hostn+1):
        for j in range(2, 3):
            host='%s%d-%d'%(hostprefix, i, port+j)
            if baseline:
                logfile = './ad-sgd-%dn-%dw-logs/'%(nworkers/4, nworkers)+host+'.log'
            else:
                logfile = './%dnodeslogs/'%nworkers+host+'.log'
                if nworkers == 256 and hostn < 48:
                    host='%s%d.comp.hkbu.edu.hk-%d'%(hostprefix, i, port+j)
                    logfile = './%dnodeslogs/'%nworkers+host+'.log'
                #csr42.comp.hkbu.edu.hk-5922.log
                #logfile = './%dnodeslogs-w/'%nworkers+host+'.log'
            label = host+' ('+str(nworkers)+' workers)'
            if baseline:
                label += ' Baseline'
            plot_loss(logfile, label) 

def resnet20():
    plot_with_params('resnet20', 4, 32, 0.1, 'gpu21', 'Allreduce', prefix='allreduce')
    #plot_with_params('resnet20', 4, 32, 0.01, 'gpu11', 'Allreduce', prefix='allreduce-baseline-wait-dc1-model-debug', nsupdate=1)
    plot_with_params('resnet20', 4, 32, 0.1, 'hpclgpu', '(Ref 1/4 data)', prefix='compression-modele',sparsity=0.01)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu13', 'ADPSGD+Spar', prefix='compression-dc1-model-debug',sparsity=0.999)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu13', 'ADPSGD+Gradients', prefix='adpsgd-dc1-grad-debug',sparsity=None)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu13', 'ADPSGD+Gradients Sync', prefix='adpsgd-wait-dc1-grad-debug',sparsity=None)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu21', 'Allreduce', prefix='allreduce')
    plot_with_params('resnet20', 4, 32, 0.01, 'gpu11', 'Sequence d=0.1, lr=0.01', prefix='allreduce-comp-sequence-baseline-wait-dc1-model-debug', nsupdate=1, sg=1.5, density=0.1, force_legend=True)
    plot_with_params('resnet20', 4, 32, 0.1, 'gpu11', 'Sequence d=0.1, lr=0.1, wp', prefix='allreduce-comp-sequence-baseline-gwarmup-wait-dc1-model-debug', nsupdate=1, sg=1.5, density=0.1, force_legend=True)
    plot_with_params('resnet20', 4, 32, 0.01, 'gpu17', 'Sequence d=0.1, lr=0.01, wp', prefix='allreduce-comp-sequence-baseline-gwarmup-wait-dc1-model-debug', nsupdate=1, sg=1.5, density=0.1, force_legend=True)
    #plot_with_params('resnet20', 4, 32, 0.01, 'gpu11', 'Sequence density=0.1', prefix='allreduce-comp-sequence-baseline-wait-dc1-model-debug', nsupdate=1, sg=1.5, density=0.01, force_legend=True)

def vgg16():
    #plot_with_params('vgg16', 4, 32, 0.1, 'gpu17', 'Allreduce', prefix='allreduce')
    #plot_with_params('vgg16', 4, 128, 1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 128, 1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 16, 128, 1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    plot_with_params('vgg16', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modele', title='VGG16', sparsity=0.95)
    #plot_with_params('vgg16', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modele', title='VGG16', sparsity=0.98)
    plot_with_params('vgg16', 8, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.1, 'hpclgpu', 'ADPSGD+DC4', prefix='baseline-dc4-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.01, 'hpclgpu', 'ADPSGD+DC4', prefix='baseline-dc4-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-wait-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.1, 'MGD', 'ADPSGD ', prefix='baseline-modelmgd', title='VGG16')
    plot_with_params('vgg16', 16, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 16, 32, 0.01, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 16, 32, 0.1, 'gpu20', 'ADPSGD ', prefix='baseline-modelk80', title='VGG16')
    #plot_with_params('vgg16', 16, 32, 0.0005, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    plot_with_params('vgg16', 4, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')
    plot_with_params('vgg16', 8, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')
    plot_with_params('vgg16', 16, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')

def mnistnet():
    plot_with_params('mnistnet', 100, 50, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl')
    plot_with_params('mnistnet', 1, 512, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl')
    plot_with_params('mnistnet', 1, 64, 0.01, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl')


def mnistflnet():
    prefix = 'baseline-gwarmup-wait-noniid-dc1-model-fl'
    plot_with_params('mnistflnet', 100, 50, 0.1, 'FedAvg', machine='csrlogs',  prefix='baseline-gwarmup-wait-dc1-model-fl', title='MNIST CNN')

    # plot_with_params('mnistflnet', 100, 50, 0.1, machine='gpuhome_fl',  prefix='baseline-modelhpcl', title='MNIST CNN')
    # plot_with_params('mnistflnet', 100, 50, 0.1, machine='gpuhomedc_p2p', prefix='baseline-modelhpcl', title='MNIST CNN')
    # plot_with_params('mnistflnet', 100, 50, 0.1, machine='gpuhomedc2_p2p', prefix='baseline-modelhpcl', title='MNIST CNN')
    # plot_with_params('mnistflnet', 100, 50, 0.1, machine='hswlogs', prefix='baseline-modelhpcl', title='MNIST CNN')


def cifar10flnet():
    plot_with_params('cifar10flnet', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='CIFAR-10 CNN')



def plot_one_worker():
    def _plot_with_params(bs, lr, isacc=True):
        logfile = './logs/resnet20/accresnet20-bs%d-lr%s.log' % (bs, str(lr))
        t = '(lr=%.4f, bs=%d)'%(lr, bs)
        plot_loss(logfile, t, isacc=isacc, title='resnet20') 
    _plot_with_params(32, 0.1)
    _plot_with_params(32, 0.01)
    _plot_with_params(32, 0.001)
    _plot_with_params(64, 0.1)
    _plot_with_params(64, 0.01)
    _plot_with_params(64, 0.001)
    _plot_with_params(128, 0.1)
    _plot_with_params(128, 0.01)
    _plot_with_params(128, 0.001)
    _plot_with_params(256, 0.1)
    _plot_with_params(256, 0.01)
    _plot_with_params(256, 0.001)
    _plot_with_params(512, 0.1)
    _plot_with_params(512, 0.01)
    _plot_with_params(512, 0.001)
    _plot_with_params(1024, 0.1)
    _plot_with_params(1024, 0.01)
    _plot_with_params(1024, 0.001)
    _plot_with_params(2048, 0.1)

def resnet50():
    plot_loss('baselinelogs/accresnet50-lr0.01-c40,70.log', 'allreduce 4 GPUs', isacc=False, title='ResNet-50') 

    plot_with_params('resnet50', 8, 64, 0.01, 'gpu10', 'allreduce 8 GPUs', prefix='allreduce-debug')
    plot_with_params('resnet50', 8, 64, 0.01, 'gpu16', 'ADPSGD', prefix='baseline-dc1-modelk80')

def plot_norm_diff():
    network = 'resnet20'
    bs =32
    #network = 'vgg16'
    #bs = 128
    path = './logs/allreduce-comp-gtopk-baseline-gwarmup-dc1-model-normtest/%s-n4-bs%d-lr0.1000-ns1-sg1.50-ds0.001' % (network,bs)
    epochs = 80
    arr = None
    arr2 = None
    arr3 = None
    for i in range(1, epochs):
        fn = '%s/gtopknorm-rank0-epoch%d.npy' % (path, i)
        fn2 = '%s/randknorm-rank0-epoch%d.npy' % (path, i)
        fn3 = '%s/upbound-rank0-epoch%d.npy' % (path, i)
        fn4 = '%s/densestd-rank0-epoch%d.npy' % (path, i)
        if arr is None:
            arr = np.load(fn)
            arr2 = np.load(fn2)
            arr3 = np.load(fn3)
            arr4 = np.load(fn4)
        else:
            arr = np.concatenate((arr, np.load(fn)))
            arr2 = np.concatenate((arr2, np.load(fn2)))
            arr3 = np.concatenate((arr3, np.load(fn3)))
            arr4 = np.concatenate((arr4, np.load(fn4)))
    #plt.plot(arr-arr2, label='||x-gtopK(x)||-||x-randomK(x)||')
    plt.plot(arr4, label='Gradients std')
    plt.xlabel('# of iteration')
    #plt.ylabel('||x-gtopK(x)||-||x-randomK(x)||')
    plt.title(network)
    #plt.plot(arr2, label='||x-randomK(x)||')
    #plt.plot(arr3, label='(1-K/n)||x||')

def loss(network):
    # Convergence
    gtopk_name = 'gTopKAllReduce'
    dense_name = 'DenseAllReduce'

    # resnet20
    if network == 'resnet20':
        plot_with_params(network, 4, 32, 0.1, 'gpu13', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
    elif network == 'vgg16':
        # vgg16
        plot_with_params(network, 4, 128, 0.1, 'gpu13', dense_name, prefix='allreduce-baseline-dc1-model-icdcs', nsupdate=1, force_legend=True)
        plot_with_params(network, 4, 128, 0.1, 'gpu10', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
    elif network == 'alexnet':
        plot_with_params(network, 8, 256, 0.01, 'gpu20', dense_name, prefix='allreduce-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, force_legend=True)
        plot_with_params(network, 8, 256, 0.01, 'gpu20', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
    elif network == 'resnet50':
        plot_with_params(network, 8, 64, 0.01, 'gpu10', dense_name, prefix='allreduce-debug', force_legend=True)
    #todo zhtang an4 ==============
    elif network == 'lstman4':
        plot_with_params(network, 8, 8, 1, 'gpu10', dense_name, prefix='allreduce-debug', force_legend=True)

def communication_speed():
    pass

def plot_with_params(dnn, nworkers, bs, lr, legend, isbandwidth=False, isacc=False, logfile='', title='ResNet-20', sparsity=None, nsupdate=None, sg=None, density=None, force_legend=False, scale=None, comm_ratio=None, comm_bandwidth=None, ax=None):
    # postfix='5922'
    # if prefix.find('allreduce')>=0:
    #     postfix='0'
    # if sparsity:
    #     logfile = './logs/%s/logs/%s/%s-n%d-bs%d-lr%.4f-s%.5f' % (machine, prefix, dnn, nworkers, bs, lr, sparsity)
    # elif nsupdate:
    #     logfile = './logs/%s/logs/%s/%s-n%d-bs%d-lr%.4f-ns%d' % (machine, prefix, dnn, nworkers, bs, lr, nsupdate)
    # else:
    #     logfile = './logs/%s/logs/%s/%s-n%d-bs%d-lr%.4f' % (machine, prefix, dnn, nworkers, bs, lr)
    # if sg is not None:
    #     logfile += '-sg%.2f' % sg
    # if density is not None:
    #     logfile += '-ds%s' % str(density)
    # logfile += '%s.log' % (postfix)

    print('logfile: ', logfile)
    if force_legend:
        l = legend
    else:
        l = legend+ '(lr=%.4f, bs=%d, %d workers)'%(lr, bs, nworkers)
    plot_loss(logfile, l, isbandwidth=isbandwidth, isacc=isacc, title=title, scale=scale, comm_ratio=comm_ratio, comm_bandwidth=comm_bandwidth, ax=ax)

def ICNP2020_SAPS_convergence(network, workers, if_iid, ax=None):
    FedAvg_name = 'FedAvg'
    FedAvg_sparse_name = 'S-FedAvg'
    DPSGD_FIX_name = 'D-PSGD'
    DPSGD_name = 'APS-FL'
    DCD_name = 'DCD-PSGD'
    SAPS_name = 'SAPS-FL'
    DensePSGD_name = 'PSGD'
    TopKPSGD_name = 'TopK-PSGD'


    mnistnet_name = 'mnistCNN'
    cifar10flnet_name = 'cifar10CNN'
    resnet20_name = 'resnet20'

    Algorithms = ['PSGD', 'TopK-PSGD', 'FedAvg', 'S-FedAvg', 'D-PSGD', 'DCD-PSGD', 'APS-FL', 'SAPS-FL']

    for algo in Algorithms[2:8]:
        num_comms_epoch = logs_content.get_experiment_config(workers, if_iid, network, algo)['num_comms_epoch']
        comm_data_scale = logs_content.get_experiment_config(workers, if_iid, network, algo)['comm_data_scale']
        plot_with_params(network, workers, logs_content.get_experiment_config(workers, if_iid, network, algo)['batch_size']
        , logs_content.get_experiment_config(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
            logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
            force_legend=True, ax=ax)

    # if workers =='32':
    #     if network == mnistnet_name:
    #         if if_iid :
    #             print("222")
    #             for algo in Algorithms[2:7]:
    #                 plot_with_params(network, workers, logs_content.get_log_file(workers, if_iid, network, algo)['batch_size']
    #                 , logs_content.get_log_file(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
    #                     logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
    #                     force_legend=True, ax=ax)

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DensePSGD_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DensePSGD_name), 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], TopKPSGD_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, TopKPSGD_name), 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, FedAvg_name)['lr'], FedAvg_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_name), 
    #             #     force_legend=True, ax=ax) # scale means random selection in FedAvg

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_sparse_name), 
    #             #     force_legend=True, ax=ax)

    #             # # plot_with_params(mnistnet_name, workers, 50, 0.1, DPSGD_name, isacc=True, title='MNIST CNN',
    #             # #     logfile='logs/gpuhomedclogs_before/logs/baseline-OURS-wait-dc1-model-dc/mnistflnet-n32-bs50-lr0.0500/gpu11-20111.log', 
    #             # #     force_legend=True)

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DPSGD_name), 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DCD_name), 
    #             #     force_legend=True, ax=ax)

    #             # # plot_with_params(mnistnet_name, workers, 50, 0.1, SAPS_name, isacc=True, title='MNIST CNN',
    #             # #     logfile='logs/gpuhomedclogs_before/logs/compression-OURS-wait-dc1-model-dc/mnistflnet-n32-bs50-lr0.0500-s0.99000/gpu13-20111.log', 
    #             # #     force_legend=True, ax=ax)
    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, SAPS_name), 
    #             #     force_legend=True, ax=ax)

    #     if network == cifar10flnet_name:
    #         if if_iid :
    #             for algo in Algorithms[2:7]:
    #                 plot_with_params(network, workers, logs_content.get_log_file(workers, if_iid, network, algo)['batch_size']
    #                 , logs_content.get_log_file(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
    #                     logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
    #                     force_legend=True, ax=ax)
    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'],  DensePSGD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DensePSGD_name), 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'],  TopKPSGD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, TopKPSGD_name), 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_name), 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_sparse_name), 
    #             #     force_legend=True, ax=ax)

    #             # # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='CIFAR10 CNN',
    #             # #     logfile='logs/hsw-logs/hsw-dc-logs-before0109/logs/baseline-OURS-wait-dc1-model-dc/cifar10flnet-n32-bs100-lr0.0400/hsw224-20111.log', 
    #             # #     force_legend=True)
    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DPSGD_name), 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DCD_name), 
    #             #     force_legend=True, ax=ax)

    #             # # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='CIFAR10 CNN',
    #             # #     logfile='logs/gpuhomedclogs_before/logs/compression-OURS-wait-dc1-model-dc/cifar10flnet-n32-bs100-lr0.0400-s0.99000/gpu13-20111.log', 
    #             # #     force_legend=True, ax=ax)
    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, SAPS_name), 
    #             #     force_legend=True, ax=ax)


    #     if network == resnet20_name:
    #         if if_iid :
    #             for algo in Algorithms[2:7]:
    #                 plot_with_params(network, workers, logs_content.get_log_file(workers, if_iid, network, algo)['batch_size']
    #                 , logs_content.get_log_file(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
    #                     logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
    #                     force_legend=True, ax=ax)
    #             # # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DensePSGD_name, isacc=True, title='ResNet-20',
    #             # #     logfile=logs_content.get_log_file(workers, if_iid, network, DensePSGD_name), 
    #             # #     force_legend=True, ax=ax)

    #             # # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], TopKPSGD_name, isacc=True, title='ResNet-20',
    #             # #     logfile=logs_content.get_log_file(workers, if_iid, network, TopKPSGD_name), 
    #             # #     force_legend=True, ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='ResNet-20',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_name), 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='ResNet-20',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_sparse_name), 
    #             #     force_legend=True, ax=ax)

    #             # # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name+"before", isacc=True, title='ResNet-20',
    #             # #     logfile='logs/hsw-logs/hsw-dc-logs-before0109/logs/baseline-OURS-wait-dc1-model-dc/resnet20-n32-bs64-lr0.2000/hsw224-20111.log', 
    #             # #     force_legend=True)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='ResNet-20',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DPSGD_name), 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='ResNet-20',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DCD_name), 
    #             #     force_legend=True, ax=ax)

    #             # # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='ResNet-20',
    #             # #     logfile='logs/gpuhomedclogs_before/logs/compression-OURS-wait-dc1-model-dc/resnet20-n32-bs64-lr0.1000-s0.99000/gpu13-20111.log', 
    #             # #     force_legend=True, ax=ax)
    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='ResNet-20',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, SAPS_name), 
    #             #     force_legend=True, ax=ax)

    # if workers =='14':    
    #     if network == mnistnet_name:
    #         if if_iid :
    #             print("222")
    #             for algo in Algorithms[2:7]:
    #                 plot_with_params(network, workers, logs_content.get_log_file(workers, if_iid, network, algo)['batch_size']
    #                 , logs_content.get_log_file(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
    #                     logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
    #                     force_legend=True, ax=ax)
    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_name), 
    #             #     force_legend=True, ax=ax) # scale means random selection in FedAvg

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_sparse_name), 
    #             #     force_legend=True, ax=ax)

    #             # # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='MNIST CNN',
    #             # #     logfile='logs/hsw224_14workerDPSGD/logs/baseline-OURS-wait-dc1-model-dc/mnistflnet-n14-bs50-lr0.0500/hsw224-20111.log', 
    #             # #     force_legend=True)
    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='MNIST CNN',
    #             #     logfile='logs/hsw224011222/logs/baseline-OURS-NoAsk-TOPO-RING-wait-dc1-model-dc/mnistflnet-n14-bs50-lr0.0500/hsw224-20111.log', 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='MNIST CNN',
    #             #     logfile='logs/gpuhomeDCD14worker/mnistflnet-n14-bs50-lr0.0500-s0.75000/gpu17-20111.log', 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='MNIST CNN',
    #             #     logfile='logs/hsw-eval-ours-logs/compression-OURS-NoAsk-TOPO-REAL-RealCommEval-lb-1.45-k-0.1-wait-dc1-model-dc/mnistflnet-n14-bs50-lr0.0500-s0.99000/hsw224-20111.log', 
    #             #     force_legend=True, ax=ax)




    #     if network == cifar10flnet_name:
    #         if if_iid :
    #             for algo in Algorithms[2:7]:
    #                 plot_with_params(network, workers, logs_content.get_log_file(workers, if_iid, network, algo)['batch_size']
    #                 , logs_content.get_log_file(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
    #                     logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
    #                     force_legend=True, ax=ax)
    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile='logs/scilogs0112/baseline-wait-dc1-model-fl/cifar10flnet-n14-bs100-lr0.0400/scigpu11-29223.log',
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile='logs/scilogs0112/compression-gwarmup-wait-dc1-model-fl/cifar10flnet-n14-bs100-lr0.0400-s0.99000/scigpu11-29223.log',
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile='logs/hsw224011222/logs/baseline-OURS-NoAsk-TOPO-RING-wait-dc1-model-dc/cifar10flnet-n14-bs100-lr0.0400/hsw224-20111.log', 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile='logs/gpuhomeDCD14worker/cifar10flnet-n14-bs100-lr0.0400-s0.75000/gpu17-20111.log', 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile='logs/hsw-eval-ours-logs/compression-OURS-NoAsk-TOPO-REAL-RealCommEval-lb-1.45-k-0.1-wait-dc1-model-dc/cifar10flnet-n14-bs100-lr0.0400-s0.99000/hsw224-20111.log', 
    #             #     force_legend=True, ax=ax)

    #     if network == resnet20_name:
    #         if if_iid :
    #             for algo in Algorithms[2:7]:
    #                 plot_with_params(network, workers, logs_content.get_log_file(workers, if_iid, network, algo)['batch_size']
    #                 , logs_content.get_log_file(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
    #                     logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
    #                     force_legend=True, ax=ax)
    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='ResNet-20',
    #             #     logfile='logs/hsw-fl-logs0112/baseline-wait-dc1-model-fl/resnet20-n14-bs64-lr0.1000/hsw224-8923.log',
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='ResNet-20',
    #             #     logfile='logs/hsw-fl-logs0112/compression-gwarmup-wait-dc1-model-fl/resnet20-n14-bs64-lr0.1000-s0.99000/hsw224-8923.log',
    #             #     force_legend=True, ax=ax)

    #             # # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='ResNet-20',
    #             # #     logfile='logs/hsw224_14workerDPSGD/logs/baseline-OURS-wait-dc1-model-dc/resnet20-n14-bs64-lr0.1000/hsw224-20111.log', 
    #             # #     force_legend=True)
    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='ResNet-20',
    #             #     logfile='logs/hsw224011222/logs/baseline-OURS-NoAsk-TOPO-RING-wait-dc1-model-dc/resnet20-n14-bs64-lr0.1000/hsw224-20111.log', 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='ResNet-20',
    #             #     logfile='logs/gpuhomeDCD14worker/resnet20-n14-bs64-lr0.1000-s0.75000/gpu17-20111.log', 
    #             #     force_legend=True, ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='ResNet-20',
    #             #     logfile='logs/hsw-eval-ours-logs/compression-OURS-NoAsk-TOPO-REAL-RealCommEval-lb-1.45-k-0.1-wait-dc1-model-dc/resnet20-n14-bs64-lr0.1000-s0.99000/hsw224-20111.log', 
    #             #     force_legend=True, ax=ax)

def ICNP2020_SAPS_convergence_robust(network, workers, if_iid, ax=None):
    FedAvg_name = 'FedAvg'
    FedAvg_sparse_name = 'S-FedAvg'
    DPSGD_name = 'APS-FL'
    DCD_name = 'DCD-PSGD'
    SAPS_name = 'SAPS-FL'
    DensePSGD_name = 'PSGD'
    TopKPSGD_name = 'TopK-PSGD'
    DPSGD_FIX_name = 'D-PSGD'

    mnistnet_name = 'mnistCNN'
    cifar10flnet_name = 'cifar10CNN'
    resnet20_name = 'resnet20'

    Algorithms = ['PSGD', 'TopK-PSGD', 'FedAvg', 'S-FedAvg', 'D-PSGD', 'DCD-PSGD', 'APS-FL', 'SAPS-FL']

    for algo in Algorithms[2:8]:
        #num_comms_epoch = logs_content.get_experiment_config(workers, if_iid, network, algo)['num_comms_epoch']
        #comm_data_scale = logs_content.get_experiment_config(workers, if_iid, network, algo)['comm_data_scale']
        plot_with_params(network, workers, logs_content.get_experiment_config(workers, if_iid, network, algo)['batch_size']
        , logs_content.get_experiment_config(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
            logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
            force_legend=True, ax=ax)


def ICNP2020_SAPS_communication(network, workers, sparsity=None, if_iid=True, ax=None):
    FedAvg_name = 'FedAvg'
    FedAvg_sparse_name = 'S-FedAvg'
    DPSGD_name = 'APS-FL'
    DCD_name = 'DCD-PSGD'
    SAPS_name = 'SAPS-FL'
    DensePSGD_name = 'PSGD'
    TopKPSGD_name = 'TopK-PSGD'
    DPSGD_FIX_name = 'D-PSGD'

    mnistnet_name = 'mnistCNN'
    cifar10flnet_name = 'cifar10CNN'
    resnet20_name = 'resnet20'
    # num_batches_per_epoch = {
    #     'mnistCNN': 1+int(50000/(int(workers)*50)),
    #     'cifar10CNN': 1+int(50000/(int(workers)*100)),
    #     'resnet20': 1+int(50000/(int(workers)*64))
    # }
    # comm_data_scale = {
    #     'FedAvg': 2,
    #     'S-FedAvg': 1+1*(1-0.99),
    #     'D-PSGD-FIX': 4,
    #     'D-PSGD': 4,
    #     'DCD-PSGD': 4*0.75,
    #     'SAPS-PSGD': 2*(1-0.99),
    #     'PSGD': 2,
    #     'TopK-PSGD': 2*(int(workers)-1)*(1-0.999)
    # }

    Algorithms = ['PSGD', 'TopK-PSGD', 'FedAvg', 'S-FedAvg', 'D-PSGD', 'DCD-PSGD', 'APS-FL', 'SAPS-FL']
    for algo in Algorithms[2:8]:
        num_comms_epoch = logs_content.get_experiment_config(workers, if_iid, network, algo)['num_comms_epoch']
        comm_data_scale = logs_content.get_experiment_config(workers, if_iid, network, algo)['comm_data_scale']
        plot_with_params(network, workers, logs_content.get_experiment_config(workers, if_iid, network, algo)['batch_size']
        , logs_content.get_experiment_config(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
            logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
            force_legend=True,comm_ratio=comm_data_scale*num_comms_epoch, ax=ax)

    # prefix = 'baseline-gwarmup-wait-noniid-dc1-model-fl'
    # if workers =='32':
    #     if network == mnistnet_name:
    #         if if_iid :
    #             print("222")
    #             for algo in Algorithms[2:7]:
    #                 num_comms_epoch = logs_content.get_experiment_config(workers, if_iid, network, algo)['num_comms_epoch']
    #                 comm_data_scale = logs_content.get_experiment_config(workers, if_iid, network, algo)['comm_data_scale']
    #                 plot_with_params(network, workers, logs_content.get_experiment_config(workers, if_iid, network, algo)['batch_size']
    #                 , logs_content.get_experiment_config(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
    #                     logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
    #                     force_legend=True,comm_ratio=comm_data_scale*num_comms_epoch, ax=ax)
    #             # # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DensePSGD_name, isacc=True, title='MNIST CNN',
    #             # #     logfile=logs_content.get_log_file(workers, if_iid, network, DensePSGD_name), 
    #             # #     force_legend=True, comm_ratio=comm_data_scale[DensePSGD_name]*num_batches_per_epoch[mnistnet_name], ax=ax)

    #             # # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], TopKPSGD_name, isacc=True, title='MNIST CNN',
    #             # #     logfile=logs_content.get_log_file(workers, if_iid, network, DensePSGD_name), 
    #             # #     force_legend=True, comm_ratio=comm_data_scale[TopKPSGD_name]*num_batches_per_epoch[mnistnet_name], ax=ax)

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[FedAvg_name]*1, ax=ax) # scale means random selection in FedAvg

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_sparse_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[FedAvg_sparse_name]*1, ax=ax)

    #             # # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='MNIST CNN',
    #             # #     logfile='logs/gpuhomedclogs_before/logs/baseline-OURS-wait-dc1-model-dc/mnistflnet-n32-bs50-lr0.0500/gpu11-20111.log', 
    #             # #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[mnistnet_name])
    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DPSGD_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[mnistnet_name], ax=ax)

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DCD_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DCD_name]*num_batches_per_epoch[mnistnet_name], ax=ax)

    #             # # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='MNIST CNN',
    #             # #     logfile='logs/gpuhomedclogs_before/logs/compression-OURS-wait-dc1-model-dc/mnistflnet-n32-bs50-lr0.0500-s0.99000/gpu13-20111.log', 
    #             # #     force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[mnistnet_name], ax=ax)
    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, SAPS_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[mnistnet_name], ax=ax)

    #     if network == cifar10flnet_name:
    #         if if_iid :
    #             for algo in Algorithms[2:7]:
    #                 num_comms_epoch = logs_content.get_experiment_config(workers, if_iid, network, algo)['num_comms_epoch']
    #                 comm_data_scale = logs_content.get_experiment_config(workers, if_iid, network, algo)['comm_data_scale']
    #                 plot_with_params(network, workers, logs_content.get_experiment_config(workers, if_iid, network, algo)['batch_size']
    #                 , logs_content.get_experiment_config(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
    #                     logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
    #                     force_legend=True,comm_ratio=comm_data_scale*num_comms_epoch, ax=ax)                
    #             # # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'],  DensePSGD_name, isacc=True, title='CIFAR10 CNN',
    #             # #     logfile=logs_content.get_log_file(workers, if_iid, network, DensePSGD_name), 
    #             # #     force_legend=True,comm_ratio=comm_data_scale[DensePSGD_name]*num_batches_per_epoch[cifar10flnet_name], ax=ax)

    #             # # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'],  TopKPSGD_name, isacc=True, title='CIFAR10 CNN',
    #             # #     logfile=logs_content.get_log_file(workers, if_iid, network, DensePSGD_name), 
    #             # #     force_legend=True, comm_ratio=comm_data_scale[TopKPSGD_name]*num_batches_per_epoch[cifar10flnet_name], ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[FedAvg_name]*1, ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_sparse_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[FedAvg_sparse_name]*1, ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'],DPSGD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DPSGD_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[cifar10flnet_name], ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DCD_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DCD_name]*num_batches_per_epoch[cifar10flnet_name], ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, SAPS_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[cifar10flnet_name], ax=ax)


    #     if network == resnet20_name:
    #         if if_iid :
    #             for algo in Algorithms[2:7]:
    #                 num_comms_epoch = logs_content.get_experiment_config(workers, if_iid, network, algo)['num_comms_epoch']
    #                 comm_data_scale = logs_content.get_experiment_config(workers, if_iid, network, algo)['comm_data_scale']
    #                 plot_with_params(network, workers, logs_content.get_experiment_config(workers, if_iid, network, algo)['batch_size']
    #                 , logs_content.get_experiment_config(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
    #                     logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
    #                     force_legend=True,comm_ratio=comm_data_scale*num_comms_epoch, ax=ax)
    #             # # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'],  DensePSGD_name, isacc=True, title='ResNet-20',
    #             # #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_sparse_name), 
    #             # #     force_legend=True,  comm_ratio=comm_data_scale[DensePSGD_name]*num_batches_per_epoch[resnet20_name], ax=ax)

    #             # # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], TopKPSGD_name, isacc=True, title='ResNet-20',
    #             # #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_sparse_name), 
    #             # #     force_legend=True,  comm_ratio=comm_data_scale[TopKPSGD_name]*num_batches_per_epoch[resnet20_name], ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='ResNet-20',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[FedAvg_name]*1, ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='ResNet-20',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_sparse_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[FedAvg_sparse_name]*1, ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='ResNet-20',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DPSGD_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[resnet20_name], ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='ResNet-20',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DCD_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DCD_name]*num_batches_per_epoch[resnet20_name], ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='ResNet-20',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, SAPS_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[resnet20_name], ax=ax)

    # if workers =='14':    
    #     if network == mnistnet_name:
    #         if if_iid :
    #             for algo in Algorithms[2:7]:
    #                 num_comms_epoch = logs_content.get_experiment_config(workers, if_iid, network, algo)['num_comms_epoch']
    #                 comm_data_scale = logs_content.get_experiment_config(workers, if_iid, network, algo)['comm_data_scale']
    #                 plot_with_params(network, workers, logs_content.get_experiment_config(workers, if_iid, network, algo)['batch_size']
    #                 , logs_content.get_experiment_config(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
    #                     logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
    #                     force_legend=True,comm_ratio=comm_data_scale*num_comms_epoch, ax=ax)
    #             print("222")
    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='MNIST CNN',
    #             #     logfile='logs/scilogs0112/baseline-wait-dc1-model-fl/mnistflnet-n14-bs50-lr0.0500/scigpu11-29223.log',
    #             #     force_legend=True, comm_ratio=comm_data_scale[FedAvg_name]*1, ax=ax) # scale means random selection in FedAvg

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='MNIST CNN',
    #             #     logfile='logs/scilogs0112/compression-gwarmup-wait-dc1-model-fl/mnistflnet-n14-bs50-lr0.0500-s0.99000/scigpu11-29223.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[FedAvg_sparse_name]*1, ax=ax)

    #             # # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='MNIST CNN',
    #             # #     logfile='logs/hsw224_14workerDPSGD/logs/baseline-OURS-wait-dc1-model-dc/mnistflnet-n14-bs50-lr0.0500/hsw224-20111.log', 
    #             # #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[mnistnet_name])
    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='MNIST CNN',
    #             #     logfile='logs/hsw224011222/logs/baseline-OURS-NoAsk-TOPO-RING-wait-dc1-model-dc/mnistflnet-n14-bs50-lr0.0500/hsw224-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[mnistnet_name], ax=ax)

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='MNIST CNN',
    #             #     logfile='logs/gpuhomeDCD14worker/mnistflnet-n14-bs50-lr0.0500-s0.75000/gpu17-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DCD_name]*num_batches_per_epoch[mnistnet_name], ax=ax)

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='MNIST CNN',
    #             #     logfile='logs/hsw-eval-ours-logs/compression-OURS-NoAsk-TOPO-REAL-RealCommEval-lb-1.45-k-0.1-wait-dc1-model-dc/mnistflnet-n14-bs50-lr0.0500-s0.99000/hsw224-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[mnistnet_name], ax=ax)

    #     if network == cifar10flnet_name:
    #         if if_iid :
    #             for algo in Algorithms[2:7]:
    #                 num_comms_epoch = logs_content.get_experiment_config(workers, if_iid, network, algo)['num_comms_epoch']
    #                 comm_data_scale = logs_content.get_experiment_config(workers, if_iid, network, algo)['comm_data_scale']
    #                 plot_with_params(network, workers, logs_content.get_experiment_config(workers, if_iid, network, algo)['batch_size']
    #                 , logs_content.get_experiment_config(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
    #                     logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
    #                     force_legend=True,comm_ratio=comm_data_scale*num_comms_epoch, ax=ax)                
    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile='logs/scilogs0112/baseline-wait-dc1-model-fl/cifar10flnet-n14-bs100-lr0.0400/scigpu11-29223.log',
    #             #     force_legend=True, comm_ratio=comm_data_scale[FedAvg_name]*1, ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile='logs/scilogs0112/compression-gwarmup-wait-dc1-model-fl/cifar10flnet-n14-bs100-lr0.0400-s0.99000/scigpu11-29223.log',
    #             #     force_legend=True, comm_ratio=comm_data_scale[FedAvg_sparse_name]*1, ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile='logs/hsw224011222/logs/baseline-OURS-NoAsk-TOPO-RING-wait-dc1-model-dc/cifar10flnet-n14-bs100-lr0.0400/hsw224-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[cifar10flnet_name], ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile='logs/gpuhomeDCD14worker/cifar10flnet-n14-bs100-lr0.0400-s0.75000/gpu17-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DCD_name]*num_batches_per_epoch[cifar10flnet_name], ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile='logs/hsw-eval-ours-logs/compression-OURS-NoAsk-TOPO-REAL-RealCommEval-lb-1.45-k-0.1-wait-dc1-model-dc/cifar10flnet-n14-bs100-lr0.0400-s0.99000/hsw224-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[cifar10flnet_name], ax=ax)

    #     if network == resnet20_name:
    #         if if_iid :
    #             for algo in Algorithms[2:7]:
    #                 num_comms_epoch = logs_content.get_experiment_config(workers, if_iid, network, algo)['num_comms_epoch']
    #                 comm_data_scale = logs_content.get_experiment_config(workers, if_iid, network, algo)['comm_data_scale']
    #                 plot_with_params(network, workers, logs_content.get_experiment_config(workers, if_iid, network, algo)['batch_size']
    #                 , logs_content.get_experiment_config(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
    #                     logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
    #                     force_legend=True,comm_ratio=comm_data_scale*num_comms_epoch, ax=ax)                
    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='ResNet-20',
    #             #     logfile='logs/hsw-fl-logs0112/baseline-wait-dc1-model-fl/resnet20-n14-bs64-lr0.1000/hsw224-8923.log',
    #             #     force_legend=True, comm_ratio=comm_data_scale[FedAvg_name], ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='ResNet-20',
    #             #     logfile='logs/hsw-fl-logs0112/compression-gwarmup-wait-dc1-model-fl/resnet20-n14-bs64-lr0.1000-s0.99000/hsw224-8923.log',
    #             #     force_legend=True, comm_ratio=comm_data_scale[FedAvg_sparse_name]*1, ax=ax)

    #             # # plot_with_params(resnet20_name, workers, 64, 0.1, DPSGD_name, isacc=True, title='ResNet-20',
    #             # #     logfile='logs/hsw224_14workerDPSGD/logs/baseline-OURS-wait-dc1-model-dc/resnet20-n14-bs64-lr0.1000/hsw224-20111.log', 
    #             # #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[resnet20_name])
    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='ResNet-20',
    #             #     logfile='logs/hsw224011222/logs/baseline-OURS-NoAsk-TOPO-RING-wait-dc1-model-dc/resnet20-n14-bs64-lr0.1000/hsw224-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[resnet20_name], ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='ResNet-20',
    #             #     logfile='logs/gpuhomeDCD14worker/resnet20-n14-bs64-lr0.1000-s0.75000/gpu17-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DCD_name]*num_batches_per_epoch[resnet20_name], ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='ResNet-20',
    #             #     logfile='logs/hsw-eval-ours-logs/compression-OURS-NoAsk-TOPO-REAL-RealCommEval-lb-1.45-k-0.1-wait-dc1-model-dc/resnet20-n14-bs64-lr0.1000-s0.99000/hsw224-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[resnet20_name], ax=ax)




def ICNP2020_SAPS_communication_time_iteration(network, workers, sparsity=None, if_iid=True, ax=None):
    FedAvg_name = 'FedAvg'
    FedAvg_sparse_name = 'S-FedAvg'
    DPSGD_name = 'APS-FL'
    DCD_name = 'DCD-PSGD'
    SAPS_name = 'SAPS-FL'
    DensePSGD_name = 'PSGD'
    TopKPSGD_name = 'TopK-PSGD'
    DPSGD_FIX_name = 'D-PSGD'

    mnistnet_name = 'mnistCNN'
    cifar10flnet_name = 'cifar10CNN'
    resnet20_name = 'resnet20'
    # comm_bandwidth = {}
    # num_batches_per_epoch = {
    #     'mnistCNN': 1+int(50000/(int(workers)*50)),
    #     'cifar10CNN': 1+int(50000/(int(workers)*100)),
    #     'resnet20': 1+int(50000/(int(workers)*64))
    # }

    # comm_data_scale = {
    #     'FedAvg': 2,
    #     'S-FedAvg': 1+1*(1-0.99),
    #     'APSD-PSGD': 2,
    #     'D-PSGD': 4,
    #     'DCD-PSGD': 4*0.75,
    #     'SAPS-PSGD': 2*(1-0.99),
    #     'PSGD': 2,
    #     'TopK-PSGD': 2*(int(workers)-1)*(1-0.999)
    # }
    # prefix = 'baseline-gwarmup-wait-noniid-dc1-model-fl'

    workers = workers
    Bandwidth_Decrease_Factor = 0.1
    CommunicationIerations = 500

    if workers == '32':
        bandwidth = minmax_communication_cost.random_bandwidth(32)
    elif workers == '14':
        bandwidth = minmax_communication_cost.fixed_bandwidth()

    comm_bandwidth_list = {
        'FedAvg': [],
        'S-FedAvg': [],
        'APS-FL': [],
        'D-PSGD': [],
        'DCD-PSGD': [],
        'SAPS-FL': [],
        'PSGD':[],
        'TopK-PSGD':[]
    }

    comm_bandwidth_avg = {
        'FedAvg': 0,
        'S-FedAvg': 0,
        'APS-FL': 0,
        'D-PSGD': 0,
        'DCD-PSGD': 0,
        'SAPS-FL': 0,
        'PSGD':0,
        'TopK-PSGD':0
    }

    # bandwidth_weighted = copy.deepcopy(bandwidth)
    # limited_bandwidth = minmax_communication_cost.get_limited_graph(bandwidth, 1.45)
    # limited_bandwidth_weighted = copy.deepcopy(limited_bandwidth)
    #print("limited_bandwidth:",limited_bandwidth)
    # for iteration in range(CommunicationIerations):
    #     index_roles = np.arange(int(workers))
    #     np.random.shuffle(index_roles)
    #     roles = np.ones(int(workers))
    #     roles[index_roles[0:int(int(workers)/2)]]=0
    #     _, random_bandwidth_threshold = minmax_communication_cost.get_random_match_and_bandwidth(bandwidth, int(workers))
    #     _, ring_bandwidth_threshold = minmax_communication_cost.get_ring_match_and_bandwidth(bandwidth, int(workers))
    #     _, FedAVG_bandwidth_threshold = minmax_communication_cost.get_FedAVG_match_and_bandwidth(bandwidth, int(workers))
    #     preparation_match, preparation_bandwidth_threshold = minmax_communication_cost.minmax_communication_match(limited_bandwidth_weighted, roles, 1)
    #     preparation_bandwidth_threshold = 1000000000
    #     # print(limited_bandwidth_weighted)
    #     for i in range(int(workers)):
    #         limited_bandwidth_weighted[i][preparation_match[i]] = limited_bandwidth_weighted[i][preparation_match[i]] - \
    #                     limited_bandwidth[i][preparation_match[i]] * Bandwidth_Decrease_Factor 
    #         # self.communication_record[rank][match[host]] += 1
    #         if preparation_bandwidth_threshold > limited_bandwidth[i][preparation_match[i]]:
    #             preparation_bandwidth_threshold = limited_bandwidth[i][preparation_match[i]]

    #     if iteration % 500 == 0:

    #         print("===========iteration : %d================="%(iteration))
    #         # print("random  bandwidth_threshold", random_bandwidth_threshold)
    #         print("ring  bandwidth_threshold", ring_bandwidth_threshold)
    #         print("FedAVG  bandwidth_threshold", FedAVG_bandwidth_threshold)
    #         print("preparation_match   bandwidth_threshold", preparation_bandwidth_threshold)

    #     comm_bandwidth_list[FedAvg_name].append(FedAVG_bandwidth_threshold)
    #     comm_bandwidth_list[FedAvg_sparse_name].append(FedAVG_bandwidth_threshold)
    #     comm_bandwidth_list[DPSGD_name].append(ring_bandwidth_threshold)
    #     comm_bandwidth_list[DCD_name].append(ring_bandwidth_threshold)
    #     comm_bandwidth_list[SAPS_name].append(preparation_bandwidth_threshold)
    #     comm_bandwidth_list[DensePSGD_name].append(ring_bandwidth_threshold)
    #     comm_bandwidth_list[TopKPSGD_name].append(ring_bandwidth_threshold)

    #     comm_bandwidth_avg[FedAvg_name] += FedAVG_bandwidth_threshold
    #     comm_bandwidth_avg[FedAvg_sparse_name] += FedAVG_bandwidth_threshold
    #     comm_bandwidth_avg[DPSGD_name] += ring_bandwidth_threshold
    #     comm_bandwidth_avg[DCD_name] += ring_bandwidth_threshold
    #     comm_bandwidth_avg[SAPS_name] += preparation_bandwidth_threshold
    #     comm_bandwidth_avg[DensePSGD_name] += ring_bandwidth_threshold
    #     comm_bandwidth_avg[TopKPSGD_name] += ring_bandwidth_threshold
    # ring : 0.152010572108
    # fedavg : 0.740592900747
    if workers == '32':
        comm_bandwidth_avg[FedAvg_name] = 0.740592900747#/float(workers)
        comm_bandwidth_avg[FedAvg_sparse_name] = 0.740592900747#/float(workers)
        comm_bandwidth_avg[DPSGD_FIX_name] = 0.152010572108 # 'D-PSGD'
        comm_bandwidth_avg[DPSGD_name] = 4.0508065457914935 #  'APSD-PSGD'
        comm_bandwidth_avg[DCD_name] = 0.152010572108
        comm_bandwidth_avg[SAPS_name] = 4.0508065457914935
        comm_bandwidth_avg[DensePSGD_name] = 0.152010572108
        comm_bandwidth_avg[TopKPSGD_name] = 0.152010572108
    elif workers == '14':       
        # comm_bandwidth_avg[FedAvg_name] = comm_bandwidth_avg[FedAvg_name] / CommunicationIerations
        # comm_bandwidth_avg[FedAvg_sparse_name] = comm_bandwidth_avg[FedAvg_sparse_name] / CommunicationIerations
        # comm_bandwidth_avg[DPSGD_name] = comm_bandwidth_avg[DPSGD_name] / CommunicationIerations
        # comm_bandwidth_avg[DCD_name] = comm_bandwidth_avg[DCD_name] / CommunicationIerations
        # comm_bandwidth_avg[SAPS_name] = comm_bandwidth_avg[SAPS_name] / CommunicationIerations
        # comm_bandwidth_avg[DensePSGD_name] = comm_bandwidth_avg[DensePSGD_name] / CommunicationIerations
        # comm_bandwidth_avg[TopKPSGD_name] = comm_bandwidth_avg[TopKPSGD_name] / CommunicationIerations
        comm_bandwidth_avg[FedAvg_name] = 0.963438976154721#/float(workers)
        comm_bandwidth_avg[FedAvg_sparse_name] = 0.963438976154721#/float(workers)
        comm_bandwidth_avg[DPSGD_FIX_name] = 0.026541015899914023 # 'D-PSGD'
        comm_bandwidth_avg[DPSGD_name] = 4.0508065457914935  #  'APSD-PSGD'
        comm_bandwidth_avg[DCD_name] = 0.026541015899914023
        comm_bandwidth_avg[SAPS_name] = 4.0508065457914935  #    B_thres=4.000000, T_thres=10.000000
        comm_bandwidth_avg[DensePSGD_name] = 0.026541015899914023
        comm_bandwidth_avg[TopKPSGD_name] = 0.026541015899914023


    print("===========average=================\n")
    # print("random  bandwidth_threshold", random_bandwidth_threshold)
    print("ring  bandwidth_threshold", comm_bandwidth_avg[FedAvg_name])
    print("FedAVG  bandwidth_threshold", comm_bandwidth_avg[DPSGD_name])
    print("preparation_match   bandwidth_threshold", comm_bandwidth_avg[SAPS_name])

    Algorithms = ['PSGD', 'TopK-PSGD', 'FedAvg', 'S-FedAvg', 'D-PSGD', 'DCD-PSGD', 'APS-FL', 'SAPS-FL']
    for algo in Algorithms[2:8]:
        num_comms_epoch = logs_content.get_experiment_config(workers, if_iid, network, algo)['num_comms_epoch']
        comm_data_scale = logs_content.get_experiment_config(workers, if_iid, network, algo)['comm_data_scale']
        plot_with_params(network, workers, logs_content.get_experiment_config(workers, if_iid, network, algo)['batch_size']
        , logs_content.get_experiment_config(workers, if_iid, network, algo)['lr'], algo, isacc=True, title='MNIST CNN',
            logfile=logs_content.get_log_file(workers, if_iid, network, algo), 
            force_legend=True,comm_ratio=comm_data_scale*num_comms_epoch, comm_bandwidth=comm_bandwidth_avg[algo], ax=ax)     

    # if workers =='32':    
    #     if network == mnistnet_name:
    #         if if_iid :
    #             print("222")

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DensePSGD_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DensePSGD_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DensePSGD_name]*num_batches_per_epoch[mnistnet_name], comm_bandwidth=comm_bandwidth_avg[DensePSGD_name], ax=ax)

    #             # plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             #, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], TopKPSGD_name, isacc=True, title='MNIST CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, TopKPSGD_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[TopKPSGD_name]*num_batches_per_epoch[mnistnet_name], comm_bandwidth=comm_bandwidth_avg[TopKPSGD_name], ax=ax)

    #             plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='MNIST CNN',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[FedAvg_name]*1, comm_bandwidth=comm_bandwidth_avg[FedAvg_name], ax=ax) # scale means random selection in FedAvg

    #             plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='MNIST CNN',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_sparse_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[FedAvg_sparse_name]*1, comm_bandwidth=comm_bandwidth_avg[FedAvg_sparse_name], ax=ax)

    #             plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='MNIST CNN',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, DPSGD_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[mnistnet_name], comm_bandwidth=comm_bandwidth_avg[DPSGD_name], ax=ax)

    #             plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='MNIST CNN',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, DCD_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[DCD_name]*num_batches_per_epoch[mnistnet_name], comm_bandwidth=comm_bandwidth_avg[DCD_name], ax=ax)

    #             plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='MNIST CNN',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, SAPS_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[mnistnet_name], comm_bandwidth=comm_bandwidth_avg[SAPS_name], ax=ax)

    #     if network == cifar10flnet_name:
    #         if if_iid :
    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'],  DensePSGD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DensePSGD_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DensePSGD_name]*num_batches_per_epoch[cifar10flnet_name], comm_bandwidth=comm_bandwidth_avg[DensePSGD_name], ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'],  TopKPSGD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, TopKPSGD_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[TopKPSGD_name]*num_batches_per_epoch[cifar10flnet_name], comm_bandwidth=comm_bandwidth_avg[TopKPSGD_name], ax=ax)

    #             plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='CIFAR10 CNN',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[FedAvg_name]*1, comm_bandwidth=comm_bandwidth_avg[FedAvg_name], ax=ax)

    #             plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='CIFAR10 CNN',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_sparse_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[FedAvg_sparse_name]*1, comm_bandwidth=comm_bandwidth_avg[FedAvg_sparse_name], ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile='logs/hsw-logs/hsw-dc-logs-before0109/logs/baseline-OURS-wait-dc1-model-dc/cifar10flnet-n32-bs100-lr0.0400/hsw224-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[cifar10flnet_name], comm_bandwidth=comm_bandwidth_avg[DPSGD_name])
    #             plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='CIFAR10 CNN',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, DPSGD_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[cifar10flnet_name], comm_bandwidth=comm_bandwidth_avg[DPSGD_name], ax=ax)

    #             plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='CIFAR10 CNN',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, DCD_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[DCD_name]*num_batches_per_epoch[cifar10flnet_name], comm_bandwidth=comm_bandwidth_avg[DCD_name], ax=ax)

    #             plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='CIFAR10 CNN',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, SAPS_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[cifar10flnet_name], comm_bandwidth=comm_bandwidth_avg[SAPS_name], ax=ax)

    #     if network == resnet20_name:
    #         if if_iid :

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'],  DensePSGD_name, isacc=True, title='ResNet-20',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, DensePSGD_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DensePSGD_name]*num_batches_per_epoch[resnet20_name], comm_bandwidth=comm_bandwidth_avg[DensePSGD_name], ax=ax)

    #             # plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             # , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], TopKPSGD_name, isacc=True, title='ResNet-20',
    #             #     logfile=logs_content.get_log_file(workers, if_iid, network, TopKPSGD_name), 
    #             #     force_legend=True, comm_ratio=comm_data_scale[TopKPSGD_name]*num_batches_per_epoch[resnet20_name], comm_bandwidth=comm_bandwidth_avg[TopKPSGD_name], ax=ax)

    #             plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='ResNet-20',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[FedAvg_name]*1, comm_bandwidth=comm_bandwidth_avg[FedAvg_name], ax=ax)

    #             plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='ResNet-20',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, FedAvg_sparse_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[FedAvg_sparse_name]*1, comm_bandwidth=comm_bandwidth_avg[FedAvg_sparse_name], ax=ax)

    #             # plot_with_params(resnet20_name, workers, 64, 0.07, DPSGD_name, isacc=True, title='ResNet-20',
    #             #     logfile='logs/hsw-logs/hsw-dc-logs-before0109/logs/baseline-OURS-wait-dc1-model-dc/resnet20-n32-bs64-lr0.2000/hsw224-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[resnet20_name], comm_bandwidth=comm_bandwidth_avg[DPSGD_name])
    #             plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='ResNet-20',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, DPSGD_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[resnet20_name], comm_bandwidth=comm_bandwidth_avg[DPSGD_name], ax=ax)

    #             plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='ResNet-20',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, DCD_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[DCD_name]*num_batches_per_epoch[resnet20_name], comm_bandwidth=comm_bandwidth_avg[DCD_name], ax=ax)

    #             # plot_with_params(resnet20_name, workers, 64, 0.1, SAPS_name, isacc=True, title='ResNet-20',
    #             #     logfile='logs/gpuhomedclogs_before/logs/compression-OURS-wait-dc1-model-dc/resnet20-n32-bs64-lr0.1000-s0.99000/gpu13-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[resnet20_name], comm_bandwidth=comm_bandwidth_avg[SAPS_name], ax=ax)
    #             plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='ResNet-20',
    #                 logfile=logs_content.get_log_file(workers, if_iid, network, SAPS_name), 
    #                 force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[resnet20_name], comm_bandwidth=comm_bandwidth_avg[SAPS_name], ax=ax)




    # if workers =='14':    
    #     if network == mnistnet_name:
    #         if if_iid :
    #             print("222")
    #             plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='MNIST CNN',
    #                 logfile='logs/scilogs0112/baseline-wait-dc1-model-fl/mnistflnet-n14-bs50-lr0.0500/scigpu11-29223.log',
    #                 force_legend=True, comm_ratio=comm_data_scale[FedAvg_name]*1, comm_bandwidth=comm_bandwidth_avg[FedAvg_name], ax=ax) # scale means random selection in FedAvg

    #             plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='MNIST CNN',
    #                 logfile='logs/scilogs0112/compression-gwarmup-wait-dc1-model-fl/mnistflnet-n14-bs50-lr0.0500-s0.99000/scigpu11-29223.log', 
    #                 force_legend=True, comm_ratio=comm_data_scale[FedAvg_sparse_name]*1, comm_bandwidth=comm_bandwidth_avg[FedAvg_sparse_name], ax=ax)

    #             # plot_with_params(mnistnet_name, workers, 50, 0.1, DPSGD_name, isacc=True, title='MNIST CNN',
    #             #     logfile='logs/hsw224_14workerDPSGD/logs/baseline-OURS-wait-dc1-model-dc/mnistflnet-n14-bs50-lr0.0500/hsw224-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[mnistnet_name], comm_bandwidth=comm_bandwidth_avg[DPSGD_name])
    #             plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='MNIST CNN',
    #                 logfile='logs/hsw224011222/logs/baseline-OURS-NoAsk-TOPO-RING-wait-dc1-model-dc/mnistflnet-n14-bs50-lr0.0500/hsw224-20111.log', 
    #                 force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[mnistnet_name], comm_bandwidth=comm_bandwidth_avg[DPSGD_name], ax=ax)

    #             plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='MNIST CNN',
    #                 logfile='logs/gpuhomeDCD14worker/mnistflnet-n14-bs50-lr0.0500-s0.75000/gpu17-20111.log', 
    #                 force_legend=True, comm_ratio=comm_data_scale[DCD_name]*num_batches_per_epoch[mnistnet_name], comm_bandwidth=comm_bandwidth_avg[DCD_name], ax=ax)

    #             plot_with_params(mnistnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='MNIST CNN',
    #                 logfile='logs/hsw-eval-ours-logs/compression-OURS-NoAsk-TOPO-REAL-RealCommEval-lb-1.45-k-0.1-wait-dc1-model-dc/mnistflnet-n14-bs50-lr0.0500-s0.99000/hsw224-20111.log', 
    #                 force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[mnistnet_name], comm_bandwidth=comm_bandwidth_avg[SAPS_name], ax=ax)

    #     if network == cifar10flnet_name:
    #         if if_iid :
    #             plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='CIFAR10 CNN',
    #                 logfile='logs/scilogs0112/baseline-wait-dc1-model-fl/cifar10flnet-n14-bs100-lr0.0400/scigpu11-29223.log',
    #                 force_legend=True, comm_ratio=comm_data_scale[FedAvg_name]*1, comm_bandwidth=comm_bandwidth_avg[FedAvg_name], ax=ax)

    #             plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='CIFAR10 CNN',
    #                 logfile='logs/scilogs0112/compression-gwarmup-wait-dc1-model-fl/cifar10flnet-n14-bs100-lr0.0400-s0.99000/scigpu11-29223.log',
    #                 force_legend=True, comm_ratio=comm_data_scale[FedAvg_sparse_name]*1, comm_bandwidth=comm_bandwidth_avg[FedAvg_sparse_name], ax=ax)

    #             # plot_with_params(cifar10flnet_name, workers, 100, 0.04, DPSGD_name, isacc=True, title='CIFAR10 CNN',
    #             #     logfile='logs/hsw224_14workerDPSGD/logs/baseline-OURS-wait-dc1-model-dc/cifar10flnet-n14-bs100-lr0.0400/hsw224-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[cifar10flnet_name], comm_bandwidth=comm_bandwidth_avg[DPSGD_name])
    #             plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='CIFAR10 CNN',
    #                 logfile='logs/hsw224011222/logs/baseline-OURS-NoAsk-TOPO-RING-wait-dc1-model-dc/cifar10flnet-n14-bs100-lr0.0400/hsw224-20111.log', 
    #                 force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[cifar10flnet_name], comm_bandwidth=comm_bandwidth_avg[DPSGD_name], ax=ax)

    #             plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='CIFAR10 CNN',
    #                 logfile='logs/gpuhomeDCD14worker/cifar10flnet-n14-bs100-lr0.0400-s0.75000/gpu17-20111.log', 
    #                 force_legend=True, comm_ratio=comm_data_scale[DCD_name]*num_batches_per_epoch[cifar10flnet_name], comm_bandwidth=comm_bandwidth_avg[DCD_name], ax=ax)

    #             plot_with_params(cifar10flnet_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='CIFAR10 CNN',
    #                 logfile='logs/hsw-eval-ours-logs/compression-OURS-NoAsk-TOPO-REAL-RealCommEval-lb-1.45-k-0.1-wait-dc1-model-dc/cifar10flnet-n14-bs100-lr0.0400-s0.99000/hsw224-20111.log', 
    #                 force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[cifar10flnet_name], comm_bandwidth=comm_bandwidth_avg[SAPS_name], ax=ax)

    #     if network == resnet20_name:
    #         if if_iid :
    #             plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_name, isacc=True, title='ResNet-20',
    #                 logfile='logs/hsw-fl-logs0112/baseline-wait-dc1-model-fl/resnet20-n14-bs64-lr0.1000/hsw224-8923.log',
    #                 force_legend=True, comm_ratio=comm_data_scale[FedAvg_name]*1, comm_bandwidth=comm_bandwidth_avg[FedAvg_name], ax=ax)

    #             plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], FedAvg_sparse_name, isacc=True, title='ResNet-20',
    #                 logfile='logs/hsw-fl-logs0112/compression-gwarmup-wait-dc1-model-fl/resnet20-n14-bs64-lr0.1000-s0.99000/hsw224-8923.log',
    #                 force_legend=True, comm_ratio=comm_data_scale[FedAvg_sparse_name]*1, comm_bandwidth=comm_bandwidth_avg[FedAvg_sparse_name], ax=ax)

    #             # plot_with_params(resnet20_name, workers, 64, 0.1, DPSGD_name, isacc=True, title='ResNet-20',
    #             #     logfile='logs/hsw224_14workerDPSGD/logs/baseline-OURS-wait-dc1-model-dc/resnet20-n14-bs64-lr0.1000/hsw224-20111.log', 
    #             #     force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[resnet20_name], comm_bandwidth=comm_bandwidth_avg[DPSGD_name])
    #             plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DPSGD_name, isacc=True, title='ResNet-20',
    #                 logfile='logs/hsw224011222/logs/baseline-OURS-NoAsk-TOPO-RING-wait-dc1-model-dc/resnet20-n14-bs64-lr0.1000/hsw224-20111.log', 
    #                 force_legend=True, comm_ratio=comm_data_scale[DPSGD_name]*num_batches_per_epoch[resnet20_name], comm_bandwidth=comm_bandwidth_avg[DPSGD_name], ax=ax)

    #             plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], DCD_name, isacc=True, title='ResNet-20',
    #                 logfile='logs/gpuhomeDCD14worker/resnet20-n14-bs64-lr0.1000-s0.75000/gpu17-20111.log', 
    #                 force_legend=True, comm_ratio=comm_data_scale[DCD_name]*num_batches_per_epoch[resnet20_name], comm_bandwidth=comm_bandwidth_avg[DCD_name], ax=ax)

    #             plot_with_params(resnet20_name, workers, logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['batch_size']
    #             , logs_content.get_log_file(workers, if_iid, network, DensePSGD_name)['lr'], SAPS_name, isacc=True, title='ResNet-20',
    #                 logfile='logs/hsw-eval-ours-logs/compression-OURS-NoAsk-TOPO-REAL-RealCommEval-lb-1.45-k-0.1-wait-dc1-model-dc/resnet20-n14-bs64-lr0.1000-s0.99000/hsw224-20111.log', 
    #                 force_legend=True, comm_ratio=comm_data_scale[SAPS_name]*num_batches_per_epoch[resnet20_name], comm_bandwidth=comm_bandwidth_avg[SAPS_name], ax=ax)




def get_ring_and_fedavg_bandwidth_avg():
    ring_list = []
    fedavg_list = []
    ring_avg = 0
    fedavg_avg = 0

    workers = '32'
    Bandwidth_Decrease_Factor = 0.1
    CommunicationIerations = 5000

    for iteration in range(CommunicationIerations):
        if workers == '32':
            bandwidth = minmax_communication_cost.random_bandwidth(32)
        bandwidth_weighted = copy.deepcopy(bandwidth)
        limited_bandwidth = minmax_communication_cost.get_limited_graph(bandwidth, 1.45)
        limited_bandwidth_weighted = copy.deepcopy(limited_bandwidth)
    #print("limited_bandwidth:",limited_bandwidth)

        index_roles = np.arange(int(workers))
        np.random.shuffle(index_roles)
        roles = np.ones(int(workers))
        roles[index_roles[0:int(int(workers)/2)]]=0

        _, ring_bandwidth_threshold = minmax_communication_cost.get_ring_match_and_bandwidth(bandwidth, int(workers))
        _, FedAVG_bandwidth_threshold = minmax_communication_cost.get_FedAVG_match_and_bandwidth(bandwidth, int(workers))

        if iteration % 500 == 0:

            print("===========iteration : %d================="%(iteration))
            # print("random  bandwidth_threshold", random_bandwidth_threshold)
            print("ring  bandwidth_threshold", ring_bandwidth_threshold)
            print("FedAVG  bandwidth_threshold", FedAVG_bandwidth_threshold)

        ring_list.append(ring_bandwidth_threshold)
        fedavg_list.append(FedAVG_bandwidth_threshold)

        ring_avg += ring_bandwidth_threshold
        fedavg_avg += FedAVG_bandwidth_threshold

    ring_avg = ring_avg / CommunicationIerations
    fedavg_avg = fedavg_avg / CommunicationIerations

    print("===========average=================\n")
    # print("random  bandwidth_threshold", random_bandwidth_threshold)
    print("ring  bandwidth_threshold", ring_avg)
    print("FedAVG  bandwidth_threshold", fedavg_avg)
    # ring : 0.152010572108
    # fedavg : 0.740592900747
    return ring_list, fedavg_list, ring_avg, fedavg_avg


def ICNP2020_bandwidth(network, workers, sparsity=None, if_iid=True, ax=None):
    FedAvg_name = 'FedAvg'
    FedAvg_sparse_name = 'S-FedAvg'
    DPSGD_name = 'D-PSGD'
    DCD_name = 'DCD-PSGD'
    SAPS_name = 'SAPS-FL'
    RamdomChoose = 'RandomChoose'
    mnistnet_name = 'mnistCNN'
    cifar10flnet_name = 'cifar10CNN'
    resnet20_name = 'resnet20'
    # comm_bandwidth = {}
    num_batches_per_epoch = {
        'mnistCNN': 1+int(50000/(int(workers)*50)),
        'cifar10CNN': 1+int(50000/(int(workers)*100)),
        'resnet20': 1+int(50000/(int(workers)*64))
    }
    comm_data_scale = {
        'FedAvg': 2,
        'S-FedAvg': 1+1*(1-0.99),
        'D-PSGD': 4,
        'DCD-PSGD': 4*0.75,
        'SAPS-FL': 2*(1-0.99)
    }
    # prefix = 'baseline-gwarmup-wait-noniid-dc1-model-fl'

    workers = workers
    Bandwidth_Decrease_Factor = 0.1
    #CommunicationIerations = 2000
    CommunicationIerations = 500
    if workers == '32':
        bandwidth = minmax_communication_cost.random_bandwidth(32)
    elif workers == '14':
        bandwidth = minmax_communication_cost.fixed_bandwidth()

    comm_bandwidth_list = {
        'FedAvg': [],
        'S-FedAvg': [],
        'D-PSGD': [],
        'DCD-PSGD': [],
        'SAPS-FL': [],
        'RandomChoose': []
    }

    comm_bandwidth_avg = {
        'FedAvg': 0,
        'S-FedAvg': 0,
        'D-PSGD': 0,
        'DCD-PSGD': 0,
        'SAPS-FL': 0,
        'RandomChoose': 0
    }

    bandwidth_weighted = copy.deepcopy(bandwidth)
    limited_bandwidth = minmax_communication_cost.get_limited_graph(bandwidth, 1.45)
    limited_bandwidth_weighted = copy.deepcopy(limited_bandwidth)
    print("limited_bandwidth:",limited_bandwidth)
    B_thres=4
    T_thres=10
    # B_thres=1.45
    # T_thres=10
    SAPS_gossip1 = minmax_communication_cost.SAPS_gossip(bandwidth, B_thres=B_thres, T_thres=T_thres)

    for iteration in range(CommunicationIerations):
        index_roles = np.arange(int(workers))
        np.random.shuffle(index_roles)
        roles = np.ones(int(workers))
        roles[index_roles[0:int(int(workers)/2)]]=0
        _, random_bandwidth_threshold = minmax_communication_cost.get_random_match_and_bandwidth(bandwidth, int(workers))
        _, ring_bandwidth_threshold = minmax_communication_cost.get_ring_match_and_bandwidth(bandwidth, int(workers))
        _, FedAVG_bandwidth_threshold = minmax_communication_cost.get_FedAVG_match_and_bandwidth(bandwidth, int(workers))
        _, random_bandwidth_threshold = minmax_communication_cost.get_random_match_and_bandwidth(bandwidth, int(workers))
        preparation_match, preparation_bandwidth_threshold = minmax_communication_cost.minmax_communication_match(limited_bandwidth_weighted, roles, 1)
        preparation_bandwidth_threshold = 1000000000
        SAPS_match, SAPS_bandwidth = SAPS_gossip1.generate_match(iteration)
        # print(limited_bandwidth_weighted)
        for i in range(int(workers)):
            limited_bandwidth_weighted[i][preparation_match[i]] = limited_bandwidth_weighted[i][preparation_match[i]] - \
                        limited_bandwidth[i][preparation_match[i]] * Bandwidth_Decrease_Factor 
            # self.communication_record[rank][match[host]] += 1
            if preparation_bandwidth_threshold > limited_bandwidth[i][preparation_match[i]]:
                preparation_bandwidth_threshold = limited_bandwidth[i][preparation_match[i]]

        if iteration % 500 == 0:

            print("===========iteration : %d================="%(iteration))
            # print("random  bandwidth_threshold", random_bandwidth_threshold)
            print("ring  bandwidth_threshold", ring_bandwidth_threshold)
            print("FedAVG  bandwidth_threshold", FedAVG_bandwidth_threshold)
            print("preparation_match   bandwidth_threshold", preparation_bandwidth_threshold)

        comm_bandwidth_list[FedAvg_name].append(FedAVG_bandwidth_threshold)
        comm_bandwidth_list[FedAvg_sparse_name].append(FedAVG_bandwidth_threshold)
        comm_bandwidth_list[DPSGD_name].append(ring_bandwidth_threshold)
        comm_bandwidth_list[DCD_name].append(ring_bandwidth_threshold)
        comm_bandwidth_list[SAPS_name].append(SAPS_bandwidth)
        comm_bandwidth_list[RamdomChoose].append(random_bandwidth_threshold)

        comm_bandwidth_avg[FedAvg_name] += FedAVG_bandwidth_threshold
        comm_bandwidth_avg[FedAvg_sparse_name] += FedAVG_bandwidth_threshold
        comm_bandwidth_avg[DPSGD_name] += ring_bandwidth_threshold
        comm_bandwidth_avg[DCD_name] += ring_bandwidth_threshold
        comm_bandwidth_avg[SAPS_name] += SAPS_bandwidth
        comm_bandwidth_avg[RamdomChoose] += random_bandwidth_threshold
    # ring : 0.152010572108
    # fedavg : 0.740592900747
    # ours : 1.59878346991
    if workers == '32':
        comm_bandwidth_avg[FedAvg_name] = 0.740592900747
        comm_bandwidth_avg[FedAvg_sparse_name] = 0.740592900747
        comm_bandwidth_avg[DPSGD_name] = 0.152010572108
        comm_bandwidth_avg[DCD_name] = 0.152010572108
        comm_bandwidth_avg[SAPS_name] = comm_bandwidth_avg[SAPS_name] / CommunicationIerations
        # comm_bandwidth_avg[DensePSGD_name] = 0.152010572108
        # comm_bandwidth_avg[TopKPSGD_name] = 0.152010572108
        comm_bandwidth_avg[RamdomChoose] = comm_bandwidth_avg[RamdomChoose] /  CommunicationIerations
    elif workers == '14':       
        comm_bandwidth_avg[FedAvg_name] = comm_bandwidth_avg[FedAvg_name] / CommunicationIerations
        comm_bandwidth_avg[FedAvg_sparse_name] = comm_bandwidth_avg[FedAvg_sparse_name] / CommunicationIerations
        comm_bandwidth_avg[DPSGD_name] = comm_bandwidth_avg[DPSGD_name] / CommunicationIerations
        comm_bandwidth_avg[DCD_name] = comm_bandwidth_avg[DCD_name] / CommunicationIerations
        comm_bandwidth_avg[SAPS_name] = comm_bandwidth_avg[SAPS_name] / CommunicationIerations
        # comm_bandwidth_avg[DensePSGD_name] = comm_bandwidth_avg[DensePSGD_name] / CommunicationIerations
        # comm_bandwidth_avg[TopKPSGD_name] = comm_bandwidth_avg[TopKPSGD_name] / CommunicationIerations
        comm_bandwidth_avg[RamdomChoose] = comm_bandwidth_avg[RamdomChoose] /  CommunicationIerations

    print("===========average=================\n")
    # print("random  bandwidth_threshold", random_bandwidth_threshold)
    print("ring  bandwidth_threshold", comm_bandwidth_avg[FedAvg_name])
    print("FedAVG  bandwidth_threshold", comm_bandwidth_avg[DPSGD_name])
    print("preparation_match   bandwidth_threshold", comm_bandwidth_avg[SAPS_name])
    print("RamdomChoose   bandwidth_threshold", comm_bandwidth_avg[RamdomChoose])

    # plot_bandwidth(None, FedAvg_name, isbandwidth=False, plot_loss=False, isacc=False, title='Bandwidth', comm_bandwidth=comm_bandwidth_list[FedAvg_name])
    # plot_bandwidth(None, FedAvg_sparse_name, isbandwidth=False, plot_loss=False, isacc=False, title='Bandwidth', comm_bandwidth=comm_bandwidth_list[FedAvg_sparse_name])
    plot_bandwidth(None, DPSGD_name, isbandwidth=False, plot_loss=False, isacc=False, title='Bandwidth', comm_bandwidth=comm_bandwidth_list[DPSGD_name][100:], ax=ax)
    plot_bandwidth(None, DCD_name, isbandwidth=False, plot_loss=False, isacc=False, title='Bandwidth',  comm_bandwidth=comm_bandwidth_list[DCD_name][100:], ax=ax)
    plot_bandwidth(None, SAPS_name, isbandwidth=False, plot_loss=False, isacc=False, title='Bandwidth', comm_bandwidth=comm_bandwidth_list[SAPS_name][100:], ax=ax)
    plot_bandwidth(None, RamdomChoose, isbandwidth=False, plot_loss=False, isacc=False, title='Bandwidth',  comm_bandwidth=comm_bandwidth_list[RamdomChoose][100:], ax=ax)

def ICNP2020_SAPS():
    def convergence(network, workers):
        global max_epochs
        max_epochs = max_epochs_dict[network]
        #workers = '32'
        if_iid = True
        plt.figure()
        fig, ax = plt.subplots(1,1,figsize=(5,3.4))
        ICNP2020_SAPS_convergence(network=network, workers=workers, if_iid=if_iid, ax=ax)

        ax.set_xlim(xmin=-1)
        ax.legend(fontsize=FONTSIZE, loc='lower right')
        if network == 'mnistCNN':
            pass
        else:
            pass
            #ax.legend().set_visible(False)
        plt.subplots_adjust(bottom=0.16, left=0.15, right=0.96, top=0.95)
        plt.savefig('%s/%s_%s_convergence.pdf' % (OUTPUTPATH, network, workers))
        #plt.show()

    def communication_data(network, workers):
        global max_epochs
        max_epochs = max_epochs_dict[network]

        if_iid = True
        plt.figure()
        fig, ax = plt.subplots(1,1,figsize=(5,3.4))
        ICNP2020_SAPS_communication(network=network, workers=workers, if_iid=if_iid, ax=ax)
        ax.set_xscale('log')
        ax.set_xlim(xmin=-1)
        # plt.legend()
        ax.legend(fontsize=FONTSIZE, loc='lower right')
        if network == 'mnistCNN':
            pass
        else:
            pass
            #ax.legend().set_visible(False)
        plt.subplots_adjust(bottom=0.16, left=0.15, right=0.96, top=0.95)
        plt.savefig('%s/%s_%s_communication_data.pdf' % (OUTPUTPATH, network, workers))
        #plt.show()

    def communication_time(network, workers):
        global max_epochs
        max_epochs = max_epochs_dict[network]
        if_iid = True
        plt.figure()
        fig, ax = plt.subplots(1,1,figsize=(5,3.4))
        ICNP2020_SAPS_communication_time_iteration(network=network, workers=workers, if_iid=if_iid, ax=ax)
        ax.set_xscale('log')
        ax.set_xlim(xmin=-1)

        # plt.legend()
        ax.legend(fontsize=FONTSIZE,loc='lower right')
        if network == 'mnistCNN':
            pass
        else:
            pass
            #ax.legend().set_visible(False)
        plt.subplots_adjust(bottom=0.16, left=0.15, right=0.96, top=0.95)
        plt.savefig('%s/%s_%s_communication_speed.pdf' % (OUTPUTPATH, network, workers))
        #plt.show()

    def robust_convergence(network, workers):
        global max_epochs
        max_epochs = max_epochs_dict[network]
        #workers = '32'
        if_iid = True
        plt.figure()
        fig, ax = plt.subplots(1,1,figsize=(5,3.4))
        ICNP2020_SAPS_convergence_robust(network=network, workers=workers, if_iid=if_iid, ax=ax)

        ax.set_xlim(xmin=-1)
        ax.legend(fontsize=FONTSIZE, loc='lower right')
        if network == 'mnistCNN':
            pass
        else:
            pass
            #ax.legend().set_visible(False)
        plt.subplots_adjust(bottom=0.16, left=0.15, right=0.96, top=0.95)
        plt.savefig('%s/robust_%s_%s_convergence.pdf' % (OUTPUTPATH, network, workers))
        #plt.show()


    def plot_bandwidth(workers):
        global max_epochs
        max_epochs = 200
        if_iid = True
        plt.figure()
        fig, ax = plt.subplots(1,1,figsize=(5,4))
        ICNP2020_bandwidth(network=network, workers=workers, if_iid=if_iid, ax=ax)
        ax.set_xlim(xmin=-1)
        # plt.legend()
        ax.legend(fontsize=14, ncol=2, loc = "upper center")
        plt.ylim(ymax = 6)
        plt.subplots_adjust(bottom=0.13, left=0.14, right=0.96, top=0.94)
        plt.savefig('%s/%s_bandwidth.pdf' % (OUTPUTPATH, workers))
        #plt.savefig('%s_communication_time.pdf' % (network))
        #plt.show()

    def plot_heatmap_B(workers):
        if workers == '14':
            T_thres = [1, 2, 5, 10, 15]
            B_thres = [0.5, 1, 1.45, 5, 10]
            data = np.array([[0.9698, 0.9949, 1.2214, 1.216, 1.21906],
                            [1.00635, 1.01103, 1.27491, 1.2709, 1.27826],
                            [0.98561, 0.96588, 1.50635, 1.5117, 1.5133],
                            [0.9816, 1.02006, 1.03244, 0.98929, 0.94983],
                            [0.96956, 0.9923, 1.03913, 1.00535, 1.02408]])

        elif workers == '32':
            T_thres = [1, 2, 5, 10, 15]
            B_thres = [0.5, 1, 2, 3, 4]
            data = np.array([[0.255476, 0.27804,  0.73375, 0.72597, 0.75813],
                            [0.24929, 0.26922, 1.27491, 1.2709, 1.23381],
                            [0.25501, 0.29978, 2.16188, 2.18949, 2.17933],
                            [0.26145, 0.27057, 3.13464, 3.14867, 3.13427],
                            [0.26664, 0.27256, 3.56649, 4.04863, 4.05027]])

        #heatmap = np.zeros((len(T_thres), len(B_thres)))
        #for i in range(len(T_thres)):
        #    for j in range(len(B_thres)):
        #            heatmap[i][j] = data
        plt.figure()
        fig, ax1 = plt.subplots(figsize=(5,4))

        #plt.subplots_adjust(left=0.0, right=1.00, top=1.0, bottom=0.15)
        plt.subplots_adjust(left=0.17, right=0.95, top=0.97, bottom=0.16)
        #print('heatmap: ', heatmap)
        im = ax1.imshow(data)
        for i in range(len(B_thres)):
            for j in range(len(T_thres)):
                text = ax1.text(j, i, '%.3f'%data[i, j], ha="center", va="center", color="w", fontsize=FONTSIZE)
        #cbar = ax1.figure.colorbar(im, ax=ax1, **cbar_kw)
        cbar = ax1.figure.colorbar(im, ax=ax1)
        cbar.ax.set_ylabel('Bandwidth [Mbits/s]', rotation=-90, va="bottom")
        ax1.set_xticks(np.arange(len(T_thres)))
        ax1.set_yticks(np.arange(len(B_thres)))
        ax1.set_xticklabels(T_thres)
        ax1.set_yticklabels(B_thres)
        ax1.set_xlabel('T_thres [iter]')
        ax1.set_ylabel('B_thres [Mbits/s]')
        ax1.tick_params(which="minor", bottom=False, left=False)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")
        u.update_fontsize(ax1, FONTSIZE)
        u.update_fontsize(cbar.ax, FONTSIZE)
        #fig.tight_layout()
        plt.savefig('bandwidth_%s_with_thres.pdf'%workers)
        #plt.show()

    #network = ["mnistCNN", "cifar10CNN", "resnet20"]
    network = ["mnistCNN", "resnet20"]
    #network = ["mnistCNN"]
    workers = ['14', '32']# '32'
    #convergence(network[1], '32')
    #communication_data(network[2], '32')
    #communication_time(network[2], '32')
    #robust_convergence(network[0], '32')
    #plot_bandwidth(workers[1])
    #plot_heatmap_B('32')
    #plot_heatmap_B('14')
    for w in ['32']:
        for item in network:
            #convergence(item, w)
            #communication_data(item, w)
            #communication_time(item, w)
            pass
        #plot_bandwidth(w)


if __name__ == '__main__':
    #resnet20()
    ICNP2020_SAPS()
    #get_ring_and_fedavg_bandwidth_avg()