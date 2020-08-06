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
from All_process import get_serialized_log

#markers=['.','x','o','v','^','<','>','1','2','3','4','8','s','p','*']
#markers=['.','x','o','v','^','<','>']
markers=[None]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'olive']
#colors = colors[2:7]
#colors = colors[0:4]
colors = colors[0:8]
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
LINE_WIDTH = 1.5
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


def plot_one_line(x_data, y_data, legend, scale=None, ax=None):
    new_y_data = []
    if scale:
        for item in y_data:
            for i in range(scale):
                new_y_data.append(item)
        y_data = new_y_data

    print(legend, ":", max(y_data))
    #ax.set_title(get_real_title(title))
    # marker = markeriter.next()
    # color = coloriter.next()
    marker = next(markeriter)
    color = next(coloriter)
    #print('marker: ', marker)
    ax.plot(x_data, y_data, label=legend, marker=marker, linewidth=LINE_WIDTH, markerfacecolor='none', color=color)


def plot_figure(Data_Get_Configs, training_or_tensor, title, x_label, y_label, file_path):
    plt.figure()
    fig, ax = plt.subplots(1,1,figsize=(5,3.4))

    if title:
           plt.title(title)

    for Data_Get_Config in Data_Get_Configs:
        y_data = get_serialized_log(Data_Get_Config.dir_path, training_or_tensor=training_or_tensor, model=Data_Get_Config.model, 
            tensor_size=Data_Get_Config.tensor_size, DMLC_PS=Data_Get_Config.DMLC_PS, 
            batch_size=Data_Get_Config.batch_size, num_iters=Data_Get_Config.num_iters, 
            nworkers=Data_Get_Config.nworkers, nservers=Data_Get_Config.nservers, 
            worker_id=Data_Get_Config.worker_id, local_rank=Data_Get_Config.local_rank)
        plot_one_line(Data_Get_Config.x_data, y_data, Data_Get_Config.legend, scale=None, ax=ax)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(xmin=-1)
    ax.legend(fontsize=FONTSIZE, loc='lower right')
    ax.grid(linestyle=':')
    u.update_fontsize(ax, FONTSIZE)

    #ax.legend().set_visible(False)
    plt.subplots_adjust(bottom=0.16, left=0.15, right=0.96, top=0.95)
    # plt.savefig(file_path)
    plt.show()

def plot_training_comm():

    data_Get_Configs = []
    for DMLC_PS in ['10.0.0.11', '192.168.0.11']:
        for model in ['alexnet', 'resnet50']:
            for same in [' ', 'same']:
                if same == 'same':
                    legend = DMLC_PS+'-'+model+'-layerwise' + '-same'
                    dir_path = 'bps_logs0804/bps_layerwise_same_log'
                else:
                    legend = DMLC_PS+'-'+model+'-layerwise'
                    dir_path = 'bps_logs0804/bps_layerwise_log'

                data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[model], tensor_size=[],
                    DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
                    worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend=legend)
                data_Get_Configs.append(data_Get_Config)

    file_path = './training_comm.pdf'
    plot_figure(data_Get_Configs, training_or_tensor='training', title=None, x_label='servers', y_label='Img/sec', file_path=file_path)

def plot_tensor_comm():

    data_Get_Configs = []

    # data_Get_Configs.append(u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_log', model=[], tensor_size=['64'],
    #     DMLC_PS=['192.168.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='192.168.0.11-tensor-size64'))


    data_Get_Configs.append( u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_log', model=[], tensor_size=['256'],
        DMLC_PS=['10.0.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
        worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='10.0.0.11-tensor-size256'))

    # data_Get_Configs.append( u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_same_log', model=[], tensor_size=['64'],
    #     DMLC_PS=['192.168.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='192.168.0.11-tensor-size64-same'))

    data_Get_Configs.append( u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_same_log', model=[], tensor_size=['256'],
        DMLC_PS=['10.0.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
        worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='10.0.0.11-tensor-size256-same'))

    file_path = './tensor_comm.pdf'
    plot_figure(data_Get_Configs, training_or_tensor='tensor', title=None, x_label='servers', y_label='Time (Seconds)', file_path=file_path)


#plot_training_comm()

plot_tensor_comm()




























