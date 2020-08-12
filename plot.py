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
import numpy as np
from All_process import get_serialized_log
from All_process import extract_one_tensor_Iter_Time_log

markers=['.','x','o','v','^','<','>','1','2','3','4','8','s','p','*']
#markers=['.','x','o','v','^','<','>']

colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'darkgreen', 'darkblue', 'brown', 'darkorange']
#colors = colors[2:7]
#colors = colors[0:4]
colors = colors[0:12]
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


def plot_one_line(x_data, y_data, legend, scale=None, ax=None, marker=None, color=None):
    new_y_data = []
    if scale:
        for item in y_data:
            for i in range(scale):
                new_y_data.append(item)
        y_data = new_y_data

    print(legend, ":", max(y_data))
    #ax.set_title(get_real_title(title))
    #marker = markeriter.next()
    # color = coloriter.next()
    # marker = next(markeriter)
    # color = next(coloriter)
    #print('marker: ', marker)
    ax.plot(x_data, y_data, label=legend, marker=marker, linewidth=LINE_WIDTH, markerfacecolor='none', color=color)


def plot_figure(Data_Get_Configs, training_or_tensor, title, x_label, y_label, file_path, legend_location, subplots_adjust):
    plt.figure()
    fig, ax = plt.subplots(1,1,figsize=(5,3.4))

    if title:
           plt.title(title)

    for Data_Get_Config in Data_Get_Configs:
        plot_one_line(Data_Get_Config.x_data, Data_Get_Config.y_data, Data_Get_Config.legend, scale=None, ax=ax,
            color=Data_Get_Config.color, marker=Data_Get_Config.marker)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(xmin=-1)
    ax.legend(fontsize=FONTSIZE, loc=legend_location)
    ax.grid(linestyle=':')
    u.update_fontsize(ax, FONTSIZE)

    #ax.legend().set_visible(False)
    plt.subplots_adjust(bottom=subplots_adjust[0], left=subplots_adjust[1], right=subplots_adjust[2], top=subplots_adjust[3])
    # plt.savefig(file_path)
    plt.show()

def plot_training_comm(root_path):
    legend_location = 'lower right'
    subplots_adjust = [0.16, 0.15, 0.96, 0.95]

    data_Get_Configs = []
    for DMLC_PS in ['10.0.0.11', '192.168.0.11']:
        for model in ['alexnet', 'resnet50']:
            for same in [' ', 'same']:
                if same == 'same':
                    legend = DMLC_PS+'-'+model+'-layerwise' + '-same'
                    dir_path = root_path+'/bps_layerwise_same_log'
                else:
                    legend = DMLC_PS+'-'+model+'-layerwise'
                    dir_path = root_path+'/bps_layerwise_log'

                data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[model], tensor_size=[], KB='1',
                    DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
                    worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend=legend)
                data_Get_Config.color = next(coloriter)
                data_Get_Config.marker = next(markeriter)
                data_Get_Configs.append(data_Get_Config)

    file_path = './training_comm.pdf'
    for Data_Get_Config in data_Get_Configs:
        y_data = get_serialized_log(Data_Get_Config.dir_path, training_or_tensor='training', model=Data_Get_Config.model, 
            tensor_size=Data_Get_Config.tensor_size, KB=Data_Get_Config.KB, DMLC_PS=Data_Get_Config.DMLC_PS, 
            batch_size=Data_Get_Config.batch_size, num_iters=Data_Get_Config.num_iters, 
            nworkers=Data_Get_Config.nworkers, nservers=Data_Get_Config.nservers, 
            worker_id=Data_Get_Config.worker_id, local_rank=Data_Get_Config.local_rank)
        Data_Get_Config.y_data=y_data

    plot_figure(data_Get_Configs, training_or_tensor='training', title=None, x_label='servers', y_label='Img/sec',
         file_path=file_path, legend_location=legend_location, subplots_adjust=subplots_adjust)

def plot_tensor_comm(root_path):
    legend_location = 'upper right'
    subplots_adjust = [0.16, 0.15, 0.96, 0.95]
    data_Get_Configs = []

    # bps_logs0804
    # data_Get_Configs.append(u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_log', model=[], tensor_size=['64'],
    #     DMLC_PS=['192.168.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='192.168.0.11-tensor-size64'))


    # data_Get_Configs.append( u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_log', model=[], tensor_size=['256'],
    #     DMLC_PS=['10.0.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='10.0.0.11-tensor-size256'))

    # data_Get_Configs.append( u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_same_log', model=[], tensor_size=['64'],
    #     DMLC_PS=['192.168.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='192.168.0.11-tensor-size64-same'))

    # data_Get_Configs.append( u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_same_log', model=[], tensor_size=['256'],
    #     DMLC_PS=['10.0.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='10.0.0.11-tensor-size256-same'))


    for DMLC_PS in ['10.0.0.11', '192.168.0.11']:
        #if DMLC_PS == '10.0.0.11':
        if DMLC_PS == '192.168.0.11':
            #pass
            continue
        # for tensor_size in ['512', '1024', '2048', '4096', '8192', '16384',
        #      '32768', '65536', '131072', '262144', '524288', '1048576']:
        for tensor_size in ['512', '2048', '8192']:
        #for tensor_size in ['128', '512', '2048', '8192']:
        #for tensor_size in ['8', '16', '32', '64', '128', '256']:
        #for tensor_size in ['8']:
            for same in [' ', 'same']:
                if same == 'same':
                    #continue
                    legend = DMLC_PS+'-tensor-size' + str(tensor_size) + '-same'
                    dir_path = root_path+'/one_tensor_test_same_log'
                else:
                    #continue
                    legend = DMLC_PS+'-tensor-size' + str(tensor_size)
                    dir_path = root_path+'/one_tensor_test_log'
                Data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size], KB='1',
                    DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
                    worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend=legend)
                # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
                #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '4', '8'],
                #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '4', '8'], legend=legend)
                # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
                #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['3', '5', '6', '7'],
                #     worker_id=['0'], local_rank=['0'], x_data=['3', '5', '6', '7'], legend=legend)


                Data_Get_Config.color = next(coloriter)
                Data_Get_Config.marker = next(markeriter)
                y_data = get_serialized_log(Data_Get_Config.dir_path, training_or_tensor='tensor', model=Data_Get_Config.model, 
                    tensor_size=Data_Get_Config.tensor_size, KB=Data_Get_Config.KB, DMLC_PS=Data_Get_Config.DMLC_PS, 
                    batch_size=Data_Get_Config.batch_size, num_iters=Data_Get_Config.num_iters, 
                    nworkers=Data_Get_Config.nworkers, nservers=Data_Get_Config.nservers, 
                    worker_id=Data_Get_Config.worker_id, local_rank=Data_Get_Config.local_rank)
                # ==========================================================================
                Data_Get_Config.y_data=y_data
                data_Get_Configs.append(Data_Get_Config)

    file_path = './tensor_comm.pdf'
    plot_figure(data_Get_Configs, training_or_tensor='tensor', title=None, x_label='servers', y_label='Time (Seconds)',
         file_path=file_path, legend_location=legend_location, subplots_adjust=subplots_adjust)

def plot_tensor_comm_with_size(DMLC_PS, same):
    legend_location = 'lower right'
    subplots_adjust = [0.16, 0.15, 0.96, 0.95]
    data_Get_Configs = []

    # bps_logs0804
    # data_Get_Configs.append(u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_log', model=[], tensor_size=['64'],
    #     DMLC_PS=['192.168.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='192.168.0.11-tensor-size64'))


    # data_Get_Configs.append( u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_log', model=[], tensor_size=['256'],
    #     DMLC_PS=['10.0.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='10.0.0.11-tensor-size256'))

    # data_Get_Configs.append( u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_same_log', model=[], tensor_size=['64'],
    #     DMLC_PS=['192.168.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='192.168.0.11-tensor-size64-same'))

    # data_Get_Configs.append( u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_same_log', model=[], tensor_size=['256'],
    #     DMLC_PS=['10.0.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='10.0.0.11-tensor-size256-same'))


    for nservers in ['1', '2', '3', '4', '5', '6', '7', '8']:
        x_data = []
        y_data = []
        # tensor_size = ['512', '2048', '8192']
        # root_path = 'bps_logs0807_2'
        # if same == 'same':
        #     legend = DMLC_PS+'-nservers' + str(nservers) + '-same'
        #     dir_path = root_path+'/one_tensor_test_same_log'
        # else:
        #     legend = DMLC_PS+'-nservers' + str(nservers)
        #     dir_path = root_path+'/one_tensor_test_log'
        # x_data += [ int(item) for item in tensor_size ]
        # Data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=tensor_size, KB='1',
        #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=[nservers],
        #     worker_id=['0'], local_rank=['0'], x_data=[], legend=legend)


        # y_data += get_serialized_log(Data_Get_Config.dir_path, training_or_tensor='tensor', model=Data_Get_Config.model, 
        #     tensor_size=Data_Get_Config.tensor_size, KB=Data_Get_Config.KB, DMLC_PS=Data_Get_Config.DMLC_PS, 
        #     batch_size=Data_Get_Config.batch_size, num_iters=Data_Get_Config.num_iters, 
        #     nworkers=Data_Get_Config.nworkers, nservers=Data_Get_Config.nservers, 
        #     worker_id=Data_Get_Config.worker_id, local_rank=Data_Get_Config.local_rank)
        # 1024 4096 16384
        tensor_size  = ['512', '1024', '2048', '4096', '8192', '16384',
              '32768', '65536', '131072', '262144', '524288', '1048576']
        tensor_size  = ['512', '1024', '2048', '4096', '8192', '16384']
        tensor_size  = ['8192', '16384']
        #root_path = 'bps_logs0809_rdma'
        root_path = 'bps_logs0811'
        if same == 'same':
            legend = DMLC_PS+'-nservers' + str(nservers) + '-same'
            dir_path = root_path+'/one_tensor_test_same_log'
        else:
            legend = DMLC_PS+'-nservers' + str(nservers)
            dir_path = root_path+'/one_tensor_test_log'
        x_data += [ int(item) for item in tensor_size ]
        Data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=tensor_size, KB='1',
            DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=[nservers],
            worker_id=['0'], local_rank=['0'], x_data=[], legend=legend)
        y_data += get_serialized_log(Data_Get_Config.dir_path, training_or_tensor='tensor', model=Data_Get_Config.model, 
            tensor_size=Data_Get_Config.tensor_size, KB='1', DMLC_PS=Data_Get_Config.DMLC_PS, 
            batch_size=Data_Get_Config.batch_size, num_iters=Data_Get_Config.num_iters, 
            nworkers=Data_Get_Config.nworkers, nservers=Data_Get_Config.nservers, 
            worker_id=Data_Get_Config.worker_id, local_rank=Data_Get_Config.local_rank)


        tensor_size = ['8', '16', '32', '64', '128', '256']
        root_path = 'bps_logs0807'
        KB = '0'
        if DMLC_PS == '10.0.0.11':
            root_path = 'bps_logs0811'
            tensor_size = [ str(int(item)*1024*4) for item in tensor_size ]
            KB = '1'
        if same == 'same':
            legend = DMLC_PS+'-nservers' + str(nservers) + '-same'
            dir_path = root_path+'/one_tensor_test_same_log'
        else:
            legend = DMLC_PS+'-nservers' + str(nservers)
            dir_path = root_path+'/one_tensor_test_log'

        # Data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=tensor_size, KB='0',
        #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=[nservers],
        #     worker_id=['0'], local_rank=['0'], x_data=[], legend=legend)
        if KB == '0':
            x_data += [ int(item)*1024*4 for item in tensor_size  ]
        else:
            x_data += [ int(item) for item in tensor_size  ]

        y_data += get_serialized_log(dir_path, training_or_tensor='tensor', model=Data_Get_Config.model, 
            tensor_size=tensor_size, KB=KB, DMLC_PS=Data_Get_Config.DMLC_PS, 
            batch_size=Data_Get_Config.batch_size, num_iters=Data_Get_Config.num_iters, 
            nworkers=Data_Get_Config.nworkers, nservers=Data_Get_Config.nservers, 
            worker_id=Data_Get_Config.worker_id, local_rank=Data_Get_Config.local_rank)






        Data_Get_Config.x_data=x_data
        Data_Get_Config.y_data=y_data


        Data_Get_Config.color = next(coloriter)
        Data_Get_Config.marker = next(markeriter)
        data_Get_Configs.append(Data_Get_Config)


    file_path = './tensor_comm.pdf'
    plot_figure(data_Get_Configs, training_or_tensor='tensor', title=None, x_label='Tensor Size (KB)', y_label='Time (Seconds)',
         file_path=file_path, legend_location=legend_location, subplots_adjust=subplots_adjust)


def plot_tensor_comm_with_iters():
    legend_location = 'upper right'
    subplots_adjust = [0.16, 0.15, 0.96, 0.95]
    data_Get_Configs = []

    # bps_logs0804
    # data_Get_Configs.append(u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_log', model=[], tensor_size=['64'],
    #     DMLC_PS=['192.168.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='192.168.0.11-tensor-size64'))


    # data_Get_Configs.append( u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_log', model=[], tensor_size=['256'],
    #     DMLC_PS=['10.0.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='10.0.0.11-tensor-size256'))

    # data_Get_Configs.append( u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_same_log', model=[], tensor_size=['64'],
    #     DMLC_PS=['192.168.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='192.168.0.11-tensor-size64-same'))

    # data_Get_Configs.append( u.Data_Get_Config(dir_path='bps_logs0804/one_tensor_test_same_log', model=[], tensor_size=['256'],
    #     DMLC_PS=['10.0.0.11'], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend='10.0.0.11-tensor-size256-same'))

    #root_path = 'bps_logs0809_rdma'
    #root_path = 'bps_logs0807_2'
    root_path = 'bps_logs0811'
    #for nservers in ['1', '2', '3', '4', '5', '6', '7', '8']:
    for nservers in ['5', '6', '7', '8']:
        for DMLC_PS in ['10.0.0.11', '192.168.0.11']:
            #if DMLC_PS == '10.0.0.11':
            if DMLC_PS == '192.168.0.11':
                #pass
                continue
            for tensor_size in ['512', '1024', '2048', '4096', '8192', '16384',
                 '32768', '65536', '131072', '262144', '524288', '1048576']:
            #for tensor_size in ['512', '2048', '8192']:
            #for tensor_size in ['128', '512', '2048', '8192']:
            #for tensor_size in ['8', '16', '32', '64', '128', '256']:
            # for tensor_size in ['8192']:
                for same in [' ', 'same']:
                    if same == 'same':
                        continue
                        legend = DMLC_PS+'-tensor-size' + str(tensor_size) + 'nservers' + str(nservers) + '-same'
                        dir_path = root_path+'/one_tensor_test_same_log'
                    else:
                        #continue
                        legend = DMLC_PS+'-tensor-size' + str(tensor_size) + 'nservers' + str(nservers)
                        dir_path = root_path+'/one_tensor_test_log'
                    Data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size], KB='1',
                        DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=[nservers],
                        worker_id=['0'], local_rank=['0'], x_data=[' '], legend=legend)
                    # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
                    #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '4', '8'],
                    #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '4', '8'], legend=legend)
                    # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
                    #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['3', '5', '6', '7'],
                    #     worker_id=['0'], local_rank=['0'], x_data=['3', '5', '6', '7'], legend=legend)
                    color = next(coloriter)
                    Data_Get_Config.color = color
                    Data_Get_Config.marker = next(markeriter)
                    y_data = extract_one_tensor_Iter_Time_log(Data_Get_Config.dir_path, training_or_tensor='tensor', model=Data_Get_Config.model[0], 
                        tensor_size=Data_Get_Config.tensor_size[0], KB=Data_Get_Config.KB, DMLC_PS=Data_Get_Config.DMLC_PS[0], 
                        batch_size=Data_Get_Config.batch_size[0], num_iters=Data_Get_Config.num_iters[0], 
                        nworkers=Data_Get_Config.nworkers[0], nservers=Data_Get_Config.nservers[0], 
                        worker_id=Data_Get_Config.worker_id[0], local_rank=Data_Get_Config.local_rank[0])

                    #print(y_data)
                    x_data = list(range(0, len(y_data)))
                    Data_Get_Config.y_data=y_data
                    Data_Get_Config.x_data=x_data
                    data_Get_Configs.append(Data_Get_Config)

                    origin_mean = np.mean(y_data[10:50])
                    origin_std = np.std(y_data[10:50])
                    print("origin_mean: %f" % origin_mean)
                    print("origin_std: %f" % origin_std)
                    y_data_adjust = [ item if item < origin_mean + 1*origin_std and item > origin_mean - 2*origin_std else origin_mean for item in y_data ]
                    new_mean = np.mean(y_data_adjust)
                    new_std = np.mean(y_data_adjust)
                    print("new_mean: %f" % new_mean)
                    print("new_std: %f" % new_std)
                    Data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size], KB='1',
                        DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=[nservers],
                        worker_id=['0'], local_rank=['0'], x_data=[' '], legend=legend+'adjust')
                    Data_Get_Config.y_data=y_data_adjust
                    Data_Get_Config.x_data=x_data
                    Data_Get_Config.color = color
                    Data_Get_Config.marker = next(markeriter)
                    data_Get_Configs.append(Data_Get_Config)

    file_path = './tensor_comm_with_iters.pdf'
    plot_figure(data_Get_Configs, training_or_tensor='tensor', title=None, x_label='iters', y_label='Time (Seconds)',
         file_path=file_path, legend_location=legend_location, subplots_adjust=subplots_adjust)


if __name__ == '__main__':

    root_path='bps_logs0804'
    plot_training_comm(root_path)

    # root_path='bps_logs0807_2'
    # #root_path='bps_logs0807'
    # plot_tensor_comm(root_path)

    # '10.0.0.11', '192.168.0.11'
    # ' ', 'same'
    # plot_tensor_comm_with_size(DMLC_PS='10.0.0.11', same=' ')
    # #plot_tensor_comm_with_size(DMLC_PS='192.168.0.11', same=' ')
    # plot_tensor_comm_with_size(DMLC_PS='10.0.0.11', same='same')
    # #plot_tensor_comm_with_size(DMLC_PS='192.168.0.11', same='same')

    # root_path='bps_logs0809_rdma'
    # plot_tensor_comm(root_path)


    # plot_tensor_comm_with_iters()
    pass
