import numpy as np 
from scipy import optimize
from scipy.optimize import leastsq
import numpy as np
import xlrd
from scipy.optimize import differential_evolution
import warnings
import utils as u
import plot
import itertools
import json
from All_process import get_serialized_log


file_bps = './bps_logs/bps.xls'
file_trace = './bps_logs/trace.xls'


#file_name = file_bps
file_name = file_trace
kind = 'trace'
network = 'RDMA' # 'TCP' 'RDMA'

TCP_B = 10000000000  #10Gb
RDMA_B = 100000000000   # 100Gb
# old Alexnet = 60965128 * 32 #     243860512 B number of params * bits 22 1950884096
Alexnet = 61100840 * 32   # 244403360 B    1955226880
# old Resnet50 = 46159168 * 32 #    184636672 B 1477093376
Resnet50 = 25557032 * 32 # 102228128 B   817825024
# old Densenet121 = 7978856 * 32 #  31915424 B 255323392
Densenet121 = 7978856 * 32  # 31915424 B 255323392
# old Vgg16 = 138344128 * 32 #      553376512 B  4427012096 512000000
Vgg16 = 138357544  * 32 # 553430176  4427441408
servers = 4
workers = 4
models = {'alexnet':60965128* 32, 'resnet50':46159168* 32, 'densenet121':7978856* 32, 'vgg16':138344128* 32}

latency = 0.02
# PCI = 16000000000 * 8  #16GB
PCI = 15754000000 * 8  #15.754 GB/s
# tpsc_cpu = 0.05 
# tpsc_2 = 0.05 
# tpsc_cpu = 10 / 2400000000 # CPU frequency, estimated operation time of one param
# tpsc_2 = 10 / 2400000000

net_name = []
worker_num = []
server_num = []
Bandwidth = []
Com_up = []
bcast = {}
copyh2d = {}
copyd2h = {}
local_reduce = {}
pull = {}
push = {}

choosenet = 'densenet121'
def get_key(model, network, nserver, nworker):
    return model+network + nserver + nworker

def Excel_API(file_name, kind):
    if kind == 'bps':
        excel = xlrd.open_workbook(file_name,encoding_override="utf-8")
        all_sheet = excel.sheets()
        #print(all_sheet[0])
        for row in range(all_sheet[0].nrows):
            # print(len(all_sheet[0].row_values(row)))
            row_list = (all_sheet[0].row_values(row))[:]
            # for i in range(len(row_list)):
            row_split = row_list[0].split('-')
            # print(len(row_split))
            if len(row_split) == 5 and 'old' not in row_split[4] and 'id0' in row_split[4]:
                #print(row_split)
                if row_split[3][-1] == '1':
                    continue
                #if row_split[0] != 'vgg16':
                #if row_split[0] == choosenet:
                if network == 'TCP':
                # if True:
                    net_name.append(row_split[0])
                    Bandwidth.append('TCP_B')
                    worker_num.append(int(row_split[3][-1]))
                    server_num.append(int(row_split[4][8]))
                    Com_up.append(row_list[4]*1e-8)
                    # print(net_name[-1])
                    # print(Bandwidth[-1])
                    # print(worker_num[-1])
                    # print(server_num[-1])
                    # print(Com_up[-1])
            elif len(row_split) == 6 and 'old' not in row_split[5] and 'id0' in row_split[5]:
                #print(row_split)
                #continue
                if row_split[4][-1] == '1':
                    continue
                #if row_split[0] != 'vgg16':
                #if row_split[0] == choosenet:
                if network == 'RDMA':
                # if True:
                    net_name.append(row_split[0])
                    Bandwidth.append('RDMA_B')
                    worker_num.append(int(row_split[4][-1]))
                    server_num.append(int(row_split[5][8]))
                    Com_up.append(row_list[4]*1e-8)
    elif kind == 'trace':
        excel = xlrd.open_workbook(file_name,encoding_override="utf-8")
        all_sheet = excel.sheets()
        #print(all_sheet[0])
        for row in range(all_sheet[0].nrows):
            # print(len(all_sheet[0].row_values(row)))
            row_list = (all_sheet[0].row_values(row))[:]
            # for i in range(len(row_list)):
            row_split = row_list[0].split('_')
            # print(len(row_split))
            if len(row_split) == 7 and 'old' not in row_split and 'test' not in row_split:
                #print(row_split)
                if row_split[5][-1] == '1':
                    continue
                #if row_split[0] != 'vgg16':
                #if row_split[0] == choosenet:
                if network == 'TCP':
                # if True:
                    if 'id0' in row_split:
                        net_name.append(row_split[1])
                        Bandwidth.append('TCP_B')
                        worker_num.append(int(row_split[5][-1]))
                        server_num.append(int(row_split[4][-1]))
                    if get_key(row_split[1], 'TCP_B', row_split[4][-1], row_split[5][-1]) in bcast.keys():
                        bcast[get_key(row_split[1], 'TCP_B', row_split[4][-1], row_split[5][-1])] += row_list[1]*1e-6*(1/int(row_split[5][-1]))
                        copyh2d[get_key(row_split[1], 'TCP_B', row_split[4][-1], row_split[5][-1])] += row_list[2]*1e-6*(1/int(row_split[5][-1]))
                        copyd2h[get_key(row_split[1], 'TCP_B', row_split[4][-1], row_split[5][-1])] += row_list[3]*1e-6*(1/int(row_split[5][-1]))
                        local_reduce[get_key(row_split[1], 'TCP_B', row_split[4][-1], row_split[5][-1])] += row_list[4]*1e-6*(1/int(row_split[5][-1]))
                        pull[get_key(row_split[1], 'TCP_B', row_split[4][-1], row_split[5][-1])] += row_list[5]*1e-6*(1/int(row_split[5][-1]))
                        push[get_key(row_split[1], 'TCP_B', row_split[4][-1], row_split[5][-1])] += row_list[6]*1e-6*(1/int(row_split[5][-1]))
                    else:
                        bcast[get_key(row_split[1], 'TCP_B', row_split[4][-1], row_split[5][-1])] = row_list[1]*1e-6*(1/int(row_split[5][-1]))
                        copyh2d[get_key(row_split[1], 'TCP_B', row_split[4][-1], row_split[5][-1])] = row_list[2]*1e-6*(1/int(row_split[5][-1]))
                        copyd2h[get_key(row_split[1], 'TCP_B', row_split[4][-1], row_split[5][-1])] = row_list[3]*1e-6*(1/int(row_split[5][-1]))
                        local_reduce[get_key(row_split[1], 'TCP_B', row_split[4][-1], row_split[5][-1])] = row_list[4]*1e-6*(1/int(row_split[5][-1]))
                        pull[get_key(row_split[1], 'TCP_B', row_split[4][-1], row_split[5][-1])] = row_list[5]*1e-6*(1/int(row_split[5][-1]))
                        push[get_key(row_split[1], 'TCP_B', row_split[4][-1], row_split[5][-1])] = row_list[6]*1e-6*(1/int(row_split[5][-1]))
                    # print(net_name[-1])
                    # print(Bandwidth[-1])
                    # print(worker_num[-1])
                    # print(server_num[-1])
                    # print(Com_up[-1])
            elif len(row_split) == 8 and 'old' not in row_split and 'test' not in row_split:
                #print(row_split)
                #continue
                if row_split[6][-1] == '1':
                    continue
                #if row_split[0] != 'vgg16':
                #if row_split[0] == choosenet:
                if network == 'RDMA':
                # if True:
                    if 'id0' in row_split:
                        net_name.append(row_split[1])
                        Bandwidth.append('RDMA_B')
                        worker_num.append(int(row_split[6][-1]))
                        server_num.append(int(row_split[5][-1]))
                    if get_key(row_split[1], 'RDMA_B', row_split[5][-1], row_split[6][-1]) in bcast.keys():
                        bcast[get_key(row_split[1], 'RDMA_B', row_split[5][-1], row_split[6][-1])] += row_list[1]*1e-6*(1/int(row_split[6][-1]))
                        copyh2d[get_key(row_split[1], 'RDMA_B', row_split[5][-1], row_split[6][-1])] += row_list[2]*1e-6*(1/int(row_split[6][-1]))
                        copyd2h[get_key(row_split[1], 'RDMA_B', row_split[5][-1], row_split[6][-1])] += row_list[3]*1e-6*(1/int(row_split[6][-1]))
                        local_reduce[get_key(row_split[1], 'RDMA_B', row_split[5][-1], row_split[6][-1])] += row_list[4]*1e-6*(1/int(row_split[6][-1]))
                        pull[get_key(row_split[1], 'RDMA_B', row_split[5][-1], row_split[6][-1])] += row_list[5]*1e-6*(1/int(row_split[6][-1]))
                        push[get_key(row_split[1], 'RDMA_B', row_split[5][-1], row_split[6][-1])] += row_list[6]*1e-6*(1/int(row_split[6][-1]))
                    else:
                        bcast[get_key(row_split[1], 'RDMA_B', row_split[5][-1], row_split[6][-1])] = row_list[1]*1e-6*(1/int(row_split[6][-1]))
                        copyh2d[get_key(row_split[1], 'RDMA_B', row_split[5][-1], row_split[6][-1])] = row_list[2]*1e-6*(1/int(row_split[6][-1]))
                        copyd2h[get_key(row_split[1], 'RDMA_B', row_split[5][-1], row_split[6][-1])] = row_list[3]*1e-6*(1/int(row_split[6][-1]))
                        local_reduce[get_key(row_split[1], 'RDMA_B', row_split[5][-1], row_split[6][-1])] = row_list[4]*1e-6*(1/int(row_split[6][-1]))
                        pull[get_key(row_split[1], 'RDMA_B', row_split[5][-1], row_split[6][-1])] = row_list[5]*1e-6*(1/int(row_split[6][-1]))
                        push[get_key(row_split[1], 'RDMA_B', row_split[5][-1], row_split[6][-1])] = row_list[6]*1e-6*(1/int(row_split[6][-1]))



Excel_API(file_name, kind)

p0 = [0.01 , 10 / 2400000000, 10 / 2400000000, 0.02, 0.02, 0.1, 0.1, 0.005]
Com_up = np.array(Com_up)
#print(bcast)
bcast = np.array(list(bcast.values()))
#print(bcast)
copyh2d = np.array(list(copyh2d.values()))
copyd2h = np.array(list(copyd2h.values()))
local_reduce = np.array(list(local_reduce.values()))
pull = np.array(list(pull.values()))
push = np.array(list(push.values()))

worker_num = np.array(worker_num)
server_num = np.array(server_num)
bands = []
Size = []

latency_RDMA_account= []
for i in range(len(net_name)):
    Size.append(models[net_name[i]])
    if Bandwidth[i] == 'TCP_B':
        bands.append(1e10)
        latency_RDMA_account.append(1.0)
    elif Bandwidth[i] == 'RDMA_B':
        bands.append(1e11)
        latency_RDMA_account.append(0.1)



Size = np.array(Size)
bands = np.array(bands)

bands.dtype = 'float64'
#Size.dtype = 'float32'
worker_num.dtype = 'int32'
server_num.dtype = 'int32'
if kind == 'bps':
    Com_up.dtype = 'float64'
elif kind == 'trace':
    bcast.dtype = 'float64'
    copyh2d.dtype = 'float64'
    copyd2h.dtype = 'float64'
    local_reduce.dtype = 'float64'
    pull.dtype = 'float64'
    push.dtype = 'float64'


print("len(Size) %d" %len(Size))
print("len(bands) %d" %len(bands))
print("len(workers) %d" %len(worker_num))
print("len(servers) %d" %len(server_num))
print("len(Com_up) %d" %len(Com_up))
print("len(bcast) %d" %len(bcast))
print("len(local_reduce) %d" %len(local_reduce))
print("len(push) %d" %len(push))
print("len(pull) %d" %len(pull))

# xishu = leastsq(err, p0, args=(Size, bands, worker_num, server_num, latency_RDMA_account, Com_up))

# print(xishu[0])




# ===================================================================================
def get_tensor_comm_data(DMLC_PS, same):
    y_tensor_comm_data = []
    Size_tensor_comm = []
    worker_num = []
    server_num = []

    # root_path = 'bps_logs0807_2'
    #for tensor_size in ['512', '2048', '8192']:
    root_path = 'bps_logs0811'
    #for tensor_size in ['512', '1024', '2048', '4096', '8192', '16384']:
    for tensor_size in ['4096', '8192', '16384']:
    #for tensor_size in ['16384']:

    #for tensor_size in ['128', '512', '2048', '8192']:
    #for tensor_size in ['8', '16', '32', '64', '128', '256']:
    #for tensor_size in ['8']:
        # for same in [' ', 'same']:
        if same == 'same':
            legend = DMLC_PS+'-tensor-size' + str(tensor_size) + '-same'
            dir_path = root_path+'/one_tensor_test_same_log'
        else:
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
        # data_Get_Configs.append(data_Get_Config)
        y_tensor_comm_data += get_serialized_log(Data_Get_Config.dir_path, training_or_tensor='tensor', model=Data_Get_Config.model, 
            tensor_size=Data_Get_Config.tensor_size, KB=Data_Get_Config.KB, DMLC_PS=Data_Get_Config.DMLC_PS, 
            batch_size=Data_Get_Config.batch_size, num_iters=Data_Get_Config.num_iters, 
            nworkers=Data_Get_Config.nworkers, nservers=Data_Get_Config.nservers, 
            worker_id=Data_Get_Config.worker_id, local_rank=Data_Get_Config.local_rank)
        Size_tensor_comm += [int(tensor_size)*1024*8]*8
        worker_num += [int('8')]*8
        server_num += [ int(item) for item in ['1', '2', '3', '4', '5', '6', '7', '8'] ]

    root_path = 'bps_logs0807'
    KB='0'
    for tensor_size in ['8', '16', '32', '64', '128', '256']:
    # for tensor_size in ['32', '64', '128', '256']:
        if DMLC_PS == '10.0.0.11':
            root_path = 'bps_logs0811'
            tensor_size = str(int(tensor_size)*1024*4)
            KB='1'
        if same == 'same':
            legend = DMLC_PS+'-tensor-size' + str(tensor_size) + '-same'
            dir_path = root_path+'/one_tensor_test_same_log'
        else:
            legend = DMLC_PS+'-tensor-size' + str(tensor_size)
            dir_path = root_path+'/one_tensor_test_log'
        Data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size], KB=KB,
            DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
            worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend=legend)
        # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
        #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '4', '8'],
        #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '4', '8'], legend=legend)
        # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
        #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['3', '5', '6', '7'],
        #     worker_id=['0'], local_rank=['0'], x_data=['3', '5', '6', '7'], legend=legend)
        # data_Get_Configs.append(data_Get_Config)
        y_tensor_comm_data += get_serialized_log(Data_Get_Config.dir_path, training_or_tensor='tensor', model=Data_Get_Config.model, 
            tensor_size=Data_Get_Config.tensor_size, KB=Data_Get_Config.KB, DMLC_PS=Data_Get_Config.DMLC_PS, 
            batch_size=Data_Get_Config.batch_size, num_iters=Data_Get_Config.num_iters, 
            nworkers=Data_Get_Config.nworkers, nservers=Data_Get_Config.nservers, 
            worker_id=Data_Get_Config.worker_id, local_rank=Data_Get_Config.local_rank)
        if KB == '0':
            Size_tensor_comm += [int(tensor_size)*1024*1024*4*8]*8
        else:
            Size_tensor_comm += [int(tensor_size)*1024*8]*8
        print(tensor_size)
        worker_num += [int('8')]*8
        server_num += [ int(item) for item in ['1', '2', '3', '4', '5', '6', '7', '8'] ]


    #print(Size_tensor_comm)


    return np.array(y_tensor_comm_data), np.array(Size_tensor_comm), np.array(worker_num), np.array(server_num)




def tensor_comm_func(x, alpha_prep, beta_prep, alpha_pci, beta_pci, beta_pci_thres, 
            comp_aggre, alpha, beta, beta_thres, alpha_server_proc, beta_server_proc, beta_server_proc_thres,
            alpha_worker_proc, beta_worker_proc, beta_worker_proc_thres, straggler):
    (Size_tensor_comm, worker_num, server_num) = x

    # t_worker = alpha*server_num + beta*Size
    # t_server = alpha*worker_num + beta*Size*worker_num/server_num

    t_comm = 2 * (alpha*worker_num + beta*Size_tensor_comm*worker_num/server_num)

    return t_comm


def tensor_comm_func_2(x, alpha_prep, beta_prep, alpha_pci, beta_pci, beta_pci_thres, 
            comp_aggre, alpha, beta, beta_thres, alpha_server_proc, beta_server_proc, beta_server_proc_thres,
            alpha_worker_proc, beta_worker_proc, beta_worker_proc_thres, straggler):
    (Size_tensor_comm, worker_num, server_num) = x

    # bandwidth_util linear + constant :
    # beta_broad_util = 1 if Size_tensor_comm > beta_broad_thres else Size_tensor_comm/beta_broad_thres
    # beta_reduce_util = 1 if Size_tensor_comm > beta_reduce_thres else Size_tensor_comm/beta_reduce_thres
    # print(Size_tensor_comm)
    beta_pci_util = np.where(Size_tensor_comm> beta_pci_thres, 1, (Size_tensor_comm)/beta_pci_thres)
    beta_util = np.where(Size_tensor_comm/server_num > beta_thres, 1, Size_tensor_comm/server_num/beta_thres)
    # beta_pci_util = 1 if Size_tensor_comm > beta_pci_thres else Size_tensor_comm/beta_pci_thres
    # beta_util = 1 if Size_tensor_comm > beta_thres else Size_tensor_comm/beta_thres

    # prepare message:
    t_prepare = alpha_prep*(server_num + worker_num) + beta_prep*(Size_tensor_comm + Size_tensor_comm/server_num)

    # broadcast and reduce message:
    # t_broad = alpha_broad*3 + beta_broad*beta_broad_util*Size_tensor_comm
    # t_reduce = alpha_reduce*3 + beta_reduce*beta_reduce_util*Size_tensor_comm
    t_brd_red = alpha_pci*3 + beta_pci/beta_pci_util*Size_tensor_comm

    # push:
    # t_worker = alpha*server_num + beta*Size
    # t_server = alpha*worker_num + beta*Size_tensor_comm*worker_num/server_num

    # server_aggregation:
    t_aggre = comp_aggre*Size_tensor_comm*worker_num/server_num

    # pull:
    t_worker = alpha*server_num + beta/beta_util*Size_tensor_comm
    t_server = alpha*worker_num + beta/beta_util*Size_tensor_comm/server_num

    if beta_pci > beta:
        t_brd_red_pipeline_with_pull = t_brd_red + beta/beta_util*Size_tensor_comm/server_num
    else:
        t_brd_red_pipeline_with_pull = t_brd_red/server_num + t_server

    # a fixed time:
    # t_fixed

    t_comm = t_prepare + t_server + t_brd_red + t_aggre + t_brd_red_pipeline_with_pull
    # t_comm = 2 * t_server + 2 * t_brd_red + t_aggre

    return t_comm

def tensor_comm_func_3(x, alpha_prep, beta_prep, alpha_pci, beta_pci, beta_pci_thres, 
            comp_aggre, alpha, beta, beta_thres, alpha_server_proc, beta_server_proc, beta_server_proc_thres,
            alpha_worker_proc, beta_worker_proc, beta_worker_proc_thres, straggler):
    (Size_tensor_comm, worker_num, server_num) = x
    #print(Size_tensor_comm.max())

    # beta_pci_util = np.where(Size_tensor_comm/server_num > beta_pci_thres, 1, Size_tensor_comm/beta_pci_thres)
    beta_util = np.where(Size_tensor_comm/server_num > beta_thres, 1, (Size_tensor_comm/server_num)/beta_thres)
    # beta_server_proc_util = np.where(Size_tensor_comm/server_num > beta_server_proc_thres, 1, (Size_tensor_comm/server_num)/beta_server_proc_thres)
    # beta_worker_proc_util = np.where(Size_tensor_comm > beta_worker_proc_thres, 1, Size_tensor_comm/beta_worker_proc_thres)


    # t_worker = alpha*server_num + beta*Size
    # t_server = alpha*worker_num + beta/beta_util*Size_tensor_comm*worker_num/server_num
    # t_server = alpha*worker_num + beta/beta_util*Size_tensor_comm*worker_num/server_num
    t_server = alpha*worker_num + beta/beta_util*Size_tensor_comm/server_num

    # t_server_proc = alpha_server_proc*worker_num + beta_server_proc/beta_server_proc_util*Size_tensor_comm*worker_num/server_num
    #t_server_proc = alpha_server_proc*worker_num + beta_server_proc/beta_server_proc_util*Size_tensor_comm*worker_num
    #t_worker_proc = alpha_worker_proc*server_num + beta_worker_proc/beta_worker_proc_util*Size_tensor_comm
    t_server_proc = alpha_server_proc*worker_num + beta_server_proc*Size_tensor_comm*worker_num
    t_worker_proc = alpha_worker_proc*server_num + beta_worker_proc*Size_tensor_comm

    t_straggler = straggler*worker_num*server_num

    t_comm = 2 * t_server + t_server_proc + t_worker_proc + t_straggler

    return t_comm

def comm_func(x, tpsc_2, tpsc_server, synchronization, congestion, tcp_util, rdma_util, PCI_util, latency_tcp, latency_rdma, latency_local, process):

    (Size, bands, worker_num, server_num, latency_RDMA_account) = x
    def gen_rand():
        return 0

    # t_comm = 2*( (3*Size)/(15754000000 * 8) + tpsc_server + (Size/server_num)/bands )
    # t_comm = 2*( (3*Size)/(15754000000 * 8) + tpsc_server + (Size/worker_num)/bands )

    # t_comm = 2*( (4*Size)/(15754000000 * 8)  + (Size.dot(worker_num))/bands) + 4 * tpsc_2 * Size \
    #     + worker_num * tpsc_server * (Size/32) / server_num
    # t_comm = 2*( (4*Size)/(15754000000 * 8)  + (Size*worker_num/server_num)/bands )    \
    #     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_2 * (Size/32) / server_num \
    #     + latency * server_num + latency * worker_num
    # t_comm = 2*( (4*Size)/(15754000000 * 8)  + (Size*worker_num/server_num)/bands )    \
    #     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
    #     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account) 

    # t_comm = 2*( (3*Size)/(15754000000 * 8)  + (Size*worker_num/server_num)/bands )    \
    #     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
    #     + latency * worker_num * server_num.dot(latency_RDMA_account) 

    # t_comm = 2*( (3*Size)/(15754000000 * 8)  + (Size*worker_num)/bands )    \
    #     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
    #     + latency * worker_num * server_num.dot(latency_RDMA_account)

    # t_comm = 2*( (4*Size)/(15754000000 * 8)  + (Size*worker_num/server_num)*band_util/bands )    \
    #     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
    #     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
    #     + synchronization * Size * worker_num / bands + congestion * Size * worker_num /(bands * server_num)

    # t_comm = 2*( (4*Size) /(15754000000 * 8)  + (Size*worker_num/server_num)/bands )    \
    #     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
    #     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
    #     + latency_local * 1 + latency_local * 3 \
    #     + synchronization * Size * worker_num / bands + congestion * Size * worker_num /(bands * server_num)

    # t_comm = 2*( (4*Size) /(15754000000 * 8)  + (Size*worker_num/server_num)/bands )    \
    #     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
    #     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
    #     + synchronization * Size * worker_num / bands + congestion * Size * worker_num /(bands * server_num)

    # t_comm = 2*( (3*Size)* PCI_util /(15754000000 * 8)  + (Size*worker_num/server_num)*band_util/bands)    \
    #     + 3 * tpsc_2 * (Size/32) + (worker_num-1) * tpsc_server * (Size/32) / server_num \
    #     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
    #     + latency_local * 1 + latency_local * 3  + congestion * Size * worker_num /(bands * server_num)\
    #     + synchronization*(Size*worker_num/server_num)*band_util/bands + synchronization*(4*Size)* PCI_util /(15754000000 * 8) \

    # t_comm = 2*( (3*Size)* PCI_util /(15754000000 * 8)  + (Size*worker_num)*band_util/bands)    \
    #     + 3 * tpsc_2 * (Size/32) + (worker_num-1) * tpsc_server * (Size/32) / server_num \
    #     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
    #     + latency_local * 1 + latency_local * 3  + congestion \
    #     + synchronization    # seed=100, 21%
    t_comm = 2*( (3*Size)* PCI_util /(15754000000 * 8)  + (Size*worker_num)*tcp_util/bands)    \
        + 3 * tpsc_2 * (Size/32) + (worker_num-1) * tpsc_server * (Size/32) / server_num \
        + latency_tcp * server_num + latency_tcp * worker_num  \
        + latency_local * 1 + latency_local * 3  + congestion \
        + synchronization    # seed=100, 21%
    # t_comm = 2*( (3*Size)* PCI_util /(15754000000 * 8)  + (Size*worker_num)*rdma_util/bands)    \
    #     + 3 * tpsc_2 * (Size/32) + (worker_num-1) * tpsc_server * (Size/32) / server_num \
    #     + latency_rdma * server_num + latency_rdma * worker_num  \
    #     + latency_local * 1 + latency_local * 3  + congestion \
    #     + synchronization    # seed=100, 21%
    # t_comm = 2*( (3*Size)* PCI_util /(15754000000 * 8)  + (Size*worker_num)*band_util/bands * server_num)    \
    #     + 3 * tpsc_2 * (Size/32) + (worker_num-1) * tpsc_2 * (Size/32) / server_num \
    #     + latency *6 \
    #     + synchronization    # seed=100, 21%
    # t_comm = 2*( (4*Size) /(15754000000 * 8)  + (Size*worker_num/server_num)/bands)    \
    #     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
    #     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
    #     + latency_local * 1 + latency_local * 3  + congestion * Size * worker_num /(bands * server_num)\
    #     + synchronization*(Size*worker_num/server_num)/bands + synchronization*(4*Size) /(15754000000 * 8) 

    # t_comm = 2*( (3*Size) /(15754000000 * 8)  + (Size*worker_num)/bands)    \
    #     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
    #     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
    #     + latency_local * 1 + latency_local * 3 \
    #     + congestion * Size * worker_num /(bands * server_num) + congestion * Size /(bands)\
    #     + synchronization*(Size*(worker_num-1)/server_num)/bands + synchronization*(3*Size) /(15754000000 * 8) \
    #     + process * Size                   # seed=2 22.3%

    # t_comm = 2*( (4*Size)* PCI_util /(15754000000 * 8)  + (Size*worker_num/server_num)*band_util/bands )    \
    #     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
    #     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
    #     + latency_local * 1 + latency_local * 3 \
    #     + synchronization * synchronization * Size * worker_num / bands + congestion * Size * worker_num /(bands * server_num * latency)

    #print(t_comm)
    return t_comm

def local_reduce_func(x, tpsc_2, tpsc_server, synchronization, congestion, tcp_util, rdma_util, PCI_util, latency_tcp, latency_rdma, latency_local, process):

    (Size, bands, worker_num, server_num, latency_RDMA_account) = x
    def gen_rand():
        return 0

    # t_local_reduce = (3*Size)* PCI_util /(15754000000 * 8) + latency_local * 1 + 3 * tpsc_2 * (Size/32) # seed=100, 21%
    # t_local_reduce = (3*Size)* PCI_util /(15754000000 * 8) + latency_local * 1 # seed=100, 21%
    t_local_reduce = (3*Size)* PCI_util /(15754000000 * 8) + latency_local * 1 # seed=100, 21%

    return t_local_reduce

def push_func(x, tpsc_2, tpsc_server, synchronization, congestion, tcp_util, rdma_util, PCI_util, latency_tcp, latency_rdma, latency_local, process):

    (Size, bands, worker_num, server_num, latency_RDMA_account) = x
    def gen_rand():
        return 0

    # t_push = (Size*worker_num)*band_util/bands + latency * server_num.dot(latency_RDMA_account) # seed=100, 21%
    t_push = (Size*worker_num)*tcp_util/bands + latency_tcp * server_num # seed=100, 21%

    return t_push

def pull_func(x, tpsc_2, tpsc_server, synchronization, congestion, tcp_util, rdma_util, PCI_util, latency_tcp, latency_rdma, latency_local, process):

    (Size, bands, worker_num, server_num, latency_RDMA_account) = x
    def gen_rand():
        return 0

    # t_pull = (Size*worker_num)*band_util/bands + latency * worker_num.dot(latency_RDMA_account)  # seed=100, 21%
    t_pull = (Size*worker_num)*tcp_util/bands + latency_tcp * worker_num  # seed=100, 21%

    return t_pull

def bcast_func(x, tpsc_2, tpsc_server, synchronization, congestion, tcp_util, rdma_util, PCI_util, latency_tcp, latency_rdma, latency_local, process):

    (Size, bands, worker_num, server_num, latency_RDMA_account) = x
    def gen_rand():
        return 0

    t_bcast = (3*Size)* PCI_util /(15754000000 * 8) + latency_local * 3  # seed=100, 21%

    return t_bcast



def fit(func, fit_name, params, params_bound, params_origin, DMLC_PS='192.168.0.11', same=' ', save_path=None):
    if fit_name == 'tensor_comm':
        print("===============================")
        # params = [alpha_prep, beta_prep, alpha_pci, beta_pci, beta_pci_util, beta_pci_thres, 
        #     comp_aggre, alpha, beta, beta_util, beta_thres]
        y = []
        Size_tensor_comm = []
        worker_num = []
        server_num = []
        for i_DMLC_PS in DMLC_PS:
            for i_same in same:
                y_i, Size_tensor_comm_i, worker_num_i, server_num_i = get_tensor_comm_data(DMLC_PS=i_DMLC_PS, same=i_same)
                # print(Size_tensor_comm_i.max())
                print("len(Size_tensor_comm_i) %d" %len(Size_tensor_comm_i))
                print("len(worker_num_i) %d" %len(worker_num_i))
                print("len(server_num_i) %d" %len(server_num_i))
                print("len(y_i) %d" %len(y_i))
                y = np.append(y, y_i)
                Size_tensor_comm = np.append(Size_tensor_comm, Size_tensor_comm_i)
                worker_num = np.append(worker_num, worker_num_i)
                server_num = np.append(server_num, server_num_i)
        print("len(Size_tensor_comm) %d" %len(Size_tensor_comm))
        print("len(worker_num) %d" %len(worker_num))
        print("len(server_num) %d" %len(server_num))
        print("len(y) %d" %len(y))
        x = np.stack([Size_tensor_comm, worker_num, server_num])

        def sumOfSquaredError(parameterTuple):
            warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
            val = func(x, *parameterTuple)
            return np.sum((y - val) ** 2.0)

        parameterBounds_left = []
        parameterBounds_right = []
        def generate_Initial_Parameters():
            # ([0.0000005, 1e-9, 1e-9, 0.0, 0.0, 0.5, 0.5, 0.0000005, 0.0], [1., 1., 1., 1., 1., 1., 1., 1., 1.])
            parameterBounds = []
            # parameterBounds.append([0, 10.]) # seach bounds for alpha
            # parameterBounds.append([0, 10.]) # seach bounds for betta

            for value in params_bound.values():
                parameterBounds.append([value[0], value[1]])
                parameterBounds_left.append(value[0])
                parameterBounds_right.append(value[1])
            # "seed" the numpy random number generator for repeatable results
            result = differential_evolution(sumOfSquaredError, parameterBounds, seed=50)
            return result.x


        # param_bounds=([np.inf, 1],[np.inf,1])
        #popt, pcov = optimize.curve_fit(func, x, y, bounds=([0.0005, 1e-9, 1e-9, 0.1, 0.1, 0.0, 0.0, 0.0005], [1., 1., 1., 5., 5., 1., 1., 1.]))
        # popt, pcov = optimize.curve_fit(func, x, y, p0=np.asarray([0.001 , 10 / 2400000000, 10 / 2400000000, 0.5, 0.5, 0.75, 0.75, 0.001, 0.01]),
        #     bounds=([0.0005, 1e-9, 1e-9, 0.0, 0.0, 0.5, 0.5, 0.0005, 0.0], [1., 1., 1., 1., 1., 1., 1., 1., 1.]))   31%


        # popt, pcov = optimize.curve_fit(func, x, y, p0=np.asarray([0.00001 , 1e-7, 1e-7, 0.5, 0.5, 0.75, 0.75, 0.00001, 0.01]),
        #     bounds=([0.0000005, 1e-9, 1e-9, 0.0, 0.0, 0.5, 0.5, 0.0000005, 0.0], [1., 1., 1., 1., 1., 1., 1., 1., 1.]))

        # generate initial parameter values
        geneticParameters = generate_Initial_Parameters()

        # curve fit the test data
        # popt, pcov = optimize.curve_fit(func, x, y, geneticParameters, 
        #     bounds=([0, 0], [10., 10.]))
        origin_params = []
        for value in params_origin.values():
            origin_params.append(value)

        # popt, pcov = optimize.curve_fit(func, x, y, geneticParameters, 
        #     bounds=(parameterBounds_left, parameterBounds_right))
        popt, pcov = optimize.curve_fit(func, x, y, geneticParameters, 
            bounds=(parameterBounds_left, parameterBounds_right), method='trf')
        # popt, pcov = optimize.curve_fit(func, x, y, p0=origin_params, 
        #     bounds=(parameterBounds_left, parameterBounds_right), method='trf')  # dogbox trf
        # popt, pcov = optimize.curve_fit(func, x, y, geneticParameters, 
        #     bounds=([1e-9, 1e-9, 0.0, 0.0, 0.9, 0.9, 0.9, 0.0000005, 0.0000005, 0.0000005, 0.0], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))

        # ==============================================================================================
        # def mse(predict, y):
        #     return np.mean((predict - y)**2)
        # def object_func(params):
        #     predict = func(x, *params)
        #     return mse(predict, y)
        # result = optimize.basinhopping(object_func, x0=origin_params, niter=1000)
        # print(result)
        # popt = result['x']
        # ==============================================================================================

        for i, key in enumerate(params.keys()):
            params[key] = popt[i]

        # params['alpha_prep'], params['beta_prep'], params['alpha_pci'], params['beta_pci'], params['beta_pci_util'], params['beta_pci_thres'], 
        #     params['comp_aggre'], params['alpha'], params['beta'], params['beta_util'], params['beta_thres'] = tuple(popt)
        # alpha, beta = tuple(popt)
        # latency, tpsc_2, tpsc_server, synchronization, congestion, band_util, PCI_util, latency_local, process = tuple(popt)
        # print(tuple(popt))

        # print("alpha : %s" % alpha)
        # print("beta : %s" % beta)
        for key in params.keys():
            print("%s: %s" % (key, params[key]))



        predict = func(x, **params)
        # predict = func(x, alpha, beta)
        # print("predict:",predict)
        # print("real:",y)

        error = predict - y
        # print("error: %s" % error)
        # print("all accuracy: %s" % (np.fabs(error/y)))
        print("Fiting: %s network-%s-%s====================" % (fit_name, DMLC_PS, same))
        print("loss: %f" % np.mean(np.fabs(error)))
        print("error: %f %% " % np.mean(np.fabs(error/y)*100) )
        print("===============================")

        model_params = {}
        model_params['params'] = params
        model_params['error'] = np.mean(np.fabs(error/y)*100)
        model_params['DMLC_PS'] = DMLC_PS
        model_params['same'] = same
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(model_params, f)
                f.close()
        return


    x = np.stack([Size, bands, worker_num, server_num, latency_RDMA_account])

    if fit_name == 'Com_up':
        y = Com_up #+ np.random.uniform(0, 0.5, (len(Size)))/2
    elif fit_name == 'bcast':
        y = bcast
    elif fit_name == 'local_reduce':
        y = local_reduce
    elif fit_name == 'pull':
        y = pull
    elif fit_name == 'push':
        y = push

    # print(len(y),len(x))
    def sumOfSquaredError(parameterTuple):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        val = func(x, *parameterTuple)
        return np.sum((y - val) ** 2.0)

    def generate_Initial_Parameters():
        # ([0.0000005, 1e-9, 1e-9, 0.0, 0.0, 0.5, 0.5, 0.0000005, 0.0], [1., 1., 1., 1., 1., 1., 1., 1., 1.])
        parameterBounds = []

        parameterBounds.append([1e-9, 1.]) # seach bounds for b
        parameterBounds.append([1e-9, 1.]) # seach bounds for c
        parameterBounds.append([0.0, 1.]) # seach bounds for a
        parameterBounds.append([0.0, 1.]) # seach bounds for b
        parameterBounds.append([0.9, 1.]) # seach bounds for c
        parameterBounds.append([0.9, 1.]) # seach bounds for a
        parameterBounds.append([0.9, 1.]) # seach bounds for a        
        parameterBounds.append([0.0000005, 1.]) # seach bounds for a
        parameterBounds.append([0.0000005, 1.]) # seach bounds for a
        parameterBounds.append([0.0000005, 1.]) # seach bounds for b
        parameterBounds.append([0.0, 1.]) # seach bounds for c

        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=1000)
        return result.x


    # param_bounds=([np.inf, 1],[np.inf,1])
    #popt, pcov = optimize.curve_fit(func, x, y, bounds=([0.0005, 1e-9, 1e-9, 0.1, 0.1, 0.0, 0.0, 0.0005], [1., 1., 1., 5., 5., 1., 1., 1.]))
    # popt, pcov = optimize.curve_fit(func, x, y, p0=np.asarray([0.001 , 10 / 2400000000, 10 / 2400000000, 0.5, 0.5, 0.75, 0.75, 0.001, 0.01]),
    #     bounds=([0.0005, 1e-9, 1e-9, 0.0, 0.0, 0.5, 0.5, 0.0005, 0.0], [1., 1., 1., 1., 1., 1., 1., 1., 1.]))   31%


    # popt, pcov = optimize.curve_fit(func, x, y, p0=np.asarray([0.00001 , 1e-7, 1e-7, 0.5, 0.5, 0.75, 0.75, 0.00001, 0.01]),
    #     bounds=([0.0000005, 1e-9, 1e-9, 0.0, 0.0, 0.5, 0.5, 0.0000005, 0.0], [1., 1., 1., 1., 1., 1., 1., 1., 1.]))

    # generate initial parameter values
    geneticParameters = generate_Initial_Parameters()

    # curve fit the test data
    popt, pcov = optimize.curve_fit(func, x, y, geneticParameters, 
        bounds=([1e-9, 1e-9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0000005, 0.0000005, 0.0000005, 0.0], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    # popt, pcov = optimize.curve_fit(func, x, y, geneticParameters, 
    #     bounds=([1e-9, 1e-9, 0.0, 0.0, 0.9, 0.9, 0.9, 0.0000005, 0.0000005, 0.0000005, 0.0], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))

    tpsc_2, tpsc_server, synchronization, congestion, tcp_util, rdma_util, PCI_util, latency_tcp, latency_rdma, latency_local, process = tuple(popt)
    # latency, tpsc_2, tpsc_server, synchronization, congestion, band_util, PCI_util, latency_local, process = tuple(popt)
    # print(tuple(popt))
    print("tpsc_2 : %s" % tpsc_2)
    print("tpsc_server : %s" % tpsc_server)
    print("synchronization : %s" % synchronization)
    print("congestion : %s" % congestion)
    # print("band_util : %s" % band_util)
    print("tcp_util : %s" % tcp_util)
    print("rdma_util : %s" % rdma_util)
    print("PCI_util : %s" % PCI_util)
    print("latency_tcp : %s" % latency_tcp)
    print("latency_rdma : %s" % latency_rdma)
    print("latency_local : %s" % latency_local)
    print("process:%s" % process)

    predict = func(x, tpsc_2, tpsc_server, synchronization, congestion, tcp_util, rdma_util, PCI_util, latency_tcp, latency_rdma, latency_local, process)

    # print("predict:",predict)
    # print("real:",y)

    error = predict - y
    # print("error: %s" % error)
    # print("all accuracy: %s" % (np.fabs(error/y)))
    print("Fiting: %s ====================" % fit_name)
    print("loss: %f" % np.mean(np.fabs(error)))
    print("error: %f %%" % np.mean(np.fabs(error/y))*100 )
    print("===============================")


def show_fit_curve(func, params, DMLC_PS, same):
    legend_location = 'upper right'
    subplots_adjust = [0.16, 0.15, 0.96, 0.95]
    data_Get_Configs = []

    markers=['o','*']
    #markers=['.','x','o','v','^','<','>']

    colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'darkgreen', 'darkblue', 'brown', 'darkorange']
    #colors = colors[2:7]
    #colors = colors[0:4]
    colors = colors[0:12]
    markeriter = itertools.cycle(markers)
    coloriter = itertools.cycle(colors)


    root_path = 'bps_logs0807'
    KB='0'
    #root_path = 'bps_logs0807_2'
    #KB='1'
    #for tensor_size in ['512', '2048', '8192']:
    #for tensor_size in ['128', '512', '2048', '8192']:
    #for tensor_size in ['32', '64', '128', '256']:
    for tensor_size in ['8', '16', '32', '64', '128', '256']:
    #for tensor_size in ['8']:
        if DMLC_PS == '10.0.0.11':
            root_path = 'bps_logs0811'
            tensor_size = str(int(tensor_size)*1024*4)
            KB='1'
        if same == 'same':
            legend = DMLC_PS+'-tensor-size' + str(tensor_size) + '-same'
            dir_path = root_path+'/one_tensor_test_same_log'
        else:
            legend = DMLC_PS+'-tensor-size' + str(tensor_size)
            dir_path = root_path+'/one_tensor_test_log'
        Data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size], KB=KB,
            DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
            worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend=legend)
        # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
        #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '4', '8'],
        #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '4', '8'], legend=legend)
        # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
        #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['3', '5', '6', '7'],
        #     worker_id=['0'], local_rank=['0'], x_data=['3', '5', '6', '7'], legend=legend)
        y_data = get_serialized_log(Data_Get_Config.dir_path, training_or_tensor='tensor', model=Data_Get_Config.model, 
            tensor_size=Data_Get_Config.tensor_size, KB=Data_Get_Config.KB, DMLC_PS=Data_Get_Config.DMLC_PS, 
            batch_size=Data_Get_Config.batch_size, num_iters=Data_Get_Config.num_iters, 
            nworkers=Data_Get_Config.nworkers, nservers=Data_Get_Config.nservers, 
            worker_id=Data_Get_Config.worker_id, local_rank=Data_Get_Config.local_rank)
        Data_Get_Config.y_data=y_data
        color = next(coloriter)
        Data_Get_Config.color = color
        Data_Get_Config.marker = next(markeriter)
        data_Get_Configs.append(Data_Get_Config)

        legend_fit = legend + '-fit'
        Data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size], KB=KB,
            DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
            worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend=legend_fit)
        # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
        #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '4', '8'],
        #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '4', '8'], legend=legend)
        # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
        #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['3', '5', '6', '7'],
        #     worker_id=['0'], local_rank=['0'], x_data=['3', '5', '6', '7'], legend=legend)
        if KB == '0':
            Size_tensor_comm = np.array([int(tensor_size)*1024*1024*4*8]*8)
        else:
            Size_tensor_comm = np.array([int(tensor_size)*1024*8]*8)
        worker_num = np.array([int('8')]*8)
        server_num = np.array([int(item) for item in ['1', '2', '3', '4', '5', '6', '7', '8']])
        x = np.stack([Size_tensor_comm, worker_num, server_num])
        # y_data = func(x, alpha, beta)
        y_data = func(x, **params)
        Data_Get_Config.y_data=y_data
        Data_Get_Config.color = color
        Data_Get_Config.marker = next(markeriter)
        data_Get_Configs.append(Data_Get_Config)


    # =================================================================================
    KB='1'

    # root_path = 'bps_logs0807_2'
    # for tensor_size in ['512', '2048', '8192']:
    root_path = 'bps_logs0811'
    # for tensor_size in ['512', '1024', '2048', '4096', '8192', '16384']:
    for tensor_size in ['4096', '8192', '16384']:
    #for tensor_size in ['16384']:
    #for tensor_size in ['8']:
        if same == 'same':
            legend = DMLC_PS+'-tensor-size' + str(tensor_size) + '-same'
            dir_path = root_path+'/one_tensor_test_same_log'
        else:
            legend = DMLC_PS+'-tensor-size' + str(tensor_size)
            dir_path = root_path+'/one_tensor_test_log'
        Data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size], KB=KB,
            DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
            worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend=legend)
        # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
        #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '4', '8'],
        #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '4', '8'], legend=legend)
        # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
        #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['3', '5', '6', '7'],
        #     worker_id=['0'], local_rank=['0'], x_data=['3', '5', '6', '7'], legend=legend)
        y_data = get_serialized_log(Data_Get_Config.dir_path, training_or_tensor='tensor', model=Data_Get_Config.model, 
            tensor_size=Data_Get_Config.tensor_size, KB=Data_Get_Config.KB, DMLC_PS=Data_Get_Config.DMLC_PS, 
            batch_size=Data_Get_Config.batch_size, num_iters=Data_Get_Config.num_iters, 
            nworkers=Data_Get_Config.nworkers, nservers=Data_Get_Config.nservers, 
            worker_id=Data_Get_Config.worker_id, local_rank=Data_Get_Config.local_rank)
        Data_Get_Config.y_data=y_data
        color = next(coloriter)
        Data_Get_Config.color = color
        Data_Get_Config.marker = next(markeriter)
        data_Get_Configs.append(Data_Get_Config)

        legend_fit = legend + '-fit'
        Data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size], KB=KB,
            DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '3', '4', '5', '6', '7', '8'],
            worker_id=['0'], local_rank=['0'], x_data=['1', '2', '3', '4', '5', '6', '7', '8'], legend=legend_fit)
        # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
        #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['1', '2', '4', '8'],
        #     worker_id=['0'], local_rank=['0'], x_data=['1', '2', '4', '8'], legend=legend)
        # data_Get_Config = u.Data_Get_Config(dir_path=dir_path, model=[' '], tensor_size=[tensor_size],
        #     DMLC_PS=[DMLC_PS], batch_size=['64'], num_iters=['55'], nworkers=['8'], nservers=['3', '5', '6', '7'],
        #     worker_id=['0'], local_rank=['0'], x_data=['3', '5', '6', '7'], legend=legend)
        #Size_tensor_comm = np.array([int(tensor_size)*1024*1024*4*8]*8)
        Size_tensor_comm = np.array([int(tensor_size)*1024*8]*8)
        worker_num = np.array([int('8')]*8)
        server_num = np.array([int(item) for item in ['1', '2', '3', '4', '5', '6', '7', '8']])
        x = np.stack([Size_tensor_comm, worker_num, server_num])
        # y_data = func(x, alpha, beta)
        y_data = func(x, **params)
        Data_Get_Config.y_data=y_data
        Data_Get_Config.color = color
        Data_Get_Config.marker = next(markeriter)
        data_Get_Configs.append(Data_Get_Config)


    file_path = './tensor_comm.pdf'
    plot.plot_figure(data_Get_Configs, training_or_tensor='tensor', title=None, x_label='servers', y_label='Time (Seconds)',
         file_path=file_path, legend_location=legend_location, subplots_adjust=subplots_adjust)



# params = {'alpha_prep':0, 'beta_prep':0, 'alpha_pci':0, 'beta_pci':0, 'beta_pci_util':0, 'beta_pci_thres':0, 
#     'comp_aggre':0, 'alpha':0, 'beta':0, 'beta_util':0, 'beta_thres':0}
params = {'alpha_prep':0, 'beta_prep':0, 'alpha_pci':0, 'beta_pci':0, 'beta_pci_thres':0, 
    'comp_aggre':0, 'alpha':0, 'beta':0, 'beta_thres':0, 'alpha_server_proc':0, 'beta_server_proc':0, 'beta_server_proc_thres':0,
            'alpha_worker_proc':0, 'beta_worker_proc':0, 'beta_worker_proc_thres':0, 'straggler':0}
params_bound = {'alpha_prep':(0.000005, 0.005), 'beta_prep':(0, 5e-6), 'alpha_pci':(0.000005, 0.005), 'beta_pci':(1.0/1.5*1e-10, 5e-8),
    'beta_pci_thres':(10000, 1e7), 'comp_aggre':(0, 5e-6), 'alpha':(0, 0.00005), 'beta':(8.0*1e-11, 5e-8), 'beta_thres':(10000, 1e7), 
    'alpha_server_proc':(0, 0.005), 'beta_server_proc':(0, 5e-6), 'beta_server_proc_thres':(10000, 1e7),
    'alpha_worker_proc':(0, 0.005), 'beta_worker_proc':(0, 5e-6), 'beta_worker_proc_thres':(10000, 1e7), 'straggler':(0, 1e-3)}

# params_bound = {'alpha_prep':(0, 1), 'beta_prep':(0, 1), 'alpha_pci':(0, 1), 'beta_pci':(0, 1),
#     'beta_pci_thres':(100000, 1e8), 'comp_aggre':(0, 1), 'alpha':(0, 1), 'beta':(0, 1), 'beta_thres':(10000, 1e8), 
#     'alpha_server_proc':(0, 1), 'beta_server_proc':(0, 1), 'beta_server_proc_thres':(10000, 1e8),
#     'alpha_worker_proc':(0, 1), 'beta_worker_proc':(0, 1), 'beta_worker_proc_thres':(10000, 1e8), 'straggler':(0, 1)}

params_origin = {'alpha_prep':0.00005, 'beta_prep':5e-9, 'alpha_pci':0.00005, 'beta_pci':5e-9,
    'beta_pci_thres':100000, 'comp_aggre':1e-8, 'alpha':0.00005, 'beta':5e-10, 'beta_thres':100000, 
    'alpha_server_proc':0.00005, 'beta_server_proc':5e-9, 'beta_server_proc_thres':100000,
    'alpha_worker_proc':0.00005, 'beta_worker_proc':5e-9, 'beta_worker_proc_thres':100000, 'straggler':5e-5}

for key in params_bound.keys():
    # print(params_origin[key])
    # print(params_bound[key])
    if params_origin[key] > params_bound[key][1] or params_origin[key] < params_bound[key][0]:
        print("origin_params error: %s" % key)

        # parameterBounds_left = [0.000005, 5e-10, 0.00005, 1.0*1e-11, 100000, 5e-10, 0.0005, 1.0*1e-10, 100000]
        # parameterBounds_right = [0.005, 5e-6, 0.005, 5e-6, 1e8, 5e-6, 0.005, 5e-6, 1e8]

if __name__ == '__main__':
    #DMLC_PS= ['10.0.0.11', '192.168.0.11']
    # same = [' ', 'same']
    #DMLC_PS= ['10.0.0.11']
    DMLC_PS= ['192.168.0.11']
    same = ['same']
    save_path = 'model_params/model_params_tensor_comm1.json'
    # fit(tensor_comm_func_2, 'tensor_comm', params, DMLC_PS=DMLC_PS, same=same, save_path=save_path)

    # '10.0.0.11', '192.168.0.11'
    # ' ', 'same'
    fit(tensor_comm_func_2, 'tensor_comm', params, params_bound, params_origin, DMLC_PS=DMLC_PS, same=same, save_path=save_path)
    #fit(tensor_comm_func, 'tensor_comm', params, DMLC_PS=DMLC_PS, same=same, save_path=save_path)

    #tensor_comm_func_2

    with open(save_path) as f:
        model_params = json.load(f)
        params = model_params['params']
        for DMLC_PS_i in DMLC_PS:
            for same_i in same:
                show_fit_curve(tensor_comm_func_2, params, DMLC_PS_i, same_i)
        f.close()

    # fit_name = 'pull' # 'bcast', 'local_reduce', 'pull', 'push', Com_up 'tensor_comm'
    # fit(comm_func, 'Com_up')
    #fit(bcast_func, 'bcast')
    #fit(local_reduce_func, 'local_reduce')
    #fit(pull_func, 'pull')
    #fit(push_func, 'push',)






















