import numpy as np 
from scipy import optimize
from scipy.optimize import leastsq
import numpy as np
import xlrd
from scipy.optimize import differential_evolution
import warnings


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



def fit(func, fit_name):
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
        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=100)
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
    print("accuracy: %f" % np.mean(np.fabs(error/y)))
    print("===============================")

# fit_name = 'pull' # 'bcast', 'local_reduce', 'pull', 'push', Com_up
# fit(comm_func, 'Com_up')
fit(bcast_func, 'bcast')
fit(local_reduce_func, 'local_reduce')
fit(pull_func, 'pull')
fit(push_func, 'push',)






















