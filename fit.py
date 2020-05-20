import numpy as np 
from scipy import optimize
from scipy.optimize import leastsq
import numpy as np
import xlrd
from scipy.optimize import differential_evolution
import warnings

file_name = './bps_logs/bps.xls'
TCP_B = 10000000000  #10Gb
RDMA_B = 100000000000   # 100Gb
Alexnet = 60965128 * 32 #     243860512MB number of params * bits 22
Resnet50 = 46159168 * 32 #    184636672MB
Densenet121 = 7978856 * 32 #  31915424MB
Vgg16 = 138344128 * 32 #      553376512MB
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
choosenet = 'densenet121'
def Excel_API(file_name):
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
			if True:
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
			if True:
				net_name.append(row_split[0])
				Bandwidth.append('RDMA_B')
				worker_num.append(int(row_split[4][-1]))
				server_num.append(int(row_split[5][8]))
				Com_up.append(row_list[4]*1e-8)
				# print(net_name[-1])
				# print(Bandwidth[-1])
				# print(worker_num[-1])
				# print(server_num[-1])
				# print(Com_up[-1])


Excel_API(file_name)
p0 = [0.01 , 10 / 2400000000, 10 / 2400000000, 0.02, 0.02, 0.1, 0.1, 0.005]
Com_up = np.array(Com_up)

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
Com_up.dtype = 'float64'

# xishu = leastsq(err, p0, args=(Size, bands, worker_num, server_num, latency_RDMA_account, Com_up))

# print(xishu[0])


def func(x, latency, tpsc_2, tpsc_server, synchronization, congestion, band_util, PCI_util, latency_local, process):

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

    t_comm = 2*( (3*Size)* PCI_util /(15754000000 * 8)  + (Size*worker_num)*band_util/bands)    \
        + 3 * tpsc_2 * (Size/32) + (worker_num-1) * tpsc_server * (Size/32) / server_num \
        + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
        + latency_local * 1 + latency_local * 3  + congestion \
        + synchronization    # seed=100, 21%
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

# print("len(Size) %d" %len(Size))
# print("len(bands) %d" %len(bands))
# print("len(workers) %d" %len(worker_num))
# print("len(servers) %d" %len(server_num))
# print("len(Com_up) %d" %len(Com_up))

x = np.stack([Size, bands, worker_num, server_num, latency_RDMA_account])
y = Com_up #+ np.random.uniform(0, 0.5, (len(Size)))/2

def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    val = func(x, *parameterTuple)
    return np.sum((y - val) ** 2.0)

def generate_Initial_Parameters():
    # ([0.0000005, 1e-9, 1e-9, 0.0, 0.0, 0.5, 0.5, 0.0000005, 0.0], [1., 1., 1., 1., 1., 1., 1., 1., 1.])
    parameterBounds = []
    parameterBounds.append([0.0000005, 1.]) # seach bounds for a
    parameterBounds.append([1e-9, 1.]) # seach bounds for b
    parameterBounds.append([1e-9, 1.]) # seach bounds for c
    parameterBounds.append([0.0, 1.]) # seach bounds for a
    parameterBounds.append([0.0, 1.]) # seach bounds for b
    parameterBounds.append([0.5, 1.]) # seach bounds for c
    parameterBounds.append([0.5, 1.]) # seach bounds for a
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
    bounds=([0.0000005, 1e-9, 1e-9, 0.0, 0.0, 0.5, 0.5, 0.0000005, 0.0], [1., 1., 1., 1., 1., 1., 1., 1., 1.]))


latency, tpsc_2, tpsc_server, synchronization, congestion, band_util, PCI_util, latency_local, process = tuple(popt)
# print(tuple(popt))
print("latency : %s" % latency)
print("tpsc_2 : %s" % tpsc_2)
print("tpsc_server : %s" % tpsc_server)
print("synchronization : %s" % synchronization)
print("congestion : %s" % congestion)
print("band_util : %s" % band_util)
print("PCI_util : %s" % PCI_util)
print("latency_local : %s" % latency_local)
print("process:%s" % process)
#predict_comm = 2*( (3*Size)/(15754000000 * 8) + tpsc_server + (Size/server_num)/bands )
# predict_comm = 2*( (4*Size)/(15754000000 * 8)  + (Size.dot(worker_num))/bands) + 4 * tpsc_2 * Size\
# 	+ worker_num * tpsc_server * (Size/32) / server_num
# predict_comm = 2*( (4*Size)/(15754000000 * 8)  + (Size*workers/server_num)/bands )    \
# 	+ 4 * tpsc_2 * (Size/32) + worker_num * tpsc_2 * (Size/32) / server_num \
# 	+ latency * server_num + latency * workers
# predict_comm = 2*( (4*Size)/(15754000000 * 8)  + (Size*worker_num/server_num)/bands )    \
# 	+ 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
# 	+ latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account) 

# predict_comm = 2*( (4*Size)/(15754000000 * 8)  + (Size*worker_num/server_num)/bands )    \
# 	+ 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num 

# predict_comm = 2*( (3*Size)/(15754000000 * 8)  + (Size*worker_num/server_num)/bands )    \
# 	+ 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
# 	+ latency * worker_num * server_num.dot(latency_RDMA_account) 

# predict_comm = 2*( (3*Size)/(15754000000 * 8)  + (Size*worker_num)/bands )    \
# 	+ 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
# 	+ latency * worker_num * server_num.dot(latency_RDMA_account) 

# predict_comm = 2*( (4*Size)/(15754000000 * 8)  + (Size*worker_num/server_num)/bands )    \
# 	+ 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
# 	+ latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
# 	+ synchronization * Size * worker_num / bands + congestion * Size * worker_num /(bands * server_num)

# predict_comm = 2*( (4*Size)* PCI_util /(15754000000 * 8)  + (Size*worker_num/server_num)*band_util/bands)    \
#     + 3 * tpsc_2 * (Size/32) + (worker_num-1) * tpsc_server * (Size/32) / server_num \
#     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
#     + latency_local * 1 + latency_local * 3  + congestion * Size * worker_num /(bands * server_num)\
#     + synchronization*(Size*worker_num/server_num)*band_util/bands + synchronization*(4*Size)* PCI_util /(15754000000 * 8) \

# predict_comm = 2*( (4*Size) /(15754000000 * 8)  + (Size*worker_num/server_num)/bands)    \
#     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
#     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
#     + latency_local * 1 + latency_local * 3  + congestion * Size * worker_num /(bands * server_num)\
#     + synchronization*(Size*worker_num/server_num)*band_util/bands + synchronization*(4*Size)* PCI_util /(15754000000 * 8) 
# predict_comm = 2*( (4*Size) /(15754000000 * 8)  + (Size*worker_num)/bands)    \
#     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
#     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
#     + latency_local * 1 + latency_local * 3 \
#     + congestion * Size * worker_num /(bands * server_num) + congestion * Size /(bands)\
#     + synchronization*(Size*(worker_num-1)/server_num)/bands + synchronization*(3*Size) /(15754000000 * 8) \
#     + process * Size

# predict_comm = 2*( (4*Size) /(15754000000 * 8)  + (Size*worker_num)/bands)    \
#     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
#     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
#     + latency_local * 1 + latency_local * 3  + congestion * Size * worker_num /(bands * server_num)\
#     + synchronization*(Size*worker_num/server_num)/bands + synchronization*(4*Size) /(15754000000 * 8) 

# predict_comm = 2*( (4*Size)* PCI_util /(15754000000 * 8)  + (Size*worker_num/server_num)*band_util/bands )    \
#     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
#     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
#     + latency_local * 1 + latency_local * 3 + latency_local*latency_local +  synchronization * latency_local\
#     + synchronization * synchronization * Size * worker_num / bands + congestion * Size * worker_num /(bands * server_num * latency)

# predict_comm = 2*( (4*Size) /(15754000000 * 8)  + (Size*worker_num/server_num)/bands )    \
# 	+ 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
# 	+ latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
# 	+ synchronization * Size * worker_num / bands + congestion * Size * worker_num /(bands * server_num)

# predict_comm = 2*( (3*Size)* PCI_util /(15754000000 * 8)  + (Size*worker_num)*band_util/bands)    \
#     + 3 * tpsc_2 * (Size/32) + (worker_num-1) * tpsc_server * (Size/32) / server_num \
#     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
#     + latency_local * 1 + latency_local * 3  + congestion \
#     + synchronization

predict_comm = func(x, latency, tpsc_2, tpsc_server, synchronization, congestion, band_util, PCI_util, latency_local, process)
print("predict:",predict_comm)
print("real:",Com_up)

error = predict_comm - Com_up
print("error: %s" % error)
print("all accuracy: %s" % (np.fabs(error/Com_up)))
print("loss: %f" % np.mean(np.fabs(error)))
print("accuracy: %f" % np.mean(np.fabs(error/Com_up)))























