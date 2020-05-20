from scipy.optimize import leastsq
import numpy as np
import xlrd

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
			if row_split[0] == 'alexnet':
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
			if row_split[0] == 'alexnet':
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

# def fun(p, Net, Band, workers, servers):
#     """
#     定义想要拟合的函数
#     """
#     latency, tpsc_2 = p    # 从参数p获得拟合的参数
#     Size = []
#     Bandwidth = []
#     for i in range(len(net_name)):
#     	Size.append(models[net_name[i]])
#     	if Band[i] == 'TCP_B':
#     		Bandwidth.append(1e10)
#     	elif Band[i] == 'RDMA_B':
#     		Bandwidth.append(1e11)
#     print("len(Size) %d" %len(Size))
#     print("len(Bandwidth) %d" %len(Bandwidth))
#     print("len(workers) %d" %len(workers))
#     print("len(servers) %d" %len(servers))
#     Size = np.array(Size)
#     Bandwidth = np.array(Bandwidth)
#     #workers = np.array(workers)
#     #servers = np.array(servers)
#     # print(servers_1)
#     t_comm = 2*( (4*Size)/(15754000000 * 8)  + (Size*workers/servers)/Bandwidth )    \
#         + 4 * tpsc_2 * (Size/32) + workers * tpsc_2 * (Size/32) / servers \
#         + latency * servers + latency * workers
#     print(t_comm)
#     return t_comm

def fun(p, Size, bands, worker_num, server_num, latency_RDMA_account):
    """
    定义想要拟合的函数
    """
    # print("len(Size) %d" %len(Size))
    # print("len(bands) %d" %len(bands))
    # print("len(workers) %d" %len(workers))
    # print("len(servers) %d" %len(servers))

    latency, tpsc_2, tpsc_server, synchronization, congestion, band_util, PCI_util, latency_local = p    # 从参数p获得拟合的参数
    # t_comm = 2*( (3*Size)/(15754000000 * 8) + tpsc_server + (Size/server_num)/bands )

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

    # t_comm = 2*( (4*Size)* PCI_util /(15754000000 * 8)  + (Size*worker_num/server_num)*band_util/bands )    \
    #     + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
    #     + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
	# 	+ latency_local * 1 + latency_local * 3 \
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

    t_comm = 2*( (4*Size) /(15754000000 * 8)  + (Size*worker_num/server_num)/bands)    \
        + 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
        + latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
        + latency_local * 1 + latency_local * 3  + congestion * Size * worker_num /(bands * server_num)\
        + synchronization*(Size*worker_num/server_num)/bands + synchronization*(4*Size) /(15754000000 * 8) 

    #print(t_comm)
    return t_comm

def err(p, x1, x2, x3, x4, x5, y):
    # print("len(x1) %d" %len(x1))
    # print("len(x2) %d" %len(x2))
    # print("len(x3) %d" %len(x3))
    # print("len(x4) %d" %len(x4))
    # print("len(y) %d" %len(y))
    return fun(p,x1,x2,x3,x4,x5) - y

Excel_API(file_name)
# print(Size)
# #定义起始的参数 即从 y = 1*x+1 开始，其实这个值可以随便设，只不过会影响到找到最优解的时间
p0 = [0.01 , 10 / 2400000000, 10 / 2400000000, 0.02, 0.02, 0.1, 0.1, 0.005]

# #将list类型转换为 numpy.ndarray 类型，最初我直接使用
# #list 类型,结果 leastsq函数报错，后来在别的blog上看到了，原来要将类型转
# #换为numpy的类型

# x1 = np.array([150,200,250,300,350,400,600])    # 面积
# x2 = np.array([4,2,7,9,12,14,15])               # 楼层
# y1 = np.array([6450,7450,8450,9450,11450,15450,18450])   # 价格/平方米
Com_up = np.array(Com_up)


# print(net_name)
# print(worker_num)
# print(server_num)
# print(Bandwidth)
# print(Com_up)
# print('len(net_name) :%d ' % len(net_name))
# print('len(worker_num):%d' % len(worker_num))
# print('len(server_num):%d' % len(server_num))
# print('len(Bandwidth):%d' % len(Bandwidth))
# print('len(Com_up):%d' % len(Com_up))

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
# print("len(Size) %d" %len(Size))
# print("len(bands) %d" %len(bands))
# print("len(workers) %d" %len(worker_num))
# print("len(servers) %d" %len(server_num))
# print("len(Com_up) %d" %len(Com_up))
Size = np.array(Size)
bands = np.array(bands)

# print("bands.dtype %s"%bands.dtype)
# print("Size.dtype %s"%Size.dtype)
# print("worker_num.dtype %s"%worker_num.dtype)
# print("server_num.dtype %s"%server_num.dtype)
# print("Com_up.dtype %s"%Com_up.dtype)
# print("bands: %s" % bands)
# print("Size: %s" % Size)
# print("worker_num: %s" % worker_num)
# print("server_num: %s" % server_num)
# print("Com_up: %s" % Com_up)
bands.dtype = 'float64'
#Size.dtype = 'float32'
worker_num.dtype = 'int32'
server_num.dtype = 'int32'
Com_up.dtype = 'float64'
# print("bands: %s" % bands)
# print("Size: %s" % Size)
# print("worker_num: %s" % worker_num)
# print("server_num: %s" % server_num)
# print("Com_up: %s" % Com_up)
# print('latency_RDMA_account: %s' % latency_RDMA_account)
#xishu = leastsq(err, p0, args=(net_name, Bandwidth, worker_num, server_num, Com_up))
xishu = leastsq(err, p0, args=(Size, bands, worker_num, server_num, latency_RDMA_account, Com_up))

print(xishu[0])

latency, tpsc_2, tpsc_server, synchronization, congestion, band_util, PCI_util, latency_local = xishu[0]
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

# predict_comm = 2*( (4*Size)* PCI_util /(15754000000 * 8)  + (Size*worker_num/server_num)*band_util/bands )    \
# 	+ 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
# 	+ latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
# 	+ synchronization * Size * worker_num / bands + congestion * Size * worker_num /(bands * server_num)

# predict_comm = 2*( (4*Size) /(15754000000 * 8)  + (Size*worker_num/server_num)/bands )    \
# 	+ 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
# 	+ latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
# 	+ synchronization * Size * worker_num / bands + congestion * Size * worker_num /(bands * server_num)

predict_comm = 2*( (4*Size) /(15754000000 * 8)  + (Size*worker_num/server_num)/bands)    \
	+ 4 * tpsc_2 * (Size/32) + worker_num * tpsc_server * (Size/32) / server_num \
	+ latency * server_num.dot(latency_RDMA_account) + latency * worker_num.dot(latency_RDMA_account)  \
	+ latency_local * 1 + latency_local * 3  + congestion * Size * worker_num /(bands * server_num)\
	+ synchronization*(Size*worker_num/server_num)/bands + synchronization*(4*Size) /(15754000000 * 8) 

print("predict:",predict_comm)
print("real:",Com_up)

error = predict_comm - Com_up
print("error: %s" % error)
print("loss: %f" % np.mean(np.fabs(error)))
print("accuracy: %f" % np.mean(np.fabs(error/Com_up)))

