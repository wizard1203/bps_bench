import math

TCP_B = 10000000000  #10Gb
RDMA_B = 100000000000   # 100Gb
Alexnet = 60965128 * 32 #     243860512MB number of params * bits 22
Resnet50 = 46159168 * 32 #    184636672MB
Densenet121 = 7978856 * 32 #  31915424MB
Vgg16 = 138344128 * 32 #      553376512MB

eth = {'TCP_B':TCP_B, 'RDMA_B':RDMA_B}
models = {'Alexnet':60965128, 'Resnet50':46159168, 'Densenet121':7978856, 'Vgg16':138344128}
# PCI = 16000000000 * 8  #16GB
PCI = 15754000000 * 8  #15.754 GB/s
# tpsc_cpu = 0.05 
# tpsc_2 = 0.05 
latency = 0.35
RDMA_latency_acount = 0.05
tpsc_cpu = 50 / 2400000000 # CPU frequency, estimated operation time of one param
tpsc_2 = 50 / 2400000000
servers = 4
workers = 4
Bandwidth_acount = 1
Size = models['Alexnet']

#net = 'TCP_B'
net = 'RDMA_B'
Bandwidth = eth[net] * Bandwidth_acount
if net != 'TCP_B':
    latency = latency * RDMA_latency_acount

for servers in range(1,workers+1):
    #t_comm = 2*( (3*Size)/PCI + tpsc + (Size/servers)/Bandwidth )
    #t_comm = 2*( (4*Size)/PCI  + (Size*workers)/Bandwidth ) + 4 * tpsc_2 * (Size/32) + workers * tpsc_cpu * (Size/32) / servers
    # t_comm = 2*( (4*Size)/PCI  + (Size*workers/servers)/Bandwidth )    \
    #     + 4 * tpsc_2 * (Size/32) + workers * tpsc_cpu * (Size/32) / servers \
    #     + latency * servers + latency * workers
    t_comm = 2*( (4*Size)/PCI  + (Size*workers)/Bandwidth )    \
        + 4 * tpsc_2 * (Size/32) + workers * tpsc_cpu * (Size/32) / servers \
        + latency * servers + latency * workers

    print("Model: %s, bandwidth: %d, servers: %d, workers: %d, latency: %f, \n" \
        %('Alexnet', Bandwidth, servers, workers, latency))
    print("t_comm: %.10f" %(t_comm))
    # print("workers * tpsc_cpu * (Size/32): %f, 2*(Size*workers)/Bandwidth + latency: %f"\
    #      % (workers * tpsc_cpu * (Size/32), 2*(Size*workers)/Bandwidth + latency))
    # print("server min value: %f"\
    #      %(math.sqrt((workers * tpsc_cpu * (Size/32)))/(2*(Size*workers)/Bandwidth + latency)))


