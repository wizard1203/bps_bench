export DMLC_NUM_WORKER=$n_worker
export DMLC_ROLE=server
export DMLC_NUM_SERVER=$n_server
export DMLC_PS_ROOT_URI=$scheduler_ip
export DMLC_PS_ROOT_PORT=$scheduler_port
#export BYTEPS_PARTITION_BYTES=512000000
#export BYTEPS_PARTITION_BYTES=256000000

#export DMLC_ENABLE_RDMA=0
#export DMLC_INTERFACE=ib0
if [ $rdma -eq 1 ]; then
    export DMLC_ENABLE_RDMA=1
    export DMLC_INTERFACE=ib0
fi

#export BYTEPS_SERVER_ENABLE_SCHEDULE=1
export BYTEPS_SERVER_ENGINE_THREAD=8
# echo $DMLC_PS_ROOT_URI
source ~/pytorch1.4/bin/activate
export CUDA_HOME=/home/esetstore/yxwang/cuda-10.0
export LD_LIBRARY=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
bpslaunch
