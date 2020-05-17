directory=/home/esetstore/yxwang/bps_bench
cd $directory
export NVIDIA_VISIBLE_DEVICES=0,1,2,3  
export DMLC_WORKER_ID=$worker_id
export DMLC_NUM_WORKER=$n_worker
export DMLC_ROLE=worker 
export DMLC_NUM_SERVER=$n_server
export DMLC_PS_ROOT_URI=$scheduler_ip
export DMLC_PS_ROOT_PORT=$scheduler_port
export DMLC_ENABLE_RDMA=1
export DMLC_INTERFACE=ib0
export BYTEPS_SERVER_ENABLE_SCHEDULE=0
export BYTEPS_PARTITION_BYTES=512000000
#export BYTEPS_PARTITION_BYTES=256000000
source ~/pytorch1.4/bin/activate
model=$model 
source exp_config/$model.conf

batch_size=$batch_size
num_iters=$num_iters

#echo $batch_size

export BYTEPS_TRACE_ON=1
export BYTEPS_TRACE_END_STEP=15
export BYTEPS_TRACE_START_STEP=5
export BYTEPS_TRACE_DIR="./traces_${model}_network${DMLC_PS_ROOT_URI}_bs${batch_size}_iters${num_iters}_servers${n_server}_workers${n_worker}_id${worker_id}"
echo $BYTEPS_TRACE_DIR
#bpslaunch python3 /home/esetstore/yxwang/byteps/example/pytorch/benchmark_byteps.py --model alexnet --batch-size $batch_size --num-iters $num_iters
bpslaunch python3 benchmark_byteps.py --model $model --DMLC-PS ${DMLC_PS_ROOT_URI} --nworkers $n_worker --nservers $n_server --batch-size $batch_size --num-iters $num_iters --worker-id $worker_id




