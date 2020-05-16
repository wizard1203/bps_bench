export NVIDIA_VISIBLE_DEVICES=0,1,2,3  
export DMLC_WORKER_ID=$worker_id
export DMLC_NUM_WORKER=$n_worker
export DMLC_ROLE=worker 
export DMLC_NUM_SERVER=$n_server
export DMLC_PS_ROOT_URI=$scheduler_ip
export DMLC_PS_ROOT_PORT=$scheduler_port
# export BYTEPS_PARTITION_BYTES=10240

echo $BYTEPS_TRACE_DIR
model=$model 
source exp_config/$model.conf

batch_size=$batch_size
num_iters=$num_iters

echo $batch_size

export BYTEPS_TRACE_ON=1
export BYTEPS_TRACE_END_STEP=20
export BYTEPS_TRACE_START_STEP=10
export BYTEPS_TRACE_DIR="./traces_${model}_bs${batch_size}_iters${num_iters}_servers${n_server}_workers${n_worker}_id${worker_id}"
# bpslaunch python /home/esetstore/yxwang/byteps/example/pytorch/benchmark_byteps.py --model alexnet --batch-size $batch_size --num-iters $num_iters



