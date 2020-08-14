directory=/home/esetstore/yxwang/bps_test
cd $directory
export NVIDIA_VISIBLE_DEVICES=0,1,2,3  
export DMLC_WORKER_ID=$worker_id
export DMLC_NUM_WORKER=$n_worker
export DMLC_ROLE=worker 
export DMLC_NUM_SERVER=$n_server
export DMLC_PS_ROOT_URI=$scheduler_ip
export DMLC_PS_ROOT_PORT=$scheduler_port

export BYTEPS_REDUCE_ROOTS=1

if [ $rdma -eq 1 ]; then
    export DMLC_ENABLE_RDMA=1
    export DMLC_INTERFACE=ib0
fi

#export BYTEPS_SERVER_ENABLE_SCHEDULE=1
export BYTEPS_PARTITION_BYTES=$partition_size 
#export BYTEPS_PARTITION_BYTES=256000000
source ~/pytorch1.4/bin/activate
model=$model 
source exp_config/$model.conf
tensorsize=$tensorsize

batch_size=$batch_size
num_iters=$num_iters

#echo $batch_size

export BYTEPS_TRACE_ON=1
export BYTEPS_TRACE_END_STEP=50   #50
export BYTEPS_TRACE_START_STEP=1   #10
if [ $whole_grad ];then
    whole_grad_path=whole_grad
else
    whole_grad_path=layerwise
fi


if [ $same -eq 1 ]; then
    export BYTEPS_TRACE_DIR="./traces_same_${whole_grad_path}/traces_same_${model}_network${DMLC_PS_ROOT_URI}_bs${batch_size}_iters${num_iters}_servers${n_server}_workers${n_worker}_id${worker_id}"
else
    export BYTEPS_TRACE_DIR="./traces_${whole_grad_path}/traces_${model}_network${DMLC_PS_ROOT_URI}_bs${batch_size}_iters${num_iters}_servers${n_server}_workers${n_worker}_id${worker_id}"
fi
#export BYTEPS_TRACE_DIR="./traces_${model}_network${DMLC_PS_ROOT_URI}_bs${batch_size}_iters${num_iters}_servers${n_server}_workers${n_worker}_id${worker_id}"
export CUDA_HOME=/home/esetstore/yxwang/cuda-10.0
export LD_LIBRARY=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
echo $BYTEPS_TRACE_DIR
echo $CUDA_HOME
#bpslaunch python3 /home/esetstore/yxwang/byteps/example/pytorch/benchmark_byteps.py --model alexnet --batch-size $batch_size --num-iters $num_iters
#bpslaunch python3 benchmark_byteps.py --same $same --model $model --whole-grad $whole_grad --DMLC-PS ${DMLC_PS_ROOT_URI} --nworkers $n_worker --nservers $n_server --batch-size $batch_size --num-iters $num_iters --worker-id $worker_id
if [ $whole_grad ];then
    bpslaunch python3 benchmark_byteps.py --same $same --model $model --whole-grad --DMLC-PS ${DMLC_PS_ROOT_URI} --nworkers $n_worker --nservers $n_server --batch-size $batch_size --num-iters $num_iters --worker-id $worker_id
else
    bpslaunch python3 benchmark_byteps.py --same $same --model $model --DMLC-PS ${DMLC_PS_ROOT_URI} --nworkers $n_worker --nservers $n_server --batch-size $batch_size --num-iters $num_iters --worker-id $worker_id
fi



#if [ $same -eq 1 ]; then
#    export BYTEPS_TRACE_DIR="./traces_same_onlytensor/traces_onlytensor_size_${tensorsize}KB_network${DMLC_PS_ROOT_URI}_bs${batch_size}_iters${num_iters}_servers${n_server}_workers${n_worker}_id${worker_id}"
#else
#    export BYTEPS_TRACE_DIR="./traces_onlytensor/traces_onlytensor_size_${tensorsize}KB_network${DMLC_PS_ROOT_URI}_bs${batch_size}_iters${num_iters}_servers${n_server}_workers${n_worker}_id${worker_id}"
#fi
#export BYTEPS_TRACE_DIR="./traces_onlytensor/traces_onlytensor_size_${tensorsize}KB_network${DMLC_PS_ROOT_URI}_bs${batch_size}_iters${num_iters}_servers${n_server}_workers${n_worker}_id${worker_id}"
#bpslaunch python3 tensor_byteps.py --same $same --DMLC-PS ${DMLC_PS_ROOT_URI} --nworkers $n_worker --nservers $n_server --worker-id $worker_id --tensor-size $tensorsize




