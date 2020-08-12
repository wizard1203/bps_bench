directory=/home/esetstore/yxwang/bps_test
cd $directory




model=alexnet
model=resnet50

#export BYTEPS_SERVER_ENABLE_SCHEDULE=1
export BYTEPS_PARTITION_BYTES=$partition_size 
#export BYTEPS_PARTITION_BYTES=256000000
source ~/pytorch1.4/bin/activate
source exp_config/$model.conf
tensorsize=$tensorsize

batch_size=$batch_size
num_iters=$num_iters

#echo $batch_size



#export BYTEPS_TRACE_DIR="./traces_${model}_network${DMLC_PS_ROOT_URI}_bs${batch_size}_iters${num_iters}_servers${n_server}_workers${n_worker}_id${worker_id}"
export CUDA_HOME=/home/esetstore/yxwang/cuda-10.0
export LD_LIBRARY=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

echo $CUDA_HOME
#bpslaunch python3 /home/esetstore/yxwang/byteps/example/pytorch/benchmark_byteps.py --model alexnet --batch-size $batch_size --num-iters $num_iters
python3 profiling.py --model $model --batch-size $batch_size --num-iters $num_iters 
