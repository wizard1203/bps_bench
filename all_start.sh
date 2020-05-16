directory=/home/esetstore/yxwang/bps_bench
echo $directory

n_server=1
n_worker=1
n_scheduler=1

#scheduler_ip=10.0.0.13
scheduler_ip=192.168.0.19
scheduler_port=1234

model=alexnet

worker_id=0

source net_config/scheduler_${n_scheduler}.conf
for number in "${remotehosts[@]}"
do
    host=gpu$number
    #echo $host
    args="n_server=$n_server n_worker=$n_worker n_scheduler=$n_scheduler scheduler_ip=$scheduler_ip scheduler_port=$scheduler_port bash start_scheduler.sh"
    cmd="cd $directory; $args" 
    echo $host
    echo $cmd
    ssh $host $cmd &
done

source net_config/server_${n_server}.conf
for number in "${remotehosts[@]}"
do
    host=gpu$number
    #echo $host
    args="n_server=$n_server n_worker=$n_worker n_scheduler=$n_scheduler scheduler_ip=$scheduler_ip scheduler_port=$scheduler_port bash start_server.sh"
    cmd="cd $directory; $args" 
    echo $host
    echo $cmd
    ssh $host $cmd &
done

source net_config/worker_${n_worker}.conf
for number in "${remotehosts[@]}"
do
    host=gpu$number
    #echo $host
    args="n_server=$n_server n_worker=$n_worker n_scheduler=$n_scheduler scheduler_ip=$scheduler_ip scheduler_port=$scheduler_port worker_id=$worker_id  model=$model bash start_worker.sh"
    cmd="cd $directory; $args"
    echo $host
    echo $cmd 
    ssh $host $cmd &
    worker_id=$(expr $worker_id + 1)
done












