directory=/home/esetstore/yxwang/bps_test
echo $directory

n_server=1
n_worker=4
n_scheduler=1

#scheduler_ip=10.0.0.19
#scheduler_ip=192.168.0.23
scheduler_ip=10.0.0.23
#scheduler_ip=192.168.0.19
scheduler_port=1234

#model=alexnet
#model_size=244403360   #243860512

model=resnet50
model_size=102228128   #184636672

#model=vgg16
#model_size=553430176  #553376512

#model=densenet121
#model_size=31915424  #31915424

#partition_size=`expr $model_size / $n_server + 1` 

# one_tensor test   *1024 * 1024 * 4
tensorsize=4
partition_size=`expr $tensorsize \* 1024 \* 1024 \* 4 / $n_server`


#partition_size=`expr $tensorsize \* 1024 \* 1024 \* 4 / $n_server - 1 \* 1024 \* 1024 \* 4 \* 3 / 2`


echo $partition_size
#partition_size=`expr $model_size / 10 + 1` 



worker_id=0

source net_config/scheduler_${n_scheduler}.conf
for number in "${remotehosts[@]}"
do
    host=gpu$number
    addr=`expr $number + 10`
    host=192.168.0.$addr
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
    addr=`expr $number + 10`
    host=192.168.0.$addr
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
    addr=`expr $number + 10`
    host=192.168.0.$addr
     #echo $host
    args="n_server=$n_server n_worker=$n_worker n_scheduler=$n_scheduler scheduler_ip=$scheduler_ip scheduler_port=$scheduler_port worker_id=$worker_id  model=$model tensorsize=$tensorsize partition_size=$partition_size bash start_worker.sh"
    cmd="cd $directory; $args"
    echo $host
    echo $cmd 
    ssh $host $cmd &
    worker_id=$(expr $worker_id + 1)
done












