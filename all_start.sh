directory=/home/esetstore/yxwang/bps_test
echo $directory

n_server=8   #1  
n_worker=8
n_scheduler=1
same=1

rdma=1
## one_tensor test   *1024 * 1024 * 4
# on_tensor test * 256 * 4

#tensorsize=256

# should be le 128, cause the byteps will limit it to the PAGE_SIZE * 4
# = 16K , if 8 servers, 128 K.
tensorsize=8192   # 128, 512, 2048, 8192  KB

partition_size=`expr $tensorsize \* 256 \* 4 / $n_server`
#partition_size=10

#partition_size=`expr $tensorsize \* 1024 \* 1024 \* 4 / $n_server`

#partition_size=`expr $tensorsize \* 1024 \* 1024 \* 4 / 1`


if [ $rdma -eq 1 ]; then
    scheduler_ip=10.0.0.11
else
    scheduler_ip=192.168.0.11
fi

#scheduler_ip=10.0.0.19
#scheduler_ip=192.168.0.23
#scheduler_ip=10.0.0.23
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
#partition_size=4096000
#partition_size=1024000



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
    args="rdma=$rdma n_server=$n_server n_worker=$n_worker n_scheduler=$n_scheduler scheduler_ip=$scheduler_ip scheduler_port=$scheduler_port bash start_scheduler.sh"
    cmd="cd $directory; $args" 
    echo $host
    echo $cmd
    ssh $host $cmd &
done

#source net_config/server_${n_server}.conf
#for number in "${remotehosts[@]}"
if [ $same -eq 1 ]; then
    source net_config/servers_same.conf
else
    source net_config/servers.conf
fi
eval remote_servers=\(\${remote_server_${n_server}[*]}\)
for number in "${remote_servers[@]}"
do
    host=gpu$number
    addr=`expr $number + 10`
    host=192.168.0.$addr
    #echo $host
    args="rdma=$rdma n_server=$n_server n_worker=$n_worker n_scheduler=$n_scheduler scheduler_ip=$scheduler_ip scheduler_port=$scheduler_port bash start_server.sh"
    cmd="cd $directory; $args" 
    echo $host
    echo $cmd
    ssh $host $cmd &
done

#source net_config/worker_${n_worker}.conf
#for number in "${remotehosts[@]}"
source net_config/workers.conf
eval remote_workers=\(\${remote_worker_${n_worker}[*]}\)
for number in "${remote_workers[@]}"
do
    host=gpu$number
    addr=`expr $number + 10`
    host=192.168.0.$addr
     #echo $host
    args="rdma=$rdma n_server=$n_server n_worker=$n_worker n_scheduler=$n_scheduler scheduler_ip=$scheduler_ip scheduler_port=$scheduler_port worker_id=$worker_id  model=$model tensorsize=$tensorsize partition_size=$partition_size same=$same  bash start_worker.sh"
    cmd="cd $directory; $args"
    echo $host
    echo $cmd 
    ssh $host $cmd &
    worker_id=$(expr $worker_id + 1)
done












