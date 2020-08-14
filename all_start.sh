directory=/home/esetstore/yxwang/bps_test
echo $directory

n_server=1   #1  
n_worker=8
n_scheduler=1
same=1

rdma=1
## one_tensor test   *1024 * 1024 * 4
# on_tensor test * 256 * 4

#tensorsize=256

# should be le 128, cause the byteps will limit it to the PAGE_SIZE * 4
# = 16K , if 8 servers, 128 K. ! 128 has problems, should be 512
#tensorsize=1024   #  512, 1024, 2048      KB
tensorsize=`expr 256 \* 1024 \* 4` # x * 1024 * 4  # 1, 2 ,4 ,8 16 32 64 128 256

#partition_size=`expr $tensorsize \* 256 \* 4 / $n_server`
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

whole_grad=true
model=alexnet
model_size=244403360   #243860512

#model=resnet50
#model_size=102228128   #184636672

#model=vgg16
#model_size=553430176  #553376512

#model=densenet121
#model_size=31915424  #31915424

#partition_size=`expr $model_size / $n_server + 1` 
#partition_size=4096000
#partition_size=1024000



#partition_size=`expr $tensorsize \* 1024 \* 1024 \* 4 / $n_server - 1 \* 1024 \* 1024 \* 4 \* 3 / 2`


#partition_size=`expr $model_size / 10 + 1` 


for model in alexnet resnet50 vgg16 densenet121
do
case $model in
    alexnet)
        model_size=244403360
        ;;
    resnet50)
        model_size=102228128
        ;;
    vgg16)
        model_size=553430176
        ;;
    densenet121)
        model_size=31915424
        ;;
esac
for n_server in 1 2 3 4 5 6 7 8
#for n_server in 8
do
worker_id=0
partition_size=`expr $model_size / $n_server + 1` 
echo $partition_size


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
    args="rdma=$rdma n_server=$n_server n_worker=$n_worker n_scheduler=$n_scheduler scheduler_ip=$scheduler_ip scheduler_port=$scheduler_port worker_id=$worker_id  model=$model tensorsize=$tensorsize partition_size=$partition_size same=$same whole_grad=$whole_grad bash start_worker.sh"
    cmd="cd $directory; $args"
    echo $host
    echo $cmd 
    ssh $host $cmd &
    #if [ $number -eq 16 ]; then
    #    ssh $host $cmd &
    #else
    #    ssh $host $cmd &
    #fi
    worker_id=$(expr $worker_id + 1)
done
wait
bash $directory/kallps.sh
wait
sleep 5s

done
done









