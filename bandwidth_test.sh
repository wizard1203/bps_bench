directory=/home/esetstore/yxwang/bps_test
echo $directory

all_hosts=("2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16")


if_tcp=1


server_number=2 # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16


if [ $if_tcp -eq 1 ]; then
    addr=`expr $server_number + 10`
    server_host=192.168.0.$addr
    #echo $host
    args="iperf -s"
    cmd="cd $directory; $args" 
    echo $server_host
    echo $cmd
    ssh $server_host $cmd &
else
    addr=`expr $server_number + 10`
    server_host=10.0.0.$addr
    #echo $host
    args="ib_read_bw"
    cmd="cd $directory; $args" 
    echo $server_host
    echo $cmd
    ssh $server_host $cmd &
fi


for host_number in "${all_hosts[@]}"
do 
    if [ $host_number -eq $server_number ]; then
        continue
    fi

    if [ $if_tcp -eq 1 ]; then
        addr=`expr $host_number + 10`
        host=192.168.0.$addr
        #echo $host
        args="iperf -c $server_host > bandwidth_results/tcp-c-${host_number}-s-${server_number}.log"
        cmd="cd $directory; $args" 
        echo $host
        echo $cmd
        ssh $host $cmd &
    else
        addr=`expr $host_number + 10`
        host=10.0.0.$addr
        #echo $host
        args="ib_read_bw $server_host > bandwidth_results/rdma-c-${host_number}-s-${server_number}.log"
        cmd="cd $directory; $args" 
        echo $host
        echo $cmd
        ssh $host $cmd &
    fi
done










