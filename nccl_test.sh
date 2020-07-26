directory=/home/esetstore/yxwang/nccl-tests/build
save_dic=/home/esetstore/yxwang/bps_test/nccl_results
echo $directory


remotehosts=("7" "9" "11" "12" "13" "14" "15" "16")



for number in "${remotehosts[@]}"
do
    host=gpu$number
    #echo $host
    args="./broadcast_perf -b 1024 -e 1024M -f 2 -g 4 -n 200 > ${save_dic}/gpu${number}_brd.txt"
    cmd="cd $directory; $args" 
    echo $host
    echo $cmd
    ssh $host $cmd &
done


for number in "${remotehosts[@]}"
do
    host=gpu$number
    #echo $host
    args="./reduce_perf -b 1024 -e 1024M -f 2 -g 4 -n 200 > ${save_dic}/gpu${number}_reduce.txt"
    cmd="cd $directory; $args" 
    echo $host
    echo $cmd
    #ssh $host $cmd &
done


