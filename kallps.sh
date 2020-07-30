directory=/home/esetstore/yxwang/bps_bench
max=16
#for ((number=9;number <= max;number++))
#do
#    host=gpu$number
#    ssh $host bash $directory/killps.sh &
#done

remotehosts=("7" "9" "11" "12" "13" "14" "15" "16")
for number in "${remotehosts[@]}"
do
    host=gpu$number
    addr=`expr $number + 10`
    host=192.168.0.$addr
    ssh $host bash $directory/killps.sh &
done




