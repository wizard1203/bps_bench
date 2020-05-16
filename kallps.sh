directory=/home/esetstore/yxwang/bps_bench
max=10
for ((number=9;number <= max;number++))
do
    host=gpu$number
    ssh $host bash $directory/killps.sh
done




