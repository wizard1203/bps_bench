directory=/home/esetstore/yxwang
max=10
for ((number=1;number <= max;number++))
do
    host=gpu$number
    ssh $host bash $directory/killps.sh
done




