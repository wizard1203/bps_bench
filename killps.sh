ps -ef|grep byteps|grep -v grep|cut -c 9-15|xargs kill -9
ps -ef|grep bpslaunch|grep -v grep|cut -c 9-15|xargs kill -9
ps -ef|grep iperf|grep -v grep|cut -c 9-15|xargs kill -9
ps -ef|grep ib_read_bw|grep -v grep|cut -c 9-15|xargs kill -9
ps -ef|grep imagenet|grep -v grep|cut -c 9-15|xargs kill -9



