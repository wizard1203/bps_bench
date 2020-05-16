ps -ef|grep byteps|grep -v grep|cut -c 9-15|xargs kill -9
ps -ef|grep bpslaunch|grep -v grep|cut -c 9-15|xargs kill -9
