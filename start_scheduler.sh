export DMLC_NUM_WORKER=$n_worker
export DMLC_ROLE=scheduler 
export DMLC_NUM_SERVER=$n_server
export DMLC_PS_ROOT_URI=$scheduler_ip
export DMLC_PS_ROOT_PORT=$scheduler_port
echo $DMLC_PS_ROOT_PORT
# export BYTEPS_PARTITION_BYTES=1024000
# bpslaunch
