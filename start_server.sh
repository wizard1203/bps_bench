export DMLC_NUM_WORKER=$n_worker
export DMLC_ROLE=server
export DMLC_NUM_SERVER=$n_server
export DMLC_PS_ROOT_URI=$scheduler_ip
export DMLC_PS_ROOT_PORT=$scheduler_port
export BYTEPS_PARTITION_BYTES=512000000
export BYTEPS_SERVER_ENABLE_SCHEDULE=0
# echo $DMLC_PS_ROOT_URI
source ~/pytorch1.4/bin/activate
bpslaunch
