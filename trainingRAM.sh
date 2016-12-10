#Shell Options
#!/bin/bash
#        
#Machine Options
#
#Log File
#$ -j y
#$ -o /farmshare/user_data/jiaming/Logs/
#$ -e /farmshare/user_data/jiaming/Logs/
#
#Send an e-mail to jiaming when job is aborted
#$ -M jiaming@stanford.edu
#$ -m ae
#
##Computer parameters 
CS229_dir="/farmshare/user_data/jiaming/CS229-TetrisIsAwesome/CS-229-RL/ADQN_RAM/"
cd ${CS229_dir}

python async_dqn_RAM.py --experiment ${exp} --num_concurrent ${threads} --game ${game}
