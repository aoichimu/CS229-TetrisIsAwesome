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
CS229_dir="/farmshare/user_data/jiaming/CS229-TetrisIsAwesome/CS-229-RL/ADQN/async-rl-master/"
cd ${CS229_dir}

python async_dqn_loss.py --experiment ${exp} --num_concurrent ${threads}
