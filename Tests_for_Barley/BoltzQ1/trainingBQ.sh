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
#Send an e-mail to edgard when job is aborted
#$ -M jiaming@stanford.edu
#$ -m ae
#
##Computer parameters 
CS229_dir="/farmshare/user_data/jiaming/CS229-TetrisIsAwesome/Tests_for_Barley/BoltzQ1/"
cd ${CS229_dir}

python async_dqn_RAM_Boltzmann.py
#python  Boltz-f=1-2/async_dqn_RAM_Boltzmann.py
#python Boltz-f=1-3/async_dqn_RAM_Boltzmann.py
#python Epsilon/async_dqn_RAM.py
