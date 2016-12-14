#Shell Options
#!/bin/bash
#        
#Machine Options
#
#Log File
#$ -j y
#$ -o /farmshare/user_data/edgard/Logs/
#$ -e /farmshare/user_data/edgard/Logs/
#
#Send an e-mail to edgard when job is aborted
#$ -M edgard@stanford.edu
#$ -m ae
#
##Computer parameters 
CS229_dir="/farmshare/user_data/edgard/CS229-TetrisIsAwesome/PAPER_TEST/"
cd ${CS229_dir}

#python Boltz-f=1/async_dqn_RAM_Boltzmann.py
#python  Boltz-f=1-2/async_dqn_RAM_Boltzmann.py
python Boltz-f=1-3/async_dqn_RAM_Boltzmann.py
#python Epsilon/async_dqn_RAM.py
