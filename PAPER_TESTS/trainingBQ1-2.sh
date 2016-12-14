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


#python ./Boltz-f=1/async_dqn.py
python ./Boltz-f=1-2/trainingBQ1-2.py
#python ./Boltz-f=1-3/trainingBQ1-3.py
#python ./Epsilon/trainingE.py
