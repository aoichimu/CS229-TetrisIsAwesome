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
CS229_dir="../CS229-TetrisIsAwesome/CS-229-RL/"
cd ${CS229_dir}

#echo ${frameskip}
#echo ${update}
#echo ${linearNet}

python simpleDQN.py -fs ${frameskip} -update ${update} -net ${linearNet}
