qsub -v frameskip='True',update='1',linearNet='True' trainingNormal.sh
qsub -v frameskip='False',update='1',linearNet='True' trainingNormal.sh
qsub -v frameskip='True',update='1',linearNet='False' trainingNormal.sh
qsub -v frameskip='False',update='1',linearNet='False' trainingNormal.sh 
