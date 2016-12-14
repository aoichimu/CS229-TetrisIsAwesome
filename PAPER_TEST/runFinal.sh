cd Boltz-f=1
qsub -l h_rt=100:00:00 -l longq=1 trainingBQ1.sh
cd ../Boltz-f=1-2
qsub -l h_rt=100:00:00 -l longq=1 trainingBQ1-2.sh
cd ../Boltz-f=1-3
qsub -l h_rt=100:00:00 -l longq=1 trainingBQ1-3.sh
cd ../Epsilon
qsub -l h_rt=100:00:00 -l longq=1 trainingE.sh

