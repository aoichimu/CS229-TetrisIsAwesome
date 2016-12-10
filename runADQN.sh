qsub -v exp='exp-Breakout-14',threads='14' -l h_rt=100:00:00 -l longq=1 trainingADQN.sh
qsub -v exp='exp-Breakout-ram-14',threads='14',game='Breakout-ram-v0' -l h_rt=100:00:00 -l longq=1 trainingRAM.sh
