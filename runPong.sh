qsub -v env='Pong-ram-v0',init='he_uniform',opt='rms',output='12-7-Pong-ram1' -l h_rt=168:00:00 -l longq=1 training.sh
qsub -v env='Pong-ram-v0',init='he_normal',opt='rms',output='12-7-Pong-ram2' -l h_rt=168:00:00 -l longq=1 training.sh
qsub -v env='Pong-ram-v0',init='glorot_uniform',opt='rms',output='12-7-Pong-ram3' -l h_rt=168:00:00 -l longq=1 training.sh
qsub -v env='Pong-ram-v0',init='glorot_uniform',opt='adam',output='12-7-Pong-ram4' -l h_rt=168:00:00 -l longq=1 training.sh

#qsub -v env='Pong-v0',init='he_uniform',opt='rms',threads='24',output='12-7-Pong-1' -l h_rt=168:00:00 -l longq=1 trainingADQN.sh
#qsub -v env='Pong-v0',init='he_normal',opt='rms',threads='24',output='12-7-Pong-2' -l h_rt=168:00:00 -l longq=1 trainingADQN.sh
#qsub -v env='Pong-v0',init='glorot_uniform',opt='rms',threads='24',output='12-7-Pong-3' -l h_rt=168:00:00 -l longq=1 trainingADQN.sh
#qsub -v env='Pong-v0',init='glorot_uniform',opt='adam',threads='24',output='12-7-Pong-4' -l h_rt=168:00:00 -l longq=1 trainingADQN.sh
