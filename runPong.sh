qsub -v env='Pong-ram-v0',init='he_uniform',opt='rms',output='12-7-Pong-ram1' training.sh
qsub -v env='Pong-ram-v0',init='he_normal',opt='rms',output='12-7-Pong-ram2' training.sh
qsub -v env='Pong-ram-v0',init='glorot_uniform',opt='rms',output='12-7-Pong-ram3' training.sh
qsub -v env='Pong-ram-v0',init='glorot_uniform',opt='adam',output='12-7-Pong-ram4' training.sh

qsub -v env='Pong-v0',init='he_uniform',opt='rms',output='12-7-Pong-1' trainingADQN.sh
qsub -v env='Pong-v0',init='he_normal',opt='rms',output='12-7-Pong-2' trainingADQN.sh
qsub -v env='Pong-v0',init='glorot_uniform',opt='rms',output='12-7-Pong-3' trainingADQN.sh
qsub -v env='Pong-v0',init='glorot_uniform',opt='adam',output='12-7-Pong-4' trainingADQN.sh
