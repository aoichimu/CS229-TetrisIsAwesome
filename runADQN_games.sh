
#batch_games=("SpaceInvaders-ram-v0" "Robotank-ram-v0" "Krull-ram-v0" "Tennis-ram-v0" "BeamRider-ram-v0" "DoubleDunk-ram-v0" "KungFuMaster-ram-v0")
batch_games=("Centipede-ram-v0" "MsPacman-ram-v0" "Asterix-ram-v0" "Boxing-ram-v0" "Atlantis-ram-v0" "VideoPinball-ram-v0") 

for game in ${batch_games[*]}
do
	qsub -v exp='12-9-'${game},threads='14',game=${game} -l h_rt=100:00:00 -l longq=1 trainingRAM.sh			
done




