#batch_games=("Centipede-ram-v0")
batch_games=("Centipede-ram-v0" "MsPacman-ram-v0" "Asterix-ram-v0" "Boxing-ram-v0" "Atlantis-ram-v0" "VideoPinball-ram-v0" "Breakout-ram-v0") # "Robotank-ram-v0" "Krull-ram-v0" "Tennis-ram-v0" "BeamRider-ram-v0" "DoubleDunk-ram-v0" "KungFuMaster-ram-v0")

for game in ${batch_games[*]}
do
	qsub -v env=${game},net='2',opt='adam',output='12-8-'${game}'-1' -l h_rt=100:00:00 -l longq=1 training.sh
	qsub -v env=${game},net='4',opt='adam',output='12-8-'${game}'-2' -l h_rt=100:00:00 -l longq=1 training.sh
done
