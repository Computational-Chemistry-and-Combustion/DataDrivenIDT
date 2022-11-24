#!/bin/bash
##Source the file first or run by : source ./Run_commands.sh
export IDCODE="${HOME}/Data_driven_Kinetics/"
export PATH=$PATH:$IDCODE
alias IDprediction="pwd>~/Data_driven_Kinetics/filelocation.txt && Run.sh"

for i in {1..1}
do
	echo "******************"
	echo "Running Cycle-$i"
	echo "******************"
	IDprediction --algo GMM --train trainset.csv --n 3
	IDprediction --algo GMM --test testset.csv
done