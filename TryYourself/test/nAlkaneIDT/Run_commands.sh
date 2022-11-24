#!/bin/bash
##Source the file first or run by : source ./Run_commands.sh
export IDCODE="~/Data_driven_Kinetics/"
export PATH=$PATH:$IDCODE
alias IDprediction="pwd>~/Data_driven_Kinetics/filelocation.txt && Run.sh"

for i in {1..1}
do
	echo "******************"
	echo "Running Cycle-$i"
	echo "******************"
	IDprediction -c 0.1 -t trainset.csv
	IDprediction -e testset.csv
done