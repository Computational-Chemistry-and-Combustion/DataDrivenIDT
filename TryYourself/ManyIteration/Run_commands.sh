#!/bin/bash
##Source the file first or run by : source ./Run_commands.sh
export IDCODE="${HOME}/Data_driven_Kinetics/"
export PATH=$PATH:$IDCODE
alias IDprediction="pwd>~/Data_driven_Kinetics/filelocation.txt && Run.sh"

for i in {1..5}
do
	echo "******************"
	echo "Running Cycle-$i"
	echo "******************"
	python generate_train_test.py
	IDprediction --algo tree --train train.csv --c 0.1
	IDprediction --algo tree --test test.csv
	IDprediction --k FixedTest.csv
done

IDprediction --f FixedTest.csv
