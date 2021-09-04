#!/bin/bash
##Source the file first or run by : source ./Run_commands.sh
for i in {1..5000}
do
	echo "******************"
	echo "Running Cycle-$i"
	echo "******************"
	python generate_train_test.py
	rm -r ./analysis/Analysis_clusters_2/object_file
	python Flag.py
done


