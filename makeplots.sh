echo "Starting to make plots"
for i in $(seq 0 1 29)
do
	echo $i
	file="results/conso-classif-deep/run_16/group_$i"
	python performance-tracking/experiments/conso_classif_deep/read_event.py --storage_path $file
done