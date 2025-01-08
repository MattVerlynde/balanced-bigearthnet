echo "Starting to save plots"
for i in $(seq 12 1 13)
do
	echo $i
	file="results/conso-classif-deep/run_4/group_$i/plots"
	mkdir -p /home/verlyndem/Documents/cahier-labo-these/static/results_deep/g$i
	cp $file/accuracies_all/accuracies.html /home/verlyndem/Documents/cahier-labo-these/static/results_deep/g$i/accuracies.html
	cp $file/losses_all/losses.html /home/verlyndem/Documents/cahier-labo-these/static/results_deep/g$i/losses.html
done
