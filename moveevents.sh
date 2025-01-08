echo "Starting to move events on HDD"
for i in $(seq 0 1 29)
do
	echo $i
	file="results/conso-classif-deep/run_4/group_$i"
    mkdir -p /media/HDD/results_conso/deep-bigen/run_4/group_$i/ShortCNN_RGB
	mv $file/ShortCNN_RGB/* /media/HDD/results_conso/deep-bigen/run_4/group_$i/ShortCNN_RGB/
done
