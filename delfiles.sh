echo "Starting to clear stderr files"
for i in $(seq 0 1 29)
do
	echo $i
	file="results/conso-classif-deep/run_5/group_$i"
    > $file/stderr.txt
done
