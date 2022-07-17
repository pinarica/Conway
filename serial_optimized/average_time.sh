#!/bin/bash --login

module load GCCcore/9.3.0
module load libpng/1.6.37

make clean
make

mkdir -p serial_times

#echo "0 50 100" | ./gol

for n in {1..10};
do
    ((size = n*25));
    echo $n*25 | bc;

    for i in {1..25};
    do

    { time echo "0 $size 500" | ./gol; } 2>> time_{$size}.txt
 
    done
    mv time_{$size}.txt serial_times
done


