#!/bin/bash --login

module load GCCcore/9.3.0
module load libpng/1.6.37

make clean
make

time echo "0 5000 10" | ./gol


