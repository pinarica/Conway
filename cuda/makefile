all: gol 

CC:=nvcc
EXT:=cu
FLAGS:=-std=c++11 -o

gol: gameoflife.$(EXT)
	$(CC) $(FLAGS) gol gameoflife.$(EXT) 

test: gol data.txt
	./gol < data.txt

clean:
	rm *.o
	rm gol
	rm *.txt
	rm slurm*
