#include "cuda.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

int n;

__global__ void iterateLive(int n, int *first_plate, int *second_plate){
// We want id ∈ [1,dim]
    int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int id = iy * (n+2) + ix;
 
    int numNeighbors;
 
    if (iy <= n && ix <= n) {
 
        // Get the number of neighbors for a given grid point
        numNeighbors = first_plate[id+(n+2)] + first_plate[id-(n+2)] //upper lower
                     + first_plate[id+1] + first_plate[id-1]         //right left
                     + first_plate[id+(n+3)] + first_plate[id-(n+3)] //diagonals
                     + first_plate[id-(n+1)] + first_plate[id+(n+1)];
 
        int cell = first_plate[id];
        // Here we have explicitly all of the game rules
        if (cell == 1 && numNeighbors < 2)
            second_plate[id] = 0;
        else if (cell == 1 && (numNeighbors == 2 || numNeighbors == 3))
            second_plate[id] = 1;
        else if (cell == 1 && numNeighbors > 3)
            second_plate[id] = 0;
        else if (cell == 0 && numNeighbors == 3)
            second_plate[id] = 1;
        else
            second_plate[id] = cell;
    }  
}

void print_plate(int *plate){
    if (n < 60) {
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= n; j++){
                printf("%d", plate[i * (n + 2) + j]);
            }
            printf("\n");
        }
    } else {
	printf("Plate too large to print to screen\n");
    }
    printf("\0");
}
/*
void plate2png(const char* filename) {
    char * img = (char *) malloc(n*n*sizeof(char));

    image_size_t sz;
    sz.width = n;
    sz.height = n; 

    for(int i = 1; i <= n; i++){
        for(int j = 1; j <= n; j++){
            int pindex = i * (n + 2) + j;
            int index = (i-1) * (n) + j;
            if (plate[!which][pindex] > 0)
		img[index] = 255; 
            else 
		img[index] = 0;
        }
    }
    printf("Writing file\n");
    write_png_file((unsigned char *) filename,img,sz);
   
    printf("done writing png\n"); 
    free(img);
    printf("done freeing memory\n");
    
}
*/

int main() { 
    int M;
    int S;
    // changed to compare computation time for different sized boards
    if(scanf("%d %d %d", &n, &S, &M) == 3){
        int random=0;
	if (n == 0) { 
	   n = S;
	   random=1;
        }
	
	// Allocate memory for plates
        int array_length = ((n+2)*(n+2));
	size_t nBytes = sizeof(int)*array_length;

	int *first_plate, *second_plate, *tmp_plate;
	cudaMallocManaged(&first_plate, nBytes);
	cudaMallocManaged(&second_plate, nBytes);
	cudaMallocManaged(&tmp_plate, nBytes);
	// Initialize Plates       
        char line[n];
        if (!random) {
            for(int i = 1; i <= n; i++){
                scanf("%s", &line);
                for(int j = 0; j < n; j++){
                    //plate[0][i * (n + 2) + j + 1] = line[j] - '0';
                }
            }
	} else {
	   for(int i = 1; i <= n; i++) 
               for(int j = 0; j < n; j++) 
                   first_plate[i * (n+2) +j + 1] = rand() % 2;
	}

	// Set up Block and Grid
	int threads = 32; // 32X32 Maximum threads per Block
	int blocks = (n+threads - 1)/threads;
	dim3 THREADS(threads, threads);
	dim3 BLOCKS(blocks, blocks);
	
        for(int i = 0; i < M; i++){
	    iterateLive<<<BLOCKS, THREADS>>>(n, first_plate, second_plate);
	    cudaDeviceSynchronize();
            
            tmp_plate = first_plate;
            first_plate = second_plate;
            second_plate = tmp_plate;
	    
	    printf("\nIteration %d:\n",i);
	    //print_plate(first_plate);
        }
	//plate2png
        printf("\n\nFinal:\n");
	print_plate(first_plate);
    }
    return 0;
}
