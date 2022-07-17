#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <stdlib.h>
#include <assert.h>
#include "png_util.h"
//#define MAX_N 1000

char *plate[2];
int which = 0;
int n;

int live(int index){
    return (plate[which][index - n - 3] 
        + plate[which][index - n - 2]
        + plate[which][index - n - 1]
        + plate[which][index - 1]
        + plate[which][index + 1]
        + plate[which][index + n + 1]
        + plate[which][index + n + 2]
        + plate[which][index + n + 3]);
}
void iteration(){
#pragma omp parallel for schedule(static)
    for(int i = 1; i <= n; i++){
        for(int j = 1; j <= n; j++){
            int index = i * (n + 2) + j;
            int num = live(index);
            if(plate[which][index]){
                plate[!which][index] =  (num == 2 || num == 3) ?
                    1 : 0;
            }else{
                plate[!which][index] = (num == 3);
            }
        }
    }
    which = !which;
}
void print_plate(){
    if (n < 60) {
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= n; j++){
                printf("%d", plate[which][i * (n + 2) + j]);
            }
            printf("\n");
        }
    } else {
	printf("Plate too large to print to screen\n");
    }
    printf("\0");
}

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
    write_png_file((char *) filename,img,sz);
   
    printf("done writing png\n"); 
    free(img);
    printf("done freeing memory\n");
    
}

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

	//Allocate memory for plates
        int arrlen = (n+2)*(n+2);
        int nBytes = sizeof(char)*arrlen;
 	char *  memblock= (char *) malloc(nBytes*2);
        char *  d_memblock;
        plate[0] = (char *) &memblock[0];
        plate[1] = (char *) &memblock[arrlen];
        for(int k=0;k < 2*arrlen;k++)
                memblock[k] = 0;
       
        char line[n];
        if (!random) {
            for(int i = 1; i <= n; i++){
                scanf("%s", &line);
                for(int j = 0; j < n; j++){
                    plate[0][i * (n + 2) + j + 1] = line[j] - '0';
                }
            }
	} else {
	   for(int i = 1; i <= n; i++) 
               for(int j = 0; j < n; j++) 
                   plate[0][i * (n+2) +j + 1] = rand() % 2;
	}

        for(int i = 0; i < M; i++){
	    printf("\nIteration %d:\n",i);
            iteration();
        }
        printf("\n\nFinal:\n");
	//plate2png("plate.png");
	print_plate();
    }
    return 0;
}
