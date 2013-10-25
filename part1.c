#include <emmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    
    /*matrix padding - make a new matrix of data_size+2 * data_size+2
    fill wherever x=0, y=0, x=data_size, y=data_size with 0s; 
    from address 0 to data_size+1, fill with 0s
    first elt = 0, copy contents of in[row2] , then 0 
    repeat until in[row data_size-1]
    fill last row [data_size+2-1] with 0s

    for(i = 0; i + blocksize < n; i+=blocksize) 
        for(j = 0; j + blocksize < n; j+=blocksize) 
            for(a=i; a < blocksize + i; a++)
                for(b=j; b < blocksize + j; b++)
                    dst[a + b*n] = src[b + a*n];
    //in cases where n/blocksize isn't an integer
    rem = i;
    for(i = 0; i < n; i++)
        for(j = rem; j < n; j++)
            dst[i + j*n] = src[j + i*n];
    for(i = rem; i < n; i++)
        for(j = 0; j< n; j++)
            dst[i + j*n] = src[j + i*n];

    */
    int new_X = data_size_X + ((KERNX-1)/2)*2;
    int new_Y = data_size_Y + ((KERNY-1)/2)*2;
    int blocksize = 4;
    float padded[new_X * new_Y]; //declaring new array
    int i, j, y, x, rem;
    //printf("new dim: %d %d \n", new_X, new_Y);
    for(x = 0; x < new_X; x++) {
		for (y = 0; y < new_Y; y++) {
			//if ((x < (KERNX-1)/2) || (y < (KERNY-1)/2) || (x > (new_X-1-(KERNX-1)/2)) || (y > (new_Y-1-(KERNY-1)/2)) { //if it's one of the four sides
			if (x == 0 || y == 0 || x == new_X-1 || y == new_Y-1) {
				padded[x*new_Y + y] = 0;
			}
			else {
				padded[x*new_Y + y] = in[(x-1) + (y-1)*data_size_X];
			}
		}
	}
	/*for (int a = 0; a < 260; a++) {
		printf("padded & in %d %f %f \n", a, padded[a+127], in[a]);
	}/*
	rem = x;
	for (x = 0; y < new_X; x++) {
		for (y = rem; x < new_Y; y++) {
			if (x == 0 || y == 0 || x == new_X-1 || y == new_Y-1) { //if it's one of the four sides
				padded[x*new_Y + y] = 0;
			}
			else {
				padded[x*new_Y + y] = in[(x-1)*data_size_Y + (y-1)];
			}
		}
	}
	for (x = rem; y < new_X; x++) {
		for (y = 0; x < new_Y; y++) {
			if (x == 0 || y == 0 || x == new_X-1 || y == new_Y-1)//if it's one of the four sides
				padded[x*new_Y + y] = 0;
			else
				padded[x*new_Y + y] = in[(x-1)*data_size_Y + (y-1)];
		}
	}*/

    // main convolution loop
	//printf("entering kernel loop");

	for(int y = 1; y < new_Y-1; y++) { // the y coordinate of theoutput location we're focusing on
		for(int x = 1; x < new_X-1; x++){ // the x coordinate of the output location we're focusing on
			//*remember to flip the kernel coordinates
			//for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){// kernel unflipped y coordinate'
			//	for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
				//if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
					//Note that the kernel needs to be flipped
			//kern_cent_x = 1, kern_cent_y = 1
			/*out[(x-1)+(y-1)*data_size_X] += //bot right
									kernel[(kern_cent_X-1)+(kern_cent_Y-1)*KERNX] * padded[(x+1) + (y+1)*new_X];
			out[(x-1)+(y-1)*data_size_X] += //bot mid
									kernel[(kern_cent_X)+(kern_cent_Y-1)*KERNX] * padded[(x) + (y+1)*new_X];
			out[(x-1)+(y-1)*data_size_X] += //bot left
									kernel[(kern_cent_X+1)+(kern_cent_Y-1)*KERNX] * padded[(x-1) + (y+1)*new_X];
			out[(x-1)+(y-1)*data_size_X] += //right
									kernel[(kern_cent_X-1)+(kern_cent_Y)*KERNX] * padded[(x+1) + (y)*new_X];
			out[(x-1)+(y-1)*data_size_X] += //center
									kernel[(kern_cent_X)+(kern_cent_Y)*KERNX] * padded[(x) + (y)*new_X];
			out[(x-1)+(y-1)*data_size_X] += //left
									kernel[(kern_cent_X+1)+(kern_cent_Y)*KERNX] * padded[(x-1) + (y)*new_X];
			out[(x-1)+(y-1)*data_size_X] += //top right
									kernel[(kern_cent_X-1)+(kern_cent_Y+1)*KERNX] * padded[(x+1) + (y-1)*new_X];
			out[(x-1)+(y-1)*data_size_X] += //top mid
									kernel[(kern_cent_X)+(kern_cent_Y+1)*KERNX] * padded[(x) + (y-1)*new_X];
			out[(x-1)+(y-1)*data_size_X] += //top left
									kernel[(kern_cent_X+1)+(kern_cent_Y+1)*KERNX] * padded[(x-1) + (y-1)*new_X];
									
			printf("FINAL: %d, %d %f \n", x-1, y-1, out[(x-1)+(y-1)*data_size_X]);
			
			if (x == 1 && y == 1) {
				printf("0 in %f padded %f \n", in[(x)*data_size_Y +(y)], padded[(x+1)*new_Y + (y+1)]);
				printf("1 in %f padded %f \n", in[(x-1)*data_size_Y +(y)], padded[(x)*new_Y + (y+1)]);
				printf("3 in %f padded %f \n", in[(x)*data_size_Y +(y-1)], padded[(x+1)*new_Y + (y)]);
				printf("4 in %f padded %f \n", in[(x-1)*data_size_Y +(y-1)], padded[(x)*new_Y + (y)]);
			}
			int out_ind = (x-1) +(y-1)*data_size_X;
			out[out_ind] += //bot right 0
							kernel[(kern_cent_X-1)+(kern_cent_Y-1)*KERNX] * padded[(x+1)*new_Y + (y+1)];
			out[out_ind] += //bot mid 1
							kernel[(kern_cent_X)+(kern_cent_Y-1)*KERNX] * padded[(x)*new_Y + (y+1)];
			out[out_ind] += //bot left 2
							kernel[(kern_cent_X+1)+(kern_cent_Y-1)*KERNX] * padded[(x-1)*new_Y + (y+1)];
			out[out_ind] += //right 3
							kernel[(kern_cent_X-1)+(kern_cent_Y)*KERNX] * padded[(x+1)*new_Y + (y)];
			out[out_ind] += //center 4
							kernel[(kern_cent_X)+(kern_cent_Y)*KERNX] * padded[(x)*new_Y + (y)];
			out[out_ind] += //left 5
							kernel[(kern_cent_X+1)+(kern_cent_Y)*KERNX] * padded[(x-1)*new_Y + (y)];
			out[out_ind] += //top right 6
							kernel[(kern_cent_X-1)+(kern_cent_Y+1)*KERNX] * padded[(x+1)*new_Y + (y-1)];
			out[out_ind] += //top mid 7
							kernel[(kern_cent_X)+(kern_cent_Y+1)*KERNX] * padded[(x)*new_Y + (y-1)];
			out[out_ind] += //top left 8
							kernel[(kern_cent_X+1)+(kern_cent_Y+1)*KERNX] * padded[(x-1)*new_Y + (y-1)];
			*/
			out[(x-1) +(y-1)*data_size_X] = //bot right 0
							  kernel[0] * padded[(x+1)*new_Y + (y+1)]
							+ kernel[1] * padded[(x)*new_Y + (y+1)]
							+ kernel[2] * padded[(x-1)*new_Y + (y+1)]
							+ kernel[3] * padded[(x+1)*new_Y + (y)]
							+ kernel[4] * padded[(x)*new_Y + (y)]
							+ kernel[5] * padded[(x-1)*new_Y + (y)]
							+ kernel[6] * padded[(x+1)*new_Y + (y-1)]
							+ kernel[7] * padded[(x)*new_Y + (y-1)]
							+ kernel[8] * padded[(x-1)*new_Y + (y-1)];
			//printf("FINAL: %d, %d %f \n", x-1, y-1, out[(x-1)*data_size_Y +(y-1)]);*/
		}
	}
	return 1;
}
