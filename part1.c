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
    
    int new_X = data_size_X+2;
    int new_Y = data_size_Y+2;
    float padded[new_X*new_Y];
	int curr=0, in_count=0;
	for (; curr < new_X; curr++)
		padded[curr] = 0;
	for (int i = 0; i < data_size_Y; i++) { //the amount of cols of input
		padded[curr] = 0;
		curr++;
		for (int j = 0; j < data_size_X; j++, curr++, in_count++) { //can do SSE because it's in multiples of 4
			padded[curr] = in[in_count];
		}
		padded[curr] = 0;
		curr++;
	}
	int rem = curr;
	for (; curr < new_X + rem; curr++)
		padded[curr] = 0;
    // main convolution loop
	//printf("entering kernel loop");

	for(int y = 1; y < new_Y-1; y++) { // the y coordinate of theoutput location we're focusing on
		for(int x = 1; x < new_X-1; x++){ // the x coordinate of the output location we're focusing on
			//*remember to flip the kernel coordinates
			out[(x-1) + (y-1)*data_size_X] =
							  kernel[8] * padded[(x-1) + (y-1)*new_X]
							+ kernel[5] * padded[(x-1) + (y)*new_X]	
							+ kernel[2] * padded[(x-1) + (y+1)*new_X]
							+ kernel[7] * padded[(x) + (y-1)*new_X]
							+ kernel[4] * padded[(x) + (y)*new_X]
							+ kernel[1] * padded[(x) + (y+1)*new_X]
							+ kernel[6] * padded[(x+1) + (y-1)*new_X]
							+ kernel[3] * padded[(x+1) + (y)*new_X]
							+ kernel[0] * padded[(x+1) + (y+1)*new_X];
			//printf("FINAL: %d, %d %f \n", x-1, y-1, out[(x-1)*data_size_Y +(y-1)]);*/
		}
	}
	return 1;
}
