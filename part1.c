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
    
    int new_X = data_size_X + ((KERNX-1)/2)*2;
    int new_Y = data_size_Y + ((KERNY-1)/2)*2;
    int blocksize = 4;
    float padded[new_X * new_Y]; //declaring new array
    int i, j, y, x, rem;
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

    // main convolution loop
	//printf("entering kernel loop");

	for(int y = 1; y < new_Y-1; y++) { // the y coordinate of theoutput location we're focusing on
		for(int x = 1; x < new_X-1; x++){ // the x coordinate of the output location we're focusing on
			//*remember to flip the kernel coordinates
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
