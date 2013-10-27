#include <emmintrin.h>
#include <nmmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    
    /* matrix padding - make a new matrix of data_size+2 * data_size+2
    fill first & last columns with 0s
    for each column until data_size_Y: first elt = 0, copy contents of from in[] , then last elt = 0 
	*/
    	
	int new_X = data_size_X + 2;
    int new_Y = data_size_Y + 2;
    float padded[new_X * new_Y];
    int curr = 0, in_count = 0;
    for (; curr < new_X; curr++) //filling first col with 0s
        padded[curr] = 0;
    for (int i = 0; i < data_size_Y; i++) { //the amount of cols of input
        padded[curr] = 0;
        curr++;
        for (int j = 0; j < data_size_X; j++, curr++, in_count++) {
            padded[curr] = in[in_count];
        }
        padded[curr] = 0;
        curr++;
    }
    int rem = curr;
    for (; curr < new_X + rem; curr++) //filling the last col with 0s
        padded[curr] = 0;

    // main convolution loop

	__m128 k8 = _mm_set_ps1(kernel[8]);
	__m128 k7 = _mm_set_ps1(kernel[7]);
	__m128 k6 = _mm_set_ps1(kernel[6]);
	__m128 k5 = _mm_set_ps1(kernel[5]);
	__m128 k4 = _mm_set_ps1(kernel[4]);
	__m128 k3 = _mm_set_ps1(kernel[3]);
	__m128 k2 = _mm_set_ps1(kernel[2]);
	__m128 k1 = _mm_set_ps1(kernel[1]);
	__m128 k0 = _mm_set_ps1(kernel[0]);

	for(int y = 1; y < new_Y-1; y++) { // the y coordinate of theoutput location we're focusing on
		for(int x = 1; x < new_X-4; x+=4) { // the x coordinate of the output location we're focusing on
		//*remember to flip the kernel coordinates
		
		//gets the next 8 pixels in vector form
		__m128 l8 = _mm_loadu_ps((float*)(padded + (x-1) + (y-1)*new_X)); 
		__m128 l7 = _mm_loadu_ps((float*)(padded + (x) + (y-1)*new_X)); 
		__m128 l6 = _mm_loadu_ps((float*)(padded + (x+1) + (y-1)*new_X)); 
		__m128 l5 = _mm_loadu_ps((float*)(padded + (x-1) + (y)*new_X)); 
		__m128 l4 = _mm_loadu_ps((float*)(padded + (x) + (y)*new_X)); 
		__m128 l3 = _mm_loadu_ps((float*)(padded + (x+1) + (y)*new_X)); 
		__m128 l2 = _mm_loadu_ps((float*)(padded + (x-1) + (y+1)*new_X)); 
		__m128 l1 = _mm_loadu_ps((float*)(padded + (x) + (y+1)*new_X)); 
		__m128 l0 = _mm_loadu_ps((float*)(padded + (x+1) + (y+1)*new_X)); 

		//the 9 kernel*pixel vectors
		__m128 n0 = _mm_mul_ps(k0, l0); //multiplying the corresponding kernel to the pixel vectors
		__m128 n1 = _mm_mul_ps(k1, l1);
		__m128 n2 = _mm_mul_ps(k2, l2);
		__m128 n3 = _mm_mul_ps(k3, l3);
		__m128 n4 = _mm_mul_ps(k4, l4);
		__m128 n5 = _mm_mul_ps(k5, l5);
		__m128 n6 = _mm_mul_ps(k6, l6);
		__m128 n7 = _mm_mul_ps(k7, l7);
		__m128 n8 = _mm_mul_ps(k8, l8);

		//summing up the column sums
		__m128 c1 = _mm_add_ps(n8, n5); 
		c1 = _mm_add_ps(c1, n2); //c1 = column1 sums
		__m128 c2 = _mm_add_ps(n7, n4);
		c2 = _mm_add_ps(c2, n1); //c2 = column2 sums
		__m128 c3 = _mm_add_ps(n6, n3); 
		c3 = _mm_add_ps(c3, n0); //c3 = column3 sums
		__m128 r_vec = _mm_add_ps(c1, c2); // r_vec is the sum of the column sums
		r_vec = _mm_add_ps(r_vec, c3); 
		_mm_storeu_ps((out+(x-1)+(y-1)*data_size_X), r_vec); //stores result vector into x, x+1, x+2, x+3
		}
	}
	return 1;
}
