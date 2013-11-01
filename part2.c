#include <emmintrin.h>
#include <nmmintrin.h>
#include <omp.h>

#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    	
    // main convolution loop
    int x = 0, y = 0;
    //top left corner //x = 0, y = 0
    out[(x) + (y)*data_size_X] =
    							kernel[1] * in[(x) + (y+1)*data_size_X]
		                        + kernel[4] * in[(x) + (y)*data_size_X]
		                        + kernel[3] * in[(x+1) + (y)*data_size_X]
		                        + kernel[0] * in[(x+1) + (y+1)*data_size_X];
	//bottom left corner
    x = data_size_X-1; //y = 0
	out[(x) + (y)*data_size_X] =
								kernel[2] * in[(x-1) + (y+1)*data_size_X]
		                        + kernel[5] * in[(x-1) + (y)*data_size_X]
		                        + kernel[4] * in[(x) + (y)*data_size_X]
		                        + kernel[1] * in[(x) + (y+1)*data_size_X];
    //top right corner
	y = data_size_Y - 1; x = 0;
	out[(x) + (y)*data_size_X] =
		                        kernel[4] * in[(x) + (y)*data_size_X]   
		                        + kernel[7] * in[(x) + (y-1)*data_size_X]     
		                        + kernel[6] * in[(x+1) + (y-1)*data_size_X]
		                        + kernel[3] * in[(x+1) + (y)*data_size_X];
	//bottom right corner
    x = data_size_X - 1; //y = data_size_Y-1
    out[(x) + (y)*data_size_X] =
		                        kernel[5] * in[(x-1) + (y)*data_size_X]
		                        + kernel[8] * in[(x-1) + (y-1)*data_size_X]
		                        + kernel[7] * in[(x) + (y-1)*data_size_X]        
		                        + kernel[4] * in[(x) + (y)*data_size_X];
    

    //left column: x = 1 to data_size_X-2, y = 0
    y = 0;
    #pragma omp parallel for
	for (x = 1; x < data_size_X-1; x++) {
		out[(x) + (y)*data_size_X] =
		                        kernel[5] * in[(x-1) + (y)*data_size_X]
		                        + kernel[2] * in[(x-1) + (y+1)*data_size_X]
		                        + kernel[1] * in[(x) + (y+1)*data_size_X]
		                        + kernel[4] * in[(x) + (y)*data_size_X]
		                        + kernel[3] * in[(x+1) + (y)*data_size_X]
		                        + kernel[0] * in[(x+1) + (y+1)*data_size_X];
	}
	//top row
	x = 0;
	#pragma omp parallel for
	for (y = 1; y < data_size_Y-1; y++) {
		out[x + (y)*data_size_X] = 
		                        kernel[7] * in[(x) + (y-1)*data_size_X]        
		                        + kernel[4] * in[(x) + (y)*data_size_X]
		                        + kernel[1] * in[(x) + (y+1)*data_size_X]
		                        + kernel[0] * in[(x+1) + (y+1)*data_size_X]
		                        + kernel[3] * in[(x+1) + (y)*data_size_X]
		                        + kernel[6] * in[(x+1) + (y-1)*data_size_X];
	}
	//bottom row
    x = data_size_X-1;
    #pragma omp parallel for
    for (y = 1; y < data_size_Y-1; y++) {
		out[x + (y)*data_size_X] = 
								kernel[4] * in[(x) + (y)*data_size_X]
		                        + kernel[7] * in[(x) + (y-1)*data_size_X]   
		                        + kernel[1] * in[(x) + (y+1)*data_size_X]
		                        + kernel[2] * in[(x-1) + (y+1)*data_size_X]    
		                        + kernel[5] * in[(x-1) + (y)*data_size_X]
		                        + kernel[8] * in[(x-1) + (y-1)*data_size_X];                        
	}
    //right column
    y = data_size_Y-1;
    #pragma omp parallel for
    for (x = 1; x < data_size_X-1; x++) {
		out[x + (y)*data_size_X] =
		                          kernel[8] * in[(x-1) + (y-1)*data_size_X]
		                        + kernel[5] * in[(x-1) + (y)*data_size_X]
		                        + kernel[4] * in[(x) + (y)*data_size_X]
		                        + kernel[7] * in[(x) + (y-1)*data_size_X]        
		                        + kernel[6] * in[(x+1) + (y-1)*data_size_X]
		                        + kernel[3] * in[(x+1) + (y)*data_size_X];
	}

	//for the inner parts of the image
	__m128 k8 = _mm_set_ps1(kernel[8]);
	__m128 k5 = _mm_set_ps1(kernel[5]);
	__m128 k2 = _mm_set_ps1(kernel[2]);
	__m128 k7 = _mm_set_ps1(kernel[7]);
	__m128 k4 = _mm_set_ps1(kernel[4]);
	__m128 k1 = _mm_set_ps1(kernel[1]);
	__m128 k6 = _mm_set_ps1(kernel[6]);
	__m128 k3 = _mm_set_ps1(kernel[3]);
	__m128 k0 = _mm_set_ps1(kernel[0]);
	#pragma omp parallel for
	for(int y = 1; y < data_size_Y-1; y++) { // the y coordinate of theoutput location we're focusing on
		int x;
		for(x = 1; x < data_size_X-4-1; x+=4) { // the x coordinate of the output location we're focusing on
		//*remember to flip the kernel coordinates
		//gets the next 9 pixels in vector form
		__m128 l8 = _mm_loadu_ps((float*)(in + (x-1) + (y-1)*data_size_X)); 
		__m128 l5 = _mm_loadu_ps((float*)(in + (x-1) + (y)*data_size_X)); 
		__m128 l2 = _mm_loadu_ps((float*)(in + (x-1) + (y+1)*data_size_X));
		__m128 l1 = _mm_loadu_ps((float*)(in + (x) + (y+1)*data_size_X)); 
		__m128 l4 = _mm_loadu_ps((float*)(in + (x) + (y)*data_size_X));
		__m128 l7 = _mm_loadu_ps((float*)(in + (x) + (y-1)*data_size_X)); 
		__m128 l6 = _mm_loadu_ps((float*)(in + (x+1) + (y-1)*data_size_X)); 
		__m128 l3 = _mm_loadu_ps((float*)(in + (x+1) + (y)*data_size_X)); 
		__m128 l0 = _mm_loadu_ps((float*)(in + (x+1) + (y+1)*data_size_X)); 

		//the 9 kernel*pixel vectors
		__m128 n8 = _mm_mul_ps(k8, l8); //multiplying the corresponding kernel to the pixel vectors
		__m128 n5 = _mm_mul_ps(k5, l5);
		__m128 n2 = _mm_mul_ps(k2, l2);
		__m128 n7 = _mm_mul_ps(k7, l7);
		__m128 n4 = _mm_mul_ps(k4, l4);
		__m128 n1 = _mm_mul_ps(k1, l1);
		__m128 n6 = _mm_mul_ps(k6, l6);
		__m128 n3 = _mm_mul_ps(k3, l3);
		__m128 n0 = _mm_mul_ps(k0, l0); 

		//summing up the column sums
		__m128 c1 = _mm_add_ps(n8, n5); 
		c1 = _mm_add_ps(c1, n2); //c1 = column1 sums
		__m128 c2 = _mm_add_ps(n7, n4);
		c2 = _mm_add_ps(c2, n1); //c2 = column2 sums
		__m128 c3 = _mm_add_ps(n6, n3); 
		c3 = _mm_add_ps(c3, n0); //c3 = column3 sums
		__m128 r_vec = _mm_add_ps(c1, c2); // r_vec is the sum of the column sums
		r_vec = _mm_add_ps(r_vec, c3); 
		_mm_storeu_ps((out+(x)+(y)*data_size_X), r_vec); //stores result vector into x, x+1, x+2, x+3
		}
		//the remaining pixels outside the multiple of 4
		//#pragma omp parallel for
		for (; x < data_size_X - 1; x++) {
			out[(x) + (y)*data_size_X] =
				                          kernel[8] * in[(x-1) + (y-1)*data_size_X]
				                        + kernel[5] * in[(x-1) + (y)*data_size_X]        
				                        + kernel[2] * in[(x-1) + (y+1)*data_size_X]
				                        + kernel[1] * in[(x) + (y+1)*data_size_X]
				                        + kernel[4] * in[(x) + (y)*data_size_X]
				                        + kernel[7] * in[(x) + (y-1)*data_size_X]
				                        + kernel[6] * in[(x+1) + (y-1)*data_size_X]
				                        + kernel[3] * in[(x+1) + (y)*data_size_X]
				                        + kernel[0] * in[(x+1) + (y+1)*data_size_X];
		}
	}
	return 1;
}
