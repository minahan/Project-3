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
    
    /* matrix padding - make a new matrix of data_size+2 * data_size+2
    fill wherever x=0, y=0, x=data_size, y=data_size with 0s; 
    from address 0 to data_size+1, fill with 0s
    first elt = 0, copy contents of in[row2] , then 0 
    repeat until in[row data_size-1]
    fill last row [data_size+2-1] with 0s

    int new_X = data_size_X + ((KERNX-1)/2)*2;
    int new_Y = data_size_Y + ((KERNY-1)/2)*2;
    int blocksize = 10;
    float padded[new_X * new_Y]; //declaring new array
    int i, j, y, x, rem;
    //printf("new dim: %d %d \n", new_X, new_Y);
    for(x = 0; x  < new_X; x++) {
		for (y = 0; y < new_Y; y++) {
			//if ((x < (KERNX-1)/2) || (y < (KERNY-1)/2) || (x > (new_X-1-(KERNX-1)/2)) || (y > (new_Y-1-(KERNY-1)/2)) { //if it's one of the four sides
			if (x == 0 || y == 0 || x == new_X-1 || y == new_Y-1)
				padded[x*new_Y + y] = 0;
			else
				//_mm_storeu_si128( (__m128i*)(padded + x*new_Y + y), _mm_loadu_si128( (__m128i*)(in + (x-1) + (y-1)*data_size_X) ));
				padded[x*new_Y + y] = in[(x-1) + (y-1)*data_size_X];
		}
	}
	*/
	int new_X = data_size_X+2;
    int new_Y = data_size_Y+2;
    float padded[new_X * new_Y];
	/*if (padded == 0) {
			printf("ERROR: Out of memory\n");
			return 1; }*/
	int curr = 0, in_count = 0;
	for (; curr < new_X; curr++)
		padded[curr] = 0;
	for (int i = 0; i < data_size_Y; i++) { //the amount of rows of input
		padded[curr] = 0;
		curr++;
		int j;
		for (j = 0; j < data_size_X-16; j+=16, curr+=16, in_count+=16) {
			_mm_storeu_ps((padded + curr + 0), _mm_loadu_ps((__m128*)(in + in_count + 0)));
			_mm_storeu_ps((padded + curr + 4), _mm_loadu_ps((__m128*)(in + in_count + 4)));
			_mm_storeu_ps((padded + curr + 8), _mm_loadu_ps((__m128*)(in + in_count + 8)));
			_mm_storeu_ps((padded + curr + 12), _mm_loadu_ps((__m128*)(in + in_count + 12)));
		}
		int rem = j;
		for (; j < (data_size_X-rem)/4*4; j+=4, curr+=4, in_count+=4) {
			_mm_storeu_ps((padded + curr), _mm_loadu_ps((__m128*)(in + in_count)));
		}
		for (; j < data_size_X; j++, curr++, in_count++) {
			_mm_storeu_ps((padded + curr), _mm_loadu_ps((__m128*)(in + in_count)));
		}
		padded[curr] = 0;
		curr++;
	}
	int rem = curr;
	for (; curr < new_X + rem; curr++)
		padded[curr] = 0;

	/*for (int a = 0; a < 150; a++) {
		printf("padded & in %d %f %f \n", a, padded[a+35], in[a]);
	}*/

    // main convolution loop
	printf("entering kernel loop");

	//You should use SSE vectors to calculate 4-pixels of output per iteration.
	/*float k0[4] = { kernel[0], kernel[0], kernel[0], kernel[0] };
	float k1[4] = { kernel[1], kernel[1], kernel[1], kernel[1] };
	float k2[4] = { kernel[2], kernel[2], kernel[2], kernel[2] };
	float k3[4] = { kernel[3], kernel[3], kernel[3], kernel[3] };
	float k4[4] = { kernel[4], kernel[4], kernel[4], kernel[4] };
	float k5[4] = { kernel[5], kernel[5], kernel[5], kernel[5] };
	float k6[4] = { kernel[6], kernel[6], kernel[6], kernel[6] };
	float k7[4] = { kernel[7], kernel[7], kernel[7], kernel[7] };
	float k8[4] = { kernel[8], kernel[8], kernel[8], kernel[8] };
	*/

	__m128 k8 = _mm_set_ps1(kernel[8]);
	__m128 k7 = _mm_set_ps1(kernel[7]);
	__m128 k6 = _mm_set_ps1(kernel[6]);
	__m128 k5 = _mm_set_ps1(kernel[5]);
	__m128 k4 = _mm_set_ps1(kernel[4]);
	__m128 k3 = _mm_set_ps1(kernel[3]);
	__m128 k2 = _mm_set_ps1(kernel[2]);
	__m128 k1 = _mm_set_ps1(kernel[1]);
	__m128 k0 = _mm_set_ps1(kernel[0]);

	//__m128 i = _mm_set_ps(0.0, 1.0, 2.0, 3.0);
	printf("hello");
	for(int y = 1; y < new_Y-1; y++) { // the y coordinate of theoutput location we're focusing on
		for(int x = 1; x < new_X-4; x+=4) { // the x coordinate of the output location we're focusing on
		//*remember to flip the kernel coordinates
		/*
		float kern[9] = { kernel[8], kernel[5], kernel[2],
					kernel[7], kernel[4], kernel[1],
					kernel[6], kernel[3], kernel[0]};
		__m128 m1 = _mm_mul_ps(_mm_loadu_ps((__m128*) kern), _mm_loadu_ps((__m128*) (padded + (x-1) + (y-1)*new_X)));
		__m128 m2 = _mm_mul_ps(_mm_loadu_ps((__m128*) (kern+3)), _mm_loadu_ps((__m128*) (padded + (x-1) + (y)*new_X)));
		__m128 m3 = _mm_mul_ps(_mm_loadu_ps((__m128*) (kern+6)), _mm_loadu_ps((__m128*) (padded + (x-1) + (y+1)*new_X)));
		__m128 sum_vec = _mm_add_ps(m1, m2);
		sum_vec = _mm_add_ps(sum_vec, m3);
		float sum[4] = {0.0, 0.0, 0.0, 0.0};
		_mm_storeu_ps(sum, sum_vec);
		//printf("sum: %f, %f, %f \n", sum[0], sum[1], sum[2]);
		out[(x-1) + (y-1)*data_size_X] = sum[0] + sum[1] + sum[2];
		
		if (x == 1 && y == 1) {
			printf("0 in %f padded %f \n", in[(x)*data_size_Y +(y)], padded[(x+1)*new_Y + (y+1)]);
			printf("1 in %f padded %f \n", in[(x-1)*data_size_Y +(y)], padded[(x)*new_Y + (y+1)]);
			printf("3 in %f padded %f \n", in[(x)*data_size_Y +(y-1)], padded[(x+1)*new_Y + (y)]);
			printf("4 in %f padded %f \n", in[(x-1)*data_size_Y +(y-1)], padded[(x)*new_Y + (y)]);
		}
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
		
		__m128* l8 = _mm_load_ps(padded + -1 + (y-1)*new_X); 
		__m128 l7 = _mm_set_ps1(((y-1)*new_X));
		__m128 l6 = _mm_set_ps1((1 + (y-1)*new_X));
		__m128 l5 = _mm_set_ps1((-1 + (y)*new_X));
		__m128 l4 = _mm_set_ps1(((y)*new_X));
		__m128 l3 = _mm_set_ps1((+1 + (y)*new_X));
		__m128 l2 = _mm_set_ps1((-1 + (y+1)*new_X));
		__m128 l1 = _mm_set_ps1(((y+1)*new_X));
		__m128 l0 = _mm_set_ps1((+1 + (y+1)*new_X));
		*/
		printf("getting loc");
		__m128 l8 = _mm_load_ps((__m128*)(padded + (x-1) + (y-1)*new_X)); //gets the next 8 pixels in vectors
		__m128 l7 = _mm_load_ps((__m128*)(padded + (x) + (y-1)*new_X)); 
		__m128 l6 = _mm_load_ps((__m128*)(padded + (x+1) + (y-1)*new_X)); 
		__m128 l5 = _mm_load_ps((__m128*)(padded + (x-1) + (y)*new_X)); 
		__m128 l4 = _mm_load_ps((__m128*)(padded + (x) + (y)*new_X)); 
		__m128 l3 = _mm_load_ps((__m128*)(padded + (x+1) + (y)*new_X)); 
		__m128 l2 = _mm_load_ps((__m128*)(padded + (x-1) + (y+1)*new_X)); 
		__m128 l1 = _mm_load_ps((__m128*)(padded + (x) + (y+1)*new_X)); 
		__m128 l0 = _mm_load_ps((__m128*)(padded + (x+1) + (y+1)*new_X)); 
		//printf("l0 %f, %f, %f, %f \n", l0);
		//the 9 kernel*pixel vectors
		printf("multiplying");
		__m128 n0 = _mm_mul_ps(k0, l0); //multiplying the corresponding kernel to the pixel vectors
		__m128 n1 = _mm_mul_ps(k1, l1);
		__m128 n2 = _mm_mul_ps(k2, l2);
		__m128 n3 = _mm_mul_ps(k3, l3);
		__m128 n4 = _mm_mul_ps(k4, l4);
		__m128 n5 = _mm_mul_ps(k5, l5);
		__m128 n6 = _mm_mul_ps(k6, l6);
		__m128 n7 = _mm_mul_ps(k7, l7);
		__m128 n8 = _mm_mul_ps(k8, l8);
		printf("summing up");
		__m128 c1 = _mm_add_ps(n8, n5); 
		c1 = _mm_add_ps(c1, n2); //c1 = column1 sums
		__m128 c2 = _mm_add_ps(n7, n4);
		c2 = _mm_add_ps(c2, n1); //c2 = column2 sums
		__m128 c3 = _mm_add_ps(n6, n3); 
		c3 = _mm_add_ps(c3, n0); //c3 = column3 sums
		__m128 r_vec = _mm_add_ps(c1, c2); // r_vec is the sum of the column sums
		r_vec = _mm_add_ps(r_vec, c3); 
		_mm_storeu_ps((out+(x-1)+(y-1)*data_size_X), r_vec); //stores result vector into x, x+1, x+2, x+3
		//printf("FINAL: %f %f %f %f \n", out[(x-1) + (y-1)*data_size_X], out[(x) + (y-1)*data_size_X], out[(x+1) + (y-1)*data_size_X], out[(x+2) + (y-1)*data_size_X]);
		
		}
	}
	return 1;

}
