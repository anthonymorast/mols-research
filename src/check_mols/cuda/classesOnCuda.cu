#include <iostream>
#include <stdio.h>
//#include "LatinSquare.h"

using namespace std;

/*
	RESOURCES:
		- http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#axzz4jNb3G5t6  (2D Arrays)


	NOTES:
	Can set these values however we please within the max for the card.
	According to the CUDA device props the GTX 1080 has a max of 1024 
	threads per block. I haven't found the max number of blocks yet. 

	An example. If we want to batch process 1,000,000 squares at a time, 
	we could have 1000 blocks and 1000 threads and special case the
	remainder.

	In our case, since we calculate the blocks as N/THREADS_PER_BLOCK, 
	one good way to think about N is the number of values in each batch.
	If we have 200 squares / batch then N = 200. If we want to process
	1,000,000 squares each run, then N = 1,000,000. Alternatively,
	we could replace <<<N/THREADS_PER_BLOCK, ...>>> with just 
	<<<N, ...>>> and have N * THREADS_PER_BLOCK = batch size.
*/
#define N 200 					// Number blocks -- might want to check what this actually is (2048*2048) is provided frequently 
#define THREADS_PER_BLOCK 2 	// Number Threads -- output from http://github.com/anthonyam12/cuda/check_card.cu is 1024 threads

__global__ void processSquares(int *checkSquare, int* allSquares, int order, int numSquares))
{
	printf("c1.x: %d\tc2.x: %d\n", c1->x, c2->x);
	for(int i = 0; i < c2->x; i++)
	{
		c2->data[i] = c1->data[0] + 10;
		printf("%d :: %d\n", i, c1->data[0]);
	}
	
	c1->setData(1, 2);
}

int main()
{
	/* 
		!!!!! IMPORTANT !!!!!
		Must allocate 200*sizeof data type, the cudaMemcpy(...) copies
		'size' number of bytes. Without mult by data type we only get 
		the first 200 bytes (if the number of ints is 200) which is 
		the first 50 ints. 
	*/
//	int *d_a, *a;;
//	int order = 3;
//	int numSquares = N;
//	int size = N * sizeof(int);

//	a = (int*)malloc(size);
//	d_a = (int*)malloc(size);

	TestClass c1(20, 6);
	TestClass c2(10, 5);

	TestClass *classes = new TestClass[10];
	for(int i = 0; i < 10; i++)
	{
		classes[i].setData(i, 10);
	}

	c1.toString();
	c2.toString();

	TestClass *d_c1;
	TestClass *d_c2;
	TestClass *d_carr;

	cudaMalloc((void**)&d_c1, sizeof(TestClass));
	cudaMalloc((void**)&d_c2, sizeof(TestClass));
	cudaMalloc((void**)&d_carr, sizeof(TestClass)*10);
	
	cudaMemcpy(d_c1, &c1, sizeof(TestClass), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c2, &c2, sizeof(TestClass), cudaMemcpyHostToDevice);
	cudaMemcpy(d_carr, &classes, sizeof(TestClass)*10, cudaMemcpyHostToDevice);
	
	// allocate and copy class data to device
	int *d_data1, *d_data2;
	int *d_dataarr[10];
	cudaMalloc((void**)&d_data1, sizeof(int)*c1.x);
	cudaMalloc((void**)&d_data2, sizeof(int)*c2.x);
	for(int i = 0; i < 10; i++)
	{
		cudaMalloc((void**)&d_dataarr[i], sizeof(classes[i].data));
		cout << sizeof(classes[i].data) << endl;
		cudaMemcpy(d_dataarr[i], classes[i].data, sizeof(classes[i].data), cudaMemcpyHostToDevice);
		cudaMemcpy(&(d_carr[i]), &d_dataarr, sizeof(int*), cudaMemcpyHostToDevice);
	}

	// copy class data to memory allocated on device for cclass data 
	cudaMemcpy(d_data1, c1.data, sizeof(int)*c1.x, cudaMemcpyHostToDevice);
	cudaMemcpy(d_data2, c2.data, sizeof(int)*c2.x, cudaMemcpyHostToDevice);

	// set pointer of data in device class (d_c1, d_c2) to allocated data 
	cudaMemcpy(&(d_c1->data), &d_data1, sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_c2->data), &d_data2, sizeof(int*), cudaMemcpyHostToDevice);

	classTest<<<1, 1>>>(d_c1, d_c2, d_carr);

	cudaMemcpy(&c2, &d_c2, sizeof(TestClass), cudaMemcpyDeviceToHost);
	// Since d_data# is the memory for the data that is actually put on the device
	// we want to copy this back to the host. Copying the device class' data
	// d_c2->data, gives us just pointer values rather than the actually values
	// at the memory locations of the device class' values. 
	cudaMemcpy(c2.data, d_data2, sizeof(int)*c2.x, cudaMemcpyDeviceToHost);


	c2.toString();
	/*
		To get the 2D array onto the card, the API is a little
		confusing. To work around, we could always create a 2D 
		array of squares and map that to a 1D array. Then pass
		and manipulate the 1D array and, upon returning, we should
		have just a 1D array (is square x mol to y).
	*/

	// sets aside some memory on the GPU for our data 
//	cudaMalloc((void**)&d_a, size);
//	cudaMalloc((void**)&a, size);

	// copy values to device only need to do this if the device needs to 
	// use the values in an array for computations. If we need to access 
	// arrray 'a' to get its value, must copy a. If we need to only
	// write to an array (never read) then don't need to copy 	
//	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

//	threadBlocks<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a);

	// Copy back to host
//	cudaMemcpy(c, d_a, size, cudaMemcpyDeviceToHost);

//	for(int i = 0; i < 200; i++)
//		cout << c[i] << endl;

//	free(a); free(c);
//	cudaFree(d_a);

	return 0;	
}
