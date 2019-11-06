#include <iostream>

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

/*
	squares - 2D -> 1D array of Latin Squares
	pitch - CUDA way to access the 2D-ness of the array
	order - LS order 
	numSquares - number of squares per batch 
*/
__global__ void threadBlocks(int *squares, size_t pitch, int order, int numSquares) 
{

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
	int *d_a;
	int order = 3;
	int numSquares = N;
	int size = N * sizeof(int);

	/*
		To get the 2D array onto the card, the API is a little
		confusing. To work around, we could always create a 2D 
		array of squares and map that to a 1D array. Then pass
		and manipulate the 1D array and, upon returning, we should
		have just a 1D array (is square x mol to y).
	*/

	// sets aside some memory on the GPU for our data 
	cudaMalloc((void**)&d_a, size);

	// copy values to device only need to do this if the device needs to 
	// use the values in an array for computations. If we need to access 
	// arrray 'a' to get its value, must copy a. If we need to only
	// write to an array (never read) then don't need to copy 	
//	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

	threadBlocks<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a);

	// Copy back to host
	cudaMemcpy(c, d_a, size, cudaMemcpyDeviceToHost);

	for(int i = 0; i < 200; i++)
		cout << c[i] << endl;

	free(a); free(c);
	cudaFree(d_a);

	return 0;	
}
