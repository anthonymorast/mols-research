#include "../../LatinSquare/LatinSquare.h"

#include <stdio.h>

__device__ void add_one(int *num)
{
	num[0] += 1;
}

__global__ void add_five(int *num)
{
	printf("%d\n", num[0]);
	add_one(num);
	printf("%d\n", num[0]);
	add_one(num);
	printf("%d\n", num[0]);
	add_one(num);
	printf("%d\n", num[0]);
	add_one(num);
	printf("%d\n", num[0]);
	add_one(num);
	printf("%d\n", num[0]);
}

__device__ void permute_rows(short* new_rows, short* values, short* new_values, int order)
{
	// assume this is done in the main device login (or on CPU)
	// new_values = (short*)malloc((order*order)*sizeof(short int));
	for(short i = 0; i < order; i++)
	{
		for(short j = 0; j < order; j++)
		{
			new_values[new_rows[i] * order + j] = values[i * order + j];
		}
	}
}

__device__ void permute_cols(short* new_cols, short* values, short* new_values, int order)
{
	for(short i = 0; i < order; i++)
	{
		for(short j = 0; j < order; j++)
		{
			new_values[j * order + i] = values[j * order + new_cols[i]];
		}
	}
}

__device__ void permute_symbols(short* syms, short* values, short* new_values, int order)
{
	short osq = order*order;
	for(short i = 0; i < osq; i++)
	{
		new_values[i] = syms[values[i]];
	}
}

__global__ void generate_squares(short* squareList, int order, short* newSquares, short* permutation)
{
	// this actually worked, first try bb
	short* new_col_values = (short*)malloc(sizeof(short) * order * order);
	permute_cols(permutation, squareList, new_col_values, order);
	for(int i = 0; i < order*order; i++)
	{
		printf("%d ", new_col_values[i]);
	}
	printf("\n");
	// get square array index
	// allocate three new value arrays (could probably just pass in the newSquares pointers)
	// call permute row, col, sym
	// store new values in newSquares
	// done.
}

int main()
{
	// probably easier to always use arrays.....
	// this seems to work for data copying.....
	// now just need to sub square logic.....
	// note also this worked for calling device functions....

	int *num = (int*)malloc(1);
	num[0] = 10;
	int *devNum;
	cudaMalloc((void**)&devNum, sizeof(int));
	cudaMemcpy(devNum, num, sizeof(int), cudaMemcpyHostToDevice);
	// how to set?
	add_five<<<1, 1>>>(devNum);

	// (<what to copy into>, <what we're copying from>, <what is the size>, <some global>)
	cudaMemcpy(num, devNum, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", num[0]);

	// need to flatten all of these 2d arrays into 1d arrays for cuda......
	int maxArraySize = 1; // number of cores?
	int order = 3;

	int squareArraySize = maxArraySize * (order*order) * sizeof(short);
	int newSquareArraySize = squareArraySize * 3;
	int permArraySize = order * sizeof(short);

	short* squares = (short*)malloc(squareArraySize);
	short* newSquares = (short*)malloc(newSquareArraySize);
	short* perm = (short*)malloc(permArraySize);

	// for testing
	// make a loop to actually set these things (i.e. flatter LS and put in squares array)
	perm[0] = 1; perm[1] = 2; perm[2] = 0;
	squares[0] = 0; squares[1] = 1; squares[2] = 2;
	squares[3] = 1; squares[4] = 2; squares[5] = 0;
	squares[6] = 2; squares[7] = 0; squares[8] = 1;
	// 0, 1, 2, 1, 2, 0, 2, 0, 1

	short *dev_squares;
	short *dev_perm;
	short *dev_new_squares;

	cudaMalloc((void**)&dev_squares, squareArraySize);
	cudaMalloc((void**)&dev_perm, permArraySize);
	cudaMalloc((void**)&dev_new_squares, newSquareArraySize);

	cudaMemcpy(dev_squares, squares, squareArraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_perm, perm, permArraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_new_squares, newSquares, newSquareArraySize, cudaMemcpyHostToDevice);

	generate_squares<<<1, 1>>>(dev_squares, order, dev_new_squares, dev_perm);

	cudaMemcpy(newSquares, dev_new_squares, newSquareArraySize, cudaMemcpyDeviceToHost);

	for(int i = 0; i < order*order*3; i++)
	{
		printf("%d ", newSquares[i]);
	}
	printf("\n");

	// Create 1 array to hold the squares to try (ARRAY1).
	// create 2nd array which is 3 times the size as ARRAY1.
	// each cuda thread will take its square, apply a row, column and symbol permutation
	// and store the 3 new squares in the new array.
	// The CPU will then add puting the new squares where they need to be (sqares to try/squares to print)
	// limit this, set a max size for ARRAY1 while keeping in mind ARRAY2 will be 3x the size
	// a short should be 2 bytes so 1,000,000 size is probably too much for, say, order 10 LS

	return 0;
}
