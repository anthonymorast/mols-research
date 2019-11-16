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

__global__ void generate_squares(short** squareList, int order, short** newSquares)
{

}

int main()
{
	printf("this is a string\n");

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

	return 0;
}
