#include "../GenerateSquares.h"

#include <stdio.h>

using namespace std;

__device__ void permute_rows(short* new_rows, short* values, short* newSquares,
	int order, int rowOffset, int myOffset)
{
	for(short i = 0; i < order; i++)
	{
		for(short j = 0; j < order; j++)
		{
			newSquares[(i * order) + myOffset + rowOffset + j] = values[new_rows[i] * order + j];
		}
	}
}

__device__ void permute_cols(short* new_cols, short* values, short* newSquares,
	int order, int colOffset, int myOffset)
{
	for(short i = 0; i < order; i++)
	{
		for(short j = 0; j < order; j++)
		{
			newSquares[(i + myOffset + colOffset) + (j * order)] = values[j * order + new_cols[i]];
		}
	}
}

__device__ void permute_symbols(short* syms, short* values, short* newSquares,
	int order, int symOffset, int myOffset)
{
	short osq = order*order;
	for(short i = 0; i < osq; i++)
	{
		newSquares[i + myOffset + symOffset] = syms[values[i]];
	}
}

__global__ void generate_squares(short* squareList, int order, short* newSquares,
	short* perms, int batchSize, int totalPerms)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < batchSize)
	{
		int osq = order*order;
		int myOffset = idx * totalPerms * osq * 3; // where in the new square list is this thread's data?
		int squareOffset = idx * osq; // where in the squares list is this thread's data?

		// where after the offset to we start storing the data in the new square list
		int rowOffset = 0;
		int colOffset = osq;
		int symOffset = 2*(osq);
		short* my_square = (short*)malloc(sizeof(short) * osq);	// add squareOffset to function calls
		short* perm = (short*)malloc(sizeof(short)*order);

		for(int i = 0; i < osq; i++)
		{
			my_square[i] = squareList[i + squareOffset];
		}

		for(int pCount = 0; pCount < totalPerms; pCount++)
		{
			for(int i = 0; i < order; i++)
			{
				perm[i] = perms[(pCount*order) + i];
			}
			permute_cols(perm, my_square, newSquares, order, colOffset, myOffset);
			permute_rows(perm, my_square, newSquares, order, rowOffset, myOffset);
			permute_symbols(perm, my_square, newSquares, order, symOffset, myOffset);
			myOffset += (osq*3);
		}
		delete[] my_square;			//!!!! ALWAYS FREE YOUR MEMORY !!!!!
		delete[] perm;
	}
}

void run_on_gpu(short* squaresToRun, int order, short* newSquares, short* perm,
	int squareArraySize, int permArraySize, int newSquareArraySize,
	int squaresToCheck, int totalPerms, short *dev_squares, short* dev_perm, short* dev_new_squares)
{
	cudaMemcpy(dev_squares, squaresToRun, squareArraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_perm, perm, permArraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_new_squares, newSquares, newSquareArraySize, cudaMemcpyHostToDevice);

	// how many blocks do we need if we use nThreads threads?
	int nThreads = 128;
	int nBlocks = (squareArraySize + nThreads - 1) / nThreads;
	generate_squares<<<nBlocks, nThreads>>>(dev_squares, order, dev_new_squares, dev_perm, squaresToCheck, totalPerms);

	cudaMemcpy(newSquares, dev_new_squares, newSquareArraySize, cudaMemcpyDeviceToHost);
}

void copy_to_vectors(short* newSquares, vector<LatinSquare> &checkSqs,
	vector<LatinSquare> &appendToSquares, int numberSquares, int order, bool updateCheckSquares,
	int totalPerms)
{
	int osq = order*order;
	long iterRange =  numberSquares*totalPerms*3*osq;
	short valuesIdx;
	short* values;
	for(long i = 0; i < iterRange; i++)
	{
		if(i % osq == 0)
		{
			if(i > 0)
			{
				LatinSquare sq = LatinSquare(order, values);
				unique_add_to_vector(sq, appendToSquares, checkSqs, updateCheckSquares);
			}
			valuesIdx = 0;
			values = (short*)malloc(sizeof(short)*osq);
		}
		values[valuesIdx] = newSquares[i];
		valuesIdx++;
	}
}

int main(int argc, char* argv[])
{

	// timers
	clock_t start, end;
	start = clock();

	// TODO: make some things globals (e.g. order, osq) to stop passing it around
	if (argc < 3)
	{
		print_usage();
		return 0;
	}

	short order = stoi(string(argv[1]));
	short osq = order*order;
	string filename_iso = string(argv[2]);
	string filename_3 = "3_perm.dat";
	string filename_n = to_string(order) + "_perm.dat";
	bool cont = true;

	if (!file_exists(filename_n))
	{
		cout << filename_n << " does not exist. Please use the utilites to generate the file." << endl;
		cont = false;
	}
	if(!file_exists(filename_iso))
	{
		cout << filename_iso << " does not exist." << endl;
		cont = false;
	}

	if (!cont)
		return 0;

	end = clock();
	double paramTime = double(end-start) / double(CLOCKS_PER_SEC);
	cout << "CUDA Param Time Taken: " << paramTime << " seconds" << endl;

	start = clock();

	ifstream isofile; isofile.open(filename_iso);
	ofstream sqfile; sqfile.open(to_string(order) + "_squares.dat");

	string line;
	vector<LatinSquare> allSqs;
	vector<LatinSquare> checkSqs;		// squares to permute, do not permute all squares everytime
	while(getline(isofile, line))
	{
		LatinSquare isoSq(order, get_array_from_line(line, osq));
		allSqs.push_back(isoSq);
		checkSqs.push_back(isoSq);
	}
	isofile.close();

	long totalPerms = my_factorial(order);
	short* perms = (short*)malloc(sizeof(short) * totalPerms * order);
	vector<short*> permVec;
	ifstream permfile; permfile.open(filename_n);
	string permline;
	int count = 0;
	while(getline(permfile, permline))
	{
		short* permArr = get_array_from_line(permline, order);
		permVec.push_back(permArr);
		for(int i = 0; i < order; i++)
		{
			perms[(count*order) + i] = permArr[i];
		}
		count++;
	}
	permfile.close();

	end = clock();
	double fileTime = double(end-start) / double(CLOCKS_PER_SEC);
	cout << "CUDA FILE Read Time Taken: " << fileTime << " seconds" << endl;

	// some random value (maybe keep it divisible by nThreads which should be a multiple of 32)
	int maxBatchSize = 2048;
	// 4352; // number of cores? (sure, although it might eat ram)
	long unsigned int numSqs;
	int permArraySize = order * sizeof(short) * totalPerms;
	int lastSquaresToCheck = 0;
	short* dev_squares; short* dev_perm; short* dev_new_squares;
	do {
		numSqs = allSqs.size();
		vector<LatinSquare> newSqVec;

		// TODO: add a permutation batch
		int checkedSquares = 0;

		while(checkedSquares < checkSqs.size())
		{
			if(checkedSquares % (maxBatchSize * 3) == 0 && checkedSquares > 0)
			{
				printf("Checked %d out of %ld squares\n", checkedSquares, checkSqs.size());
			}
			// only process up to maxBatchSize, in batches, to conserve RAM
			int squaresToCheck = (checkSqs.size() - checkedSquares) > maxBatchSize
				? maxBatchSize : (checkSqs.size() - checkedSquares);
			int squareArraySize = squaresToCheck * osq * sizeof(short);
			int newSquareArraySize = squareArraySize * totalPerms * 3;

			short* squares = (short*)malloc(squareArraySize);
			short* newSquares = (short*)malloc(newSquareArraySize);

			for(int i = 0; i < squaresToCheck; i++) 	// each square
			{
				// start at the last index (ignore first squares that have been checked)
				short* values = checkSqs.at(checkedSquares + i).get_values();
				for(int j = 0; j < osq; j++)
				{
					squares[(i*osq) + j] = values[j];
				}
			}

			start = clock();
			if(lastSquaresToCheck != squaresToCheck)
			{
				if(lastSquaresToCheck > 0)
				{
					cudaFree(dev_squares);
					cudaFree(dev_perm);
					cudaFree(dev_new_squares);
				}
				cudaMalloc((void**)&dev_squares, squareArraySize);
				cudaMalloc((void**)&dev_perm, permArraySize);
				cudaMalloc((void**)&dev_new_squares, newSquareArraySize);
			}
			end = clock();
			double cudaMallocTime = double(end-start) / double(CLOCKS_PER_SEC);
			cout << "CUDA Malloc Time Taken: " << cudaMallocTime << " seconds" << endl;

			start = clock();
			run_on_gpu(squares, order, newSquares, perms, squareArraySize,
				permArraySize, newSquareArraySize, squaresToCheck, totalPerms,
				dev_squares, dev_perm, dev_new_squares);
			end = clock();
			double gpuRunTime = double(end-start) / double(CLOCKS_PER_SEC);
			cout << "CUDA GPU Run Time Taken: " << gpuRunTime << " seconds" << endl;

			// need to store newSqVec here instead so that we can only add
			// new unique squares to the checkSqs vector
			start = clock();
			copy_to_vectors(newSquares, checkSqs, newSqVec, squaresToCheck, order, false, totalPerms);
			end = clock();
			double copyTime = double(end-start) / double(CLOCKS_PER_SEC);
			cout << "CUDA Copy Time Taken: " << copyTime << " seconds" << endl;

			checkedSquares += squaresToCheck;
			lastSquaresToCheck = squaresToCheck;

			delete[] squares;
			delete[] newSquares;
		}

		checkSqs.clear(); // just in case
		for(auto it = newSqVec.begin(); it != newSqVec.end(); it++)
		{
			unique_add_to_vector((*it), allSqs, checkSqs, true);
		}

		cout << "Start Count: " << numSqs << ", End Count: " << allSqs.size() << endl;
	} while(numSqs < allSqs.size());

	end = clock();
	double timeTaken = double(end-start) / double(CLOCKS_PER_SEC);
	cout << "CUDA Time Taken: " << timeTaken << " seconds" << endl;

	for(auto it = allSqs.begin(); it != allSqs.end(); it++)
	{
		//cout << (*it) << endl << endl;
	}

	sqfile.close();
	return 0;
}
