#include "../GenerateSquares.h"

#include <stdio.h>

using namespace std;

__device__ void permute_rows(short* new_rows, short* values, short* new_values, int order)
{
	// assume this is done in the main device login (or on CPU)
	// new_values = (short*)malloc((order*order)*sizeof(short int));
	for(short i = 0; i < order; i++)
	{
		for(short j = 0; j < order; j++)
		{
			new_values[i * order + j] = values[new_rows[i] * order + j];
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

__global__ void generate_squares(short* squareList, int order, short* newSquares,
	short* permutation, int maxBatchSize)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < maxBatchSize)
	{
		// where in the squarelist/new square list is this thread's data?
		int myOffset = idx * 3 * order * order;
		// where after the offset to we start storing the data in the new square list
		int rowOffset = 0;
		int colOffset = order*order;
		int symOffset = 2*(order*order);

		short* new_col_values = (short*)malloc(sizeof(short) * order * order);
		short* new_row_values = (short*)malloc(sizeof(short) * order * order);
		short* new_sym_values = (short*)malloc(sizeof(short) * order * order);

		permute_cols(permutation, squareList, new_col_values, order);
		permute_rows(permutation, squareList, new_row_values, order);
		permute_symbols(permutation, squareList, new_sym_values, order);

		for(int i = 0; i < order*order; i++)
		{
			newSquares[i + myOffset + rowOffset] = new_row_values[i];
			newSquares[i + myOffset + colOffset] = new_col_values[i];
			newSquares[i + myOffset + symOffset] = new_sym_values[i];
		}

		delete[] new_col_values;
		delete[] new_row_values;
		delete[] new_sym_values;
	}
}

void run_on_gpu(short* squaresToRun, int order, short* newSquares, short* perm,
	int squareArraySize, int permArraySize, int newSquareArraySize, int squaresToCheck)
{
	short* dev_squares; short* dev_perm; short* dev_new_squares;
	cudaMalloc((void**)&dev_squares, squareArraySize);
	cudaMalloc((void**)&dev_perm, permArraySize);
	cudaMalloc((void**)&dev_new_squares, newSquareArraySize);

	cudaMemcpy(dev_squares, squaresToRun, squareArraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_perm, perm, permArraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_new_squares, newSquares, newSquareArraySize, cudaMemcpyHostToDevice);

	// how many blocks do we need if we use nThreads threads?
	int nThreads = 256;
	int nBlocks = (squareArraySize + nThreads - 1) / nThreads;
	cout << nBlocks << "  " << squareArraySize << endl;
	generate_squares<<<nBlocks, nThreads>>>(dev_squares, order, dev_new_squares, dev_perm, squaresToCheck);

	cudaMemcpy(newSquares, dev_new_squares, newSquareArraySize, cudaMemcpyDeviceToHost);
}

void copy_to_vectors(short* newSquares, vector<LatinSquare> checkSqs,
	vector<LatinSquare> appendToSquares, int numberSquares, int order, bool updateCheckSquares)
{
	int osq = order*order;
	for(int i = 0; i < numberSquares; i++)
	{
		for(int k = 0; k < 3; k++)	// each square generates 3 new ones
		{
			short* values = (short*)malloc(sizeof(short)*osq);
			for(int j = 0; j < osq; j++)
			{
				values[j] = newSquares[(i*3*osq) + (k*osq) + j];
			}
			LatinSquare sq = LatinSquare(order, values);
			cout << sq << endl << endl;
			unique_add_to_vector(sq, appendToSquares, checkSqs, updateCheckSquares);
		}
	}
}

int main(int argc, char* argv[])
{
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

	int maxBatchSize = 1000; // some random value
	// 4352; // number of cores? (sure, although it might eat ram)
	long unsigned int numSqs;
	int permArraySize = order * sizeof(short);
	do {
		numSqs = allSqs.size();
		vector<LatinSquare> newSqVec;

		// flip inner/outer loops since we process many squares at a time for each permutations
		ifstream permfile; permfile.open(filename_n);
		string permline;
		while(getline(permfile, permline))
		{
			short* permArr = get_array_from_line(permline, order);

			while(checkSqs.size() > 0)
			{
				// only process up to maxBatchSize, in batches, to conserve RAM
				int squaresToCheck = checkSqs.size() > maxBatchSize ? maxBatchSize : checkSqs.size();

				int squareArraySize = squaresToCheck * osq * sizeof(short);
				int newSquareArraySize = squareArraySize * 3;

				short* squares = (short*)malloc(squareArraySize);
				short* newSquares = (short*)malloc(newSquareArraySize);

				for(int i = 0; i < squaresToCheck; i++) 	// each square
				{
					short* values = checkSqs.at(i).get_values();
					for(int j = 0; j < osq; j++)
					{
						squares[(i*osq) + j] = values[j];
					}
				}
				// remove first 'squaresToCheck' elements from the vector
				checkSqs.erase(checkSqs.begin(), checkSqs.begin() + squaresToCheck);

				run_on_gpu(squares, order, newSquares, permArr, squareArraySize,
					permArraySize, newSquareArraySize, squaresToCheck);

				// need to store newSqVec here instead so that we can only add
				// new unique squares to the checkSqs vector
				copy_to_vectors(newSquares, checkSqs, newSqVec, squaresToCheck, order, false);
			}

			delete[] permArr;
		}

		checkSqs.clear(); // just in case
		for(auto it = newSqVec.begin(); it != newSqVec.end(); it++)
		{
			unique_add_to_vector((*it), allSqs, checkSqs, true);
		}

		permfile.close();
		cout << "Start Count: " << numSqs << ", End Count: " << allSqs.size() << endl;
	} while(numSqs < allSqs.size());

	for(auto it = allSqs.begin(); it != allSqs.end(); it++)
	{
		cout << (*it) << endl << endl;
	}

	sqfile.close();
	return 0;
}
