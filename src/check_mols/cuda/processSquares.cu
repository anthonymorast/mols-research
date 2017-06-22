#include <iostream>
#include <vector>
#include <string>  
#include <fstream> 	// file I/O
#include <sstream> 	// string to int
#include <stdlib.h> 	// atol 
#include <pthread.h>	// multithreading
#include <ctype.h> 	// isDigit - valid args
#include <stdio.h> 	// printf
#include <time.h> 	// program timing - clock !!!! switch to chrono (thread-safe I believe)
#include <dirent.h> 	// directory operations
#include <sys/stat.h> 	// mkdir
#include "LatinSquare.h"

using namespace std;

struct GenerateReduceParams 
{
	int id;
	vector<LatinSquare> squares;
	vector< vector<int> > permutations;
};

bool valid_args(int arg_count, char* args[]);
void print_usage();
vector<LatinSquare> generate_reduced_squares(vector< vector<int> > permutations, vector<LatinSquare> normalizedSquares);
vector< vector<int> > read_permutations(string filename);
vector<LatinSquare> read_normalized_squares(string filename);
long string_to_formatted_long(string value);
void *threaded_reduce (void *params);
vector<LatinSquare> generateReduced(char* argv[]);

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
#define N 1000000 				// Number blocks -- might want to check what this actually is (2048*2048) is provided frequently 
#define THREADS_PER_BLOCK 1024 	// Number Threads -- output from http://github.com/anthonyam12/cuda/check_card.cu is 1024 threads

int SQUARE_ORDER = 0;
int NUMBER_THREADS = 1;
std::ofstream *fout;
std::ofstream *logOut;
pthread_mutex_t out_file_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t log_file_mutex = PTHREAD_MUTEX_INITIALIZER;
bool writeLog;

/*
	This cuda function parallelizes the checking of each square with all the squares in a batch.
	If the squares are MO mutuallyOrth = 1 and molSq is the square that checkSquare is MO with.

	checkSquare - the square to compare to others
	batchSquares - an array of squares to be compared with the checkSquare
	mutuallyOrtho - a flag: 1 => mols, 0 => not orthogonal
	molSq - if mutuallyOrtho == 1, this square is the sq that checkSq is MO with
	order - order of LS
	numSquares - not used, yet...
*/
__global__ void processSquares(int *checkSquare, int* batchSquares, int* mutuallyOrtho, int* molSq,
							   int *order, int *numSquares)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	batchSquares[index] = index;
}

int main(int argc, char* argv[])
{
	if (!valid_args(argc, argv)) 
	{
		return 0;
	}

	/* 
		!!!!! IMPORTANT !!!!!
		Must allocate 200*sizeof data type, the cudaMemcpy(...) copies
		'size' number of bytes. Without mult by data type we only get 
		the first 200 bytes (if the number of ints is 200) which is 
		the first 50 ints. 
	*/
	vector<LatinSquare> reducedSquares = generateReduced(argv);
	int totalSquares = reducedSquares.size();
	int orderSquared = SQUARE_ORDER * SQUARE_ORDER;

	int *mutuallyOrtho = (int*)malloc(sizeof(int));
	int *order = (int*)malloc(sizeof(int));
	int *molSq = (int*)malloc(sizeof(int) * orderSquared);
	int *numSquares = (int*)malloc(sizeof(int));
	int *allSquares = (int*)malloc(sizeof(int) * reducedSquares.size() * orderSquared);

	for (int i = 0; i < totalSquares; i++)
	{
		int* thisSq = reducedSquares[i].ToArray();
		int start = orderSquared * i; // end = start + orderSquared

		for(int j = 0; j < orderSquared; j++) 
		{
			allSquares[start + j] = thisSq[j];
		}
	}
	
//	for (vector<LatinSquare>::iterator it = reducedSquares.begin(); it != reducedSquares.end(); it++)
//	{
//		int *checkSquare, int* batchSquares, int* mutuallyOrtho, int* molSq,
//							   int *order, int *numSquares

		int* checkSq = reducedSquares[0].ToArray();//(*it).ToArray();
		int bSize = totalSquares < N ? totalSquares : N;
		numSquares[0] = bSize;

		int* d_checkSq, *d_molSq, *d_order, *d_mutuallyOrtho, *d_allSquares, *d_numSquares;

		// create memory on cuda device 
		cudaMalloc((void**)&d_checkSq, sizeof(int) * orderSquared);
		cudaMalloc((void**)&d_molSq, sizeof(int) * orderSquared);
		cudaMalloc((void**)&d_order, sizeof(int));
		cudaMalloc((void**)&d_mutuallyOrtho, sizeof(int));
		cudaMalloc((void**)&d_allSquares, sizeof(int) * orderSquared * bSize);
		cudaMalloc((void**)&d_numSquares, sizeof(int));

		// copy values from host to device
		cudaMemcpy(d_checkSq, checkSq, sizeof(int) * orderSquared, cudaMemcpyHostToDevice);
		cudaMemcpy(d_molSq, molSq, sizeof(int) * orderSquared, cudaMemcpyHostToDevice);
		cudaMemcpy(d_order, order, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_mutuallyOrtho, mutuallyOrtho, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_allSquares, allSquares, sizeof(int) * orderSquared * bSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_numSquares, numSquares, sizeof(int),  cudaMemcpyHostToDevice);

		processSquares<<<bSize/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_checkSq, d_allSquares, 
					  d_mutuallyOrtho, d_molSq, d_order, d_numSquares);

		cudaMemcpy(allSquares, d_allSquares, sizeof(int) * orderSquared * bSize, cudaMemcpyDeviceToHost);
		for (int i = 0; i < totalSquares; i++)
		{
			cout << bSize << " " << bSize/THREADS_PER_BLOCK << endl;
		}
//	}

	// sets aside some memory on the GPU for our data 

//	processSquares<<<batchSize/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a);

	// Copy back to host

	// free all memory

	return 0;	
}

vector<LatinSquare> generate_reduced_squares(vector< vector<int> > permutations, vector<LatinSquare> normalizedSquares)
{
	vector<LatinSquare> squares;

	pthread_t *threads = new pthread_t[NUMBER_THREADS];
	GenerateReduceParams *params = new GenerateReduceParams[NUMBER_THREADS];

	int each = normalizedSquares.size() / NUMBER_THREADS;
	int lastSize = 
		each * NUMBER_THREADS != normalizedSquares.size() ?
		(normalizedSquares.size() - (each * NUMBER_THREADS-1)) + each-1 :
		each;

	int index = 0;
	int count = 0;
	for (vector<LatinSquare>::iterator it = normalizedSquares.begin(); it != normalizedSquares.end(); it++) 
	{
		if (index < NUMBER_THREADS-1 && count == each)
		{
			index++;
			count = 0;
		}
		LatinSquare sq = *it;
		params[index].squares.push_back(sq);
		count++;
	}

	for (int i = 0; i < NUMBER_THREADS; i++)
	{
		params[i].id = i;
		params[i].permutations = permutations;
		pthread_create(&(threads[i]), NULL, threaded_reduce, &params[i]); 
	}

	for (int i = 0; i < NUMBER_THREADS; i++)
	{
		pthread_join(threads[i], NULL);
	}

	for (int i = 0; i < NUMBER_THREADS; i++)
	{
		squares.insert(squares.end(), params[i].squares.begin(), params[i].squares.end());
	}

	return squares;
}

/*
* threaded_reduce 
*
* This function is to be called by each individual thread to find a subset of the reduced Latin squares
* each vector will be contcatenated into the main vector after execution.
*
* @param params - GenerateReducedParams struct containing all parameters for this threaded function
*/
void *threaded_reduce (void *params)
{
	struct GenerateReduceParams *param = (struct GenerateReduceParams*)params;

	vector<LatinSquare> squares = param->squares;
	param->squares.clear();
	vector< vector<int> > permutations = param->permutations;

	for (int i = 0; i < squares.size(); i++) 
	{
		LatinSquare currSquare = squares.at(i);
		for (int j = 0; j < permutations.size(); j++) 
		{
			vector<int> permutation = permutations.at(j);
			param->squares.push_back(currSquare.PermuteRows(permutation));
		}
	}
}

vector<LatinSquare> generateReduced(char* argv[])
{	
	int numThreads = 8;
	stringstream(argv[4]) >> numThreads;
	NUMBER_THREADS = numThreads;

	string normalizedFileName(argv[1]);
	string permutationFile(argv[2]);
	string outputFile(argv[3]);

	fout = new std::ofstream(outputFile.c_str());
	DIR *logDir;
	logDir = opendir("./log");
	if (logDir == NULL)
	{
		mkdir("./log", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	}
	writeLog = true;
	logOut = new std::ofstream("./log/log.txt");
	if (!logOut->is_open()) 
	{
		writeLog = false;
	}
	
	// clock_t start, end;
	// start = clock();

	cout << endl;
	cout << "Reading normalized squares file \"" << normalizedFileName << "\"..." << endl;  
	vector<LatinSquare> normalizedSquares = read_normalized_squares(normalizedFileName);

	cout << "Reading permuations file \"" << permutationFile << "\"..." << endl;
	vector< vector<int> > permutations = read_permutations(permutationFile);

	cout << "Generating reduced Latin squares of order " <<  SQUARE_ORDER << "..." << endl; 
	return generate_reduced_squares(permutations, normalizedSquares);
}

/*
* read_permutations
* 
* reads all permutations from the permutationsFile
*
* @param filename - name of the file containing the permutations
* @return a vector of vectors where each inner vector is one permutation
*/
vector< vector<int> > read_permutations(string filename) 
{
 	vector< vector<int> > permutations;

 	ifstream fin;
	fin.open(filename.c_str());
	if (!fin.is_open() ) 
	{
		cout << "Unable to open permutations file \"" << filename << "\" exiting..." << endl;
		exit(0); 
	}

	int permSize;
	fin >> permSize;

	if ( permSize != (SQUARE_ORDER - 1) )
	{
		cout << "File \"" << filename << "\" is not in the correct format or ";
		cout << "contains permutations of the wrong size. Refer to /docs/formats/permutationsFiles.md ";
		cout << "for more information." << endl;
		exit(0);
	}

	int value;
	vector<int> permutation;
	permutation.push_back(1);	// first row will remain the same, this will map row 1 to row 1
	int i = 0;
	while (fin >> value) 
	{
		if ( i % permSize == 0 && i != 0) 
		{
			permutations.push_back(permutation);
			permutation.clear();
			permutation.push_back(1);
		}

		permutation.push_back(value + 1);
		i++;
	}
	permutations.push_back(permutation);

	fin.close();
 	return permutations; 
}

/*
* read_normalized_squares
* 
* reads in all normalized Latin squares from the normalizedLatinSquares file
*
* @param filename - name of the file containing the normalized Latin squares
* @return a vector of normalized Latin squares
*/
vector<LatinSquare> read_normalized_squares(string filename) 
{
	vector<LatinSquare> squares;

	ifstream fin;
	fin.open(filename.c_str());
	if (!fin.is_open() ) 
	{
		cout << "Unable to open normalized squares file \"" << filename << "\" exiting..." << endl;
		exit(0); 
	}

	// get the square order from the file (/docs/formats/normalizedSquareFiles.md)
	fin >> SQUARE_ORDER;

	string value;
	while (fin >> value) 
	{
		vector<int> values;
		long longValue = string_to_formatted_long(value);

		// Process first n numbers (otherwise the value from "while(fin >> value)" is ignored)
		for (int j = SQUARE_ORDER; j > 0; j--)
		{
			int divisor = 1;
			for (int k = 1; k < j; k++)
			{
				divisor *= 10;
			}
			int mod = divisor * 10;

			int current = j == SQUARE_ORDER ? ((longValue) / divisor) : ((longValue % mod) / divisor);
			values.push_back(current);
		}

		// Process n-1 subsequent values
		for (int i = 0; i < SQUARE_ORDER-1; i++) 
		{
			fin >> value;
			longValue = string_to_formatted_long(value);

			for (int j = SQUARE_ORDER; j > 0; j--)
			{
				int divisor = 1;
				for (int k = 1; k < j; k++)
				{
					divisor *= 10;
				}
				int mod = divisor * 10;

				int current = j == SQUARE_ORDER ? ((longValue) / divisor) : ((longValue % mod) / divisor);
				values.push_back(current);
			}
		}

		LatinSquare square(SQUARE_ORDER, values);
		values.clear();
		squares.push_back(square);
	}

	fin.close();
	return squares;
}

/*
* string_to_formatted_long
*
* Converts a string value in "McKay Format" (0-based, clustered e.g. 012345) to a usable format for us
* e.g. 12345.
*
* @param value - string value to be converted to long value
*/
long string_to_formatted_long(string value) 
{
	string formattedString;
	for (std::string::iterator it = value.begin(); it != value.end(); ++it)
	{
		int dummy;
		stringstream ss;
		ss << *it;

		string str;
		ss >> str;
		stringstream(str) >> dummy;

		stringstream ss1;
		dummy++;
		ss1 << dummy;
		formattedString += ss1.str();
	}

	return atol(formattedString.c_str());
}

/*
* valid_args
* 
* Validates the arguments passed into the program. Prints an error message and returns false
* if there are errors.
*
* @param arg_count - the number of arguments
* @param args - the vector containing the arguments
*
* @return true if valid, false otherwise
*/
bool valid_args (int arg_count, char* args[]) 
{
  	if (arg_count != 5) 
  	{
	    	cout << "Incorrect number of command line arguments." << endl;
	    	print_usage();
	    	return false;
  	}

  	string threadString = args[4];
  	for (int i = 0; i < threadString.length(); i++)
  	{
	  	if (!isdigit(args[4][i])) 
	  	{
	  		cout << "Number of threads must be an integer." << endl;
	  		print_usage();
	  		return false;
	  	}
	}
   	
   	return true;
}

/*
* print_usage
*
* Prints usage, instructing the user on how to use the program. 
*/
void print_usage() 
{
	cout << endl;
	cout << "Usage:" << endl;
	cout << "\tprocessSquares <normalized squares file> <permutations file> <output file name> <# threads>" << endl;
	cout << endl;
	cout << "Details:" << endl;
	cout << "\t<normalized square file> - must follow format outlined in docs/formats/normalizedSqaureFiles.md" << endl;
	cout << "\t<permutatins file> - must follow the format outlined in docs/formats/permutationFiles.md" << endl;
	cout << "\t<output filename> - the name of the file to be written to" << endl;
	cout << "\t<# threads> - the number of threads to be used in computations" << endl;
	cout << endl;
}
