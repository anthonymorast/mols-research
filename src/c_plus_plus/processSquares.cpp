/* Libraries */
#include <iostream>
#include <vector>
#include <string>  
#include <fstream> // file I/O
#include <sstream> // string to int
#include <stdlib.h> // atol 
#include <pthread.h> // multithreading
#include "mpi.h" // multithreading
#include <ctype.h> // isDigit - valid args
#include <stdio.h> // printf
#include <time.h> // program timing - clock
#include <dirent.h> // directory operations
#include <sys/stat.h> // mkdir

/* Custom Classes and Libraries */
#include "LatinSquare.h" // Latin square class

using namespace std;

/* Globals, Structs, etc. */
int SQUARE_ORDER = 0;
int NUMBER_THREADS = 1;

struct GenerateReduceParams 
{
	int id;
	vector<LatinSquare> squares;
	vector< vector<int> > permutations;
};

/* Functions */
bool valid_args(int arg_count, char* args[]);
void print_usage();
vector<LatinSquare> generate_reduced_squares(vector< vector<int> > permutations, vector<LatinSquare> normalizedSquares);
vector< vector<int> > read_permutations(string filename);
vector<LatinSquare> read_normalized_squares(string filename);
void find_orthogonal_squares(vector<LatinSquare> reducedSquares, string filename);
long string_to_formatted_long(string value);
void *threaded_reduce (void *params);

/*
* main - main flow of the program 
*
* Program Parameters: 
*  normalized squares file: must follow format outlined in docs/formats/normalizedSqaureFiles.md
*  permutations of order n-1 file: must follow the format outlined in docs/formats/permutationFiles.md
*  output file name: name of the output file
*  cores, threads: How many threads to launch
*/
int main (int argc, char* argv[]) 
{
	if (!valid_args(argc, argv)) 
	{
		return 0;
	}
	
	int numThreads = 8;
	stringstream(argv[4]) >> numThreads;
	NUMBER_THREADS = numThreads;

	string normalizedFileName(argv[1]);
	string permutationFile(argv[2]);
	string outputFile(argv[3]);
	
	// clock_t start, end;
	// start = clock();

	cout << endl;
	cout << "Reading normalized squares file \"" << normalizedFileName << "\"..." << endl;  
	vector<LatinSquare> normalizedSquares = read_normalized_squares(normalizedFileName);

	cout << "Reading permuations file \"" << permutationFile << "\"..." << endl;
	vector< vector<int> > permutations = read_permutations(permutationFile);

	cout << "Generating reduced Latin squares of order " <<  SQUARE_ORDER << "..." << endl; 
	vector<LatinSquare> reducedSquares = generate_reduced_squares(permutations, normalizedSquares);

	cout << endl;
	cout << "Normalized Squares: " << normalizedSquares.size() << endl;
	cout << "Permutations: " << permutations.size() << endl;
	cout << "Reduced Squares: " << reducedSquares.size() << endl;
	cout << endl;
	 
	cout << "Finding sets mutually orthogonal Latin squares..." << endl;
	find_orthogonal_squares(reducedSquares, outputFile);
	cout << endl;

	// end = clock();

	// float diff ((float)end - (float)start);
	// cout << "Total run time (in seconds): " << diff/CLOCKS_PER_SEC << endl; 
	//TODO - talk to Karlsson about thread safe timing
}

/*
* find_orthogonal_squares
* 
* finds all squares orthogonal with one another in the reducedSquares vector and prints them to file 'filename'
*
* @param reducedSquares - vector containing all reduced squares of a particular order
* @param filename - name of the file to be written
* @param numThreads - number of threads to run
*/
void find_orthogonal_squares(vector<LatinSquare> reducedSquares, string filename)
{
	ofstream fout;
	fout.open(filename.c_str(), std::ofstream::out);
	if (!fout.is_open())
	{
		cout << "There was an error opening the output file \"" << filename << "\" exiting...";
		exit(0);
	}

	bool writeLog = true;
	ofstream logOut;
	DIR *logDir;
	logDir = opendir("./log");
	if (logDir == NULL)
	{
		mkdir("./log", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	}
	logOut.open("./log/log.txt", std::ofstream::out);
	if (!logOut.is_open()) 
	{
		writeLog = false;
	}
	
	int rank;	// id/rank
	int comm_sz; 	// number of nodes/processors
	vector<LatinSquare> mySquares;

	// Break problem into smaller parts (be sure to assign first threads more squares than later threads)
	// Launch the threads (use MPI here probably)
	MPI_Init(NULL, NULL):
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// get squares for the thread
	int count = 1;
	for (vector<LatinSquare>::iterator it = mySquares.begin(); it != mySquares.end(); it++)
	{
		printf("\tThread %d processing square %d of %d...\n", rank, count, mySquares.size());
		LatinSquare currentSquare = *it;
		for (int i = 0; i < squares.size(); i++)
		{
			LatinSquare checkSquare = squares.at(i);
			// check orthogonal
		}
		count++;
		if (count % 100 == 0)
		{
			// write current square to log file
		}
	}
	
	// If there is a MO square write it to fout
	// Maybe open the files in each MPI thread and have a mutually exclusive section to write to them
	//    then write the current square being processed out of how many for each thread
	// Consider getting the number of cores for each MPI node and splitting the squares again 
	//    into smaller pthreads

	fout.close();
	logOut.close();
}

/*
* generate_reduced_squares
* 
* generates the reduced latin squares of the order indicated by any Latin square in the normalizedSquares vector.
*
* @param permutations - all possible permutations of the numbers 1..(n-1) where n is the order of Latin square
* @param normalizedSquares - a collection of Latin squares to be permuted
*
* @return a vector of reduced Latin squares of a particular order
*/
vector<LatinSquare> generate_reduced_squares(vector< vector<int> > permutations, vector<LatinSquare> normalizedSquares)
{
	vector<LatinSquare> squares;

	pthread_t *threads = new pthread_t[NUMBER_THREADS];
	GenerateReduceParams *params = new GenerateReduceParams[NUMBER_THREADS];

	int totalCount = 0;
	int each = normalizedSquares.size() / NUMBER_THREADS;
	int lastSize = 
		each * NUMBER_THREADS != normalizedSquares.size() ?
		(normalizedSquares.size() - (each * NUMBER_THREADS-1)) + each-1 :
		each;

	int currParam = 0;
	int count = 0;
	for (vector<LatinSquare>::iterator it = normalizedSquares.begin(); it != normalizedSquares.end(); it++) 
	{
		if (currParam < NUMBER_THREADS-1 && count == each)
		{
			currParam++;
			count = 0;
		}
		LatinSquare sq = *it;
		params[currParam].squares.push_back(sq);
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
	int id = param->id;

	for (int i = 0; i < squares.size(); i++) 
	{
		LatinSquare currSquare = squares.at(i);
		for (int j = 0; j < permutations.size(); j++) 
		{
			vector<int> permutation = permutations.at(j);
			param->squares.push_back(currSquare.PermuteRows(permutation));
		}
		// printf("\tThread %d processing square %d of %d...\n", id, i, squares.size());
	}
}

/*
* read_permutations
* 
* reads all permutations from the permutationsFile
*
* @param filename - name of the file containing the permutations
*
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
	permutation.push_back(1);
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
*
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

			int current = j == 6 ? ((longValue) / divisor) : ((longValue % mod) / divisor);
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

				int current = j == 6 ? ((longValue) / divisor) : ((longValue % mod) / divisor);
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
