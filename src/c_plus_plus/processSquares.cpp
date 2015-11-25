/* Libraries */
#include <iostream>
#include <vector>
#include <string>  
#include <fstream> // file I/O
#include <sstream> // string to int
#include <stdlib.h> // atol 

/* Custom Classes and Libraries */
#include "LatinSquare.h" // Latin square class

using namespace std;

/* Globals */
int SQUARE_ORDER = 0;

/* Functions */
bool valid_args(int arg_count, char* args[]);
void print_usage();
vector<LatinSquare> generate_reduced_squares(vector< vector<int> > permutations, vector<LatinSquare> normalizedSquares);
vector< vector<int> > read_permutations(string filename);
vector<LatinSquare> read_normalized_squares(string filename);
void find_orthogonal_squares(vector<LatinSquare> reducedSquares, string filename, int numThreads);
long string_to_formatted_long(string value);

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
	
	string normalizedFileName(argv[1]);
	string permutationFile(argv[2]);
	string outputFile(argv[3]);
	
	vector<LatinSquare> normalizedSquares = read_normalized_squares(normalizedFileName);
	vector< vector<int> > permutations = read_permutations(permutationFile);
	vector<LatinSquare> reducedSquares = generate_reduced_squares(permutations, normalizedSquares);

	cout << "Normalized Squares: " << normalizedSquares.size() << endl;
	cout << "Permutations: " << permutations.size() << endl;
	cout << "Reduced Squares: " << reducedSquares.size() << endl;
	 
	find_orthogonal_squares(reducedSquares, outputFile, numThreads);
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



	return squares;
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
	int i = 0;
	while (fin >> value) 
	{
		if ( i % permSize == 0 && i != 0) 
		{
			permutations.push_back(permutation);
			permutation.clear();
		}

		permutation.push_back(value);
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
* find_orthogonal_squares
* 
* finds all squares orthogonal with one another in the reducedSquares vector and prints them to file 'filename'
*
* @param reducedSquares - vector containing all reduced squares of a particular order
* @param filename - name of the file to be written
* @param numThreads - number of threads to run
*/
void find_orthogonal_squares(vector<LatinSquare> reducedSquares, string filename, int numThreads)
{
  
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