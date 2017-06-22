#include <iostream>
#include <fstream>
#include <pthread.h>
#include <string>

#include "../LatinSquare.h"

/****	Functions	****/
int** readPermFile(string filename);
int** readSquares(string filename);
int factorial(int n);
void printUsage();

int main(int argc, char* argv[]) 
{
	// Not validating parameters (lazy), just checking number of args
	if (argc < 5)
	{
		printUsage();
		return 0;
	}

	int** perms3 = readPermFile(argv[1]);
	int** permsN = readPermFile(argv[2]);
	int** squares = readSquares(argv[3]);

	return 0;
}

/*
	Quick explanation of program usage. 
*/
void printUsage()
{
	cout << endl  << "Usage:" << endl;
	cout << "\tgenerateIsoClasses <3_perm filename> <order perm filename> <square filename> <num threads>" << endl << endl;
	cout << "NOTE: the permutation files must be in the format specified in docs/permutationFiles.md" << endl << endl; 
}

/*
	Reads Latin squares from a formatted file. 
	Format:
		Line 1: square order
		Other Lines: 
			- one Latin square per line
			- each value separated by a space
			- e.g. 1 2 2 1
			- e.g. 1 2 3 3 1 2 2 3 1
*/
int** readSquares(string filename)
{
	int** squares;

	ifstream fin;
	fin.open(filename.c_str());
	if(!fin.is_open())
	{
		cout << "Unable to open squares file, exiting..." << endl;
		exit(0);
	}

	int order;
	fin >> order;
	int orderSquared = order*order;
	
	int value;
	while(fin >> value)
	{
		// reads one value at a time
		cout << value << endl;
	}

	fin.clos();
	return squares;
}

/*
	Reads a permutation file of any size, so long as the format in /docs/formats/permutationFiles.md is followed
*/
int** readPermFile(string filename)
{
	int** perms;
	
	fstream fin;
	fin.open(filename.c_str());
	if (!fin.is_open() ) 
	{
		cout << "Unable to open permutations file \"" << filename << "\" exiting..." << endl;
		exit(0); 
	}

	int permSize;
	fin >> permSize;
	int totalPerms = factorial(permSize);

	perms = new int*[totalPerms];
	for(int i = 0; i < totalPerms; i++)
		perms[i] = new int[permSize];

	int value;
	for(int i = 0; i < totalPerms; i++) 
	{
		for(int j = 0; j < permSize; j++)
		{
			fin >> value;
			perms[i][j] = value;
		}
	}

	fin.close();
	return perms;
}

/*
	Finds the factorial
*/
int factorial(int n)
{
	int fact = 1;
	for (int i = n; i > 0; i--)
		fact *= i;
	return fact;
}
