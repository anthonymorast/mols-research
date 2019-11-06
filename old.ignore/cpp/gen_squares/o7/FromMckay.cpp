/* Libraries */
#include <iostream>
#include <vector>
#include <string>  
#include <fstream> 	// file I/O
#include <sstream> 	// string to int
#include <stdlib.h> 	// atol 
#include <pthread.h>	// multithreading
#include <ctype.h> 	// isDigit - valid args
#include <stdio.h> 	// printf
#include <time.h> 	// program timing - clock
#include <dirent.h> 	// directory operations
#include <sys/stat.h> 	// mkdir

/* Custom Classes and Libraries */
#include "../../LatinSquare.h" // Latin square class

int SQUARE_ORDER = 0;

void writeNewSquareFile(vector<LatinSquare> squares);
vector<LatinSquare> read_normalized_squares();
long string_to_formatted_long(string s);

int main (int argc, char*argv[]) 
{
	vector<LatinSquare> squares = read_normalized_squares();
	writeNewSquareFile(squares);
	return 0;
}

/*
	Writes the converted squares to a new file
*/
void writeNewSquareFile(vector<LatinSquare> squares)
{
	ofstream fout;
	fout.open("new_ls.dat");
	if(!fout.is_open())
	{
		cout << "Unable to open output file, exiting..." << endl;
		exit(0);
	}

	fout << SQUARE_ORDER << endl;
	for (vector<LatinSquare>::iterator it = squares.begin(); it != squares.end(); it++) 
	{
		LatinSquare sq = *it;
		int* values = sq.ToArray();
		for(int i = 0; i < SQUARE_ORDER*SQUARE_ORDER; i++)
			fout << values[i] << " ";
		fout << endl;
	}	
	fout.close();
}

/*
* read_normalized_squares
* 
* reads in all normalized Latin squares from the normalizedLatinSquares file
*
* @param filename - name of the file containing the normalized Latin squares
* @return a vector of normalized Latin squares
*/
vector<LatinSquare> read_normalized_squares() 
{
	vector<LatinSquare> squares;

	ifstream fin;
	fin.open("docFormat_o7.txt");
	if (!fin.is_open() ) 
	{
		cout << "Unable to open normalized squares file exiting..." << endl;
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
