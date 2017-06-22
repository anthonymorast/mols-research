/* Libraries */
#include <iostream>
#include <string>  
#include <fstream> 	// file I/O

/* Custom Classes and Libraries */
#include "../LatinSquare.h" // Latin square class

/***** IMPORTANT NOTES!!!!!!!!!!!!! ******/
// I am being lazy. This needs to be changed for different orders
// The purpose of this quick script is to fix the McKay format 
// where there are not spaces between the rows of the Latin Squares.
// There is also no square order in the file so we need to set it
// manually. 
//
// We also need to change th input/output filenames for different files.
//
// The output is a txt file in the format specified in the docs.
int SQUARE_ORDER = 7;

void fixNoSpaceFormat();

int main (int argc, char*argv[]) 
{
	fixNoSpaceFormat();
	return 0;
}

void fixNoSpaceFormat() 
{
	vector<LatinSquare> squares;

	ifstream fin;
	ofstream fout;

	fin.open("latin_is7.txt");
	fout.open("docFormat.txt");
	if (!fin.is_open() || !fout.is_open()) 
	{
		cout << "Unable to open normalized squares file exiting..." << endl;
		exit(0); 
	}
	fout << SQUARE_ORDER << endl;

	int orderSquared = SQUARE_ORDER * SQUARE_ORDER;	
	string value;
	while (fin >> value) 
	{
		const char* cString = value.c_str();
		for(int i = 0; i < orderSquared; i++)
		{
			if(i > 0 && i % SQUARE_ORDER == 0)
				fout << " ";
			fout << cString[i];
		}
		fout << endl;
	}

	fin.close();
	fout.close();
}
