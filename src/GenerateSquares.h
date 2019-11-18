#include "../LatinSquare/LatinSquare.h"
#include "utils/Utils.h"

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

/// NOTE:
// We need to pass in checksqs here and only push back the squares if they
// are not in the allSqs array, that is, if they haven't been checked yet.
// Otherwise, after we have enough squares all sqs are 'new' and thus added
// to the checksqs vector. This is a better way to determine which squares
// haven't actually been checked yet.
void unique_add_to_vector(LatinSquare sq, vector<LatinSquare> &squares,
    vector<LatinSquare> &checkSqs, bool updateCheckSquares)
{
	if(find(squares.begin(), squares.end(), sq) == squares.end())
	{
		squares.push_back(sq);
		if(updateCheckSquares)
		{
			checkSqs.push_back(sq);
		}
	}
}

short* get_array_from_line(string line, int size)
{
	line.erase(remove(line.begin(), line.end(), ' '), line.end());
	short *vals = new short[size];
	const char* linearr = line.c_str();
	for(int i = 0; i < size; i++)
		vals[i] = linearr[i] - '0';
	return vals;
}

void print_usage()
{
	cout << "Usage:" << endl;
	cout << "\tgenerate_sqaures <order> <iso_reps filename>" << endl;
}
