#include "LatinSquare.h"
#include <fstream>

vector<LatinSquare> ReadInSquares (string filename);
vector< vector<int> > ReadInPermutations (string filename);
vector<LatinSquare> GenerateReduced ();

void FindOrthogonalOrder6 (vector<LatinSquare> reducedSquares);


int main (int argc, char* argv[])
{   
    vector<LatinSquare> reducedSquares = GenerateReduced();
    cout << reducedSquares.size() << endl;

    FindOrthogonalOrder6 (reducedSquares);

    return 0;
}

vector<LatinSquare> GenerateReduced()
{
    vector<LatinSquare> reduced;

    vector<LatinSquare> squares = ReadInSquares("order6norm.txt");
    vector< vector<int> > permutations = ReadInPermutations("5_perm.txt");

    for (int i = 0; i < squares.size(); i++)
    {
	LatinSquare current = squares[i];
	for (int j = 0; j < permutations.size(); j++)
	    reduced.push_back(current.PermuteRows(permutations[j]));
    }

    return reduced;
}

vector<LatinSquare> ReadInSquares (string filename)
{
    vector<LatinSquare> squares;

    ifstream fin;
    fin.open(filename.c_str());
    if (!fin)
    {
	cout << "Failed to open file " << filename << endl;
	throw new exception;
    }

    long read;
    int count = 0;
    vector<int> values;
    while (fin >> read)
    {
	if (count == 6)
	{
	    count = 0;
	    LatinSquare sq(6, values);
	    squares.push_back(sq);
	}
	count++;
    }

    fin.close();
    return squares;
}

vector< vector<int> > ReadInPermutations (string filename)
{
    vector< vector<int> > permutations;

    ifstream fin (filename.c_str());
    if (!fin)
    {
	cout << "Failed to open file " << filename << endl;
	throw new exception;
    }



    fin.close();
    return permutations;
}

void FindOrthogonalOrder6 (vector<LatinSquare> reducedSquares)
{

}
