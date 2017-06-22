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

    cout << squares.size() << " "  << permutations.size() << endl;

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
    vector<int> current;
    while (fin >> read)
    {

	if (count == 6)
	{
	    count = 0;
	    LatinSquare sq(6, current);
	    squares.push_back(sq);
	    current.clear();
	}

	if (count == 0)
	{
	    current.push_back(1);
	    current.push_back((int)((read / 10000)) + 1);
	    current.push_back((int)((read % 10000) / 1000) + 1);
	    current.push_back((int)((read % 1000) / 100) + 1);
	    current.push_back((int)((read % 100) / 10) + 1);
	    current.push_back((int)((read % 10)) + 1);
	}
	else 
	{
	    current.push_back((int)(read / 100000) + 1);
	    current.push_back((int)((read % 100000) / 10000) + 1);
	    current.push_back((int)((read % 10000) / 1000) + 1);
	    current.push_back((int)((read % 1000) / 100) + 1);
	    current.push_back((int)((read % 100) / 10) + 1);
	    current.push_back((int)((read % 10)) + 1);
	}
	count++;
    }
    
    if (current.size() > 0)
    {
	LatinSquare sq (6, current);
	squares.push_back(sq);
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

    int read;
    int count = 0;
    vector<int> permutation;
    // since we only permute the last 5 rows of the square we need to add 1 at the beginning of each permutation to keep the first row
    // as the first row. Otherwise, only 5 values will be in the vector, which isnt enough to permute the square.  
    permutation.push_back(1);
    fin >> read;	// the top of the 5_perm.txt has '5' which is the number of items being permuted, dont wann this in our vector.
    while (fin >> read)
    {
	if (count == 5)
	{
	    count = 0;
 	    permutations.push_back(permutation);
	    permutation.clear();
	    permutation.push_back(1);
	}

        permutation.push_back(read + 1);	
	count++;
    }
    if (permutation.size() > 0)
	permutations.push_back(permutation);

    fin.close();
    return permutations;
}

void FindOrthogonalOrder6 (vector<LatinSquare> reducedSquares)
{

}
