/* Libraries */
#include <iostream>
#include <vector>
#include <string>  
#include <fstream> // file I/O
#include <sstream> // string to int

/* Custom Classes and Libraries */
#include "LatinSquare.h" // Latin square class

using namespace std;

/* Globals */

/* Functions */
bool valid_args(int arg_count, char* args);
void print_usage();
vector<LatinSquare> generate_reduced_squares(vector<vector<int>> permutations, vector<LatinSquare> normalizedSquares);
vector<vector<int>> read_permutations(string filename);
vector<LatinSquare> read_normalized_squares(string filename);
void find_orthogonal_squares(vector<LatinSquare> reducedSquares, string filename, int numThreads);

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
  if (!valid_args(argc, argv[])) 
  {
    return 0;
  }
  
  int numThreads;
  stringstream(argv[3]) >> numThreads;
  
  string normalizedFileName(argv[0]);
  string permutationFile(argv[1]);
  string outputFile(argv[2]);
  
  vector<LatinSquare> normalizedSqaures = read_normalized_squares(normalizedFileName);
  vector<vector<int>> permutations = read_permutations(permutationFile);
  vector<LatinSquare> reducedSquares = generate_reduced_squares(permutations, normalizedSquares);
  
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
vector<LatinSquare> generate_reduced_squares(vector<vector<int>> permutations, vector<LatinSquare> normalizedSquares)
{
  
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
vector<vector<int>> read_permutations(string filename) 
{
  
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
bool valid_args (int arg_count, char* args) 
{
  if (arg_count != 4) 
  {
    cout << "Incorrect number of command line arguments." << endl;
    print_usage();
    return false;
  }
  
  
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
