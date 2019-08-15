#pragma once
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;

// creates a file with all permutations of values 0...n using Heap's Algorithm
void print_menu();
void permutation_menu();
void permutations(short n); 
void create_permutations_file(short *vals, short size, short n, ofstream &out);
void print_arr(short *vals, short size, ofstream &out);
void convert_mckay_menu();
void convert_mckay(string filename);
bool file_exists(string filename);
int factorial(short n);

// these are shared, keep the implementation in the header file
bool file_exists(string filename) 
{
	ifstream f(filename.c_str());
	return f.good();
}

int factorial(short n) 
{
	int prod = 1;
	for(short i = 1; i < n; i++) 
	{
		prod *= i;
	}
	return prod;
}
