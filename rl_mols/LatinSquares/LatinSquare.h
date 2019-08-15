#pragma once

#include <string>
#include <cstring> 		// memset
#include <iostream>
#include <fstream>
#include "InvalidSquareException.h"

using namespace std;

enum { ROW = 0, COL = 1, SYM = 2 };

class LatinSquare
{
public:
	LatinSquare(short order);					// kind of useless constructor
	LatinSquare(short order, short* values);
	LatinSquare(short order, short* values, short iso_class);
	LatinSquare(short order, short* values, short iso_class, short main_class);
	LatinSquare(const LatinSquare& ls);			// copy constructor

	// change the square
	void set_iso_class(short iso_class);
	void set_main_class(short main_class);
	void set_values(short* sq_values);
	void permute_rows(short* new_rows);
	void rcs_permutation(short* rcs);
	void sym_permutation(short* syms);

	// square properties
	bool is_symmetric();
	bool is_orthogonal(LatinSquare sq);
	bool is_valid();

	// visualization
	string tostring();
	void print();
	friend ostream& operator<<(ostream& os, const LatinSquare sq);
	void output_values_space(ofstream& os);
	void print_flat();
	string flatstring();

	// operators
	const bool operator==(const LatinSquare &chk_sq) const;
	const bool operator!=(const LatinSquare &chk_sq) const;

private:
	short order = -1;		// technically = i due to o_sq = -1
	short* values = NULL;
	short iso_class = -1;
	short o_sq = -1;
	short main_class = -1;

	void move_row(short curr_row, short new_row, short *new_values);
	InvalidSquareException invalidExcept;
};
