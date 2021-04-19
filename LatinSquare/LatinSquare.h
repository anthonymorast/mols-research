#pragma once

#include <string>
#include <cstring> 		// memset
#include <iostream>
#include <fstream>
#include "InvalidSquareException.h"

using namespace std;

class LatinSquare
{
public:
	LatinSquare(short order);					// kind of useless constructor
	LatinSquare(short order, short* values);
	LatinSquare(short order, short* values, short iso_class);
	LatinSquare(short order, short* values, short iso_class, short main_class);
	LatinSquare(const LatinSquare& ls);			// copy constructor
	~LatinSquare();

	// change the square
	void set_iso_class(short iso_class) { this->iso_class = iso_class; };
	void set_main_class(short main_class) { this->main_class = main_class; };
	void set_values(short* sq_values);
	void permute_rows(short* new_rows);
	void permute_cols(short* new_cols);
	void permute_symbols(short* syms);
	void rcs_permutation(short* rcs);
	void normalize();
	void reduce();

	// square properties
	bool is_symmetric();
	bool is_orthogonal(LatinSquare sq);
	bool is_normal();
	bool is_valid();

	// visualization
	string tostring();
	void print();
	friend ostream& operator<<(ostream& os, const LatinSquare sq);
	void output_values_space(ofstream& os);
	void print_flat();
	string flatstring();
	string flatstring_no_space();
	short* get_values() {return values;}

	// operators
	bool operator==(const LatinSquare &chk_sq) const;
	bool operator!=(const LatinSquare &chk_sq) const;
	// implemented so we std::set can be use which requires unique data be stored
	// this makes creating the collection of all squares MUCH faster
	bool operator<(const LatinSquare &chk_sq) const;

private:
	short order = -1;		// technically = i due to o_sq (order squared) = -1
	short* values = NULL;
	short iso_class = -1;
	short o_sq = -1;
	short main_class = -1;

	void move_row(short curr_row, short new_row, short *new_values);
	void move_col(short curr_col, short new_col, short *new_values);
	InvalidSquareException invalidExcept;
};
