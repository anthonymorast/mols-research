#include "LatinSquare.h"

// delegated constructors not working with gcc/g++, even after specifying c++11
LatinSquare::LatinSquare(short order)
{
	this->order = order;
	this->o_sq = order * order;
}

LatinSquare::LatinSquare(short order, short* values)
{
	this->order = order;
	this->o_sq = order * order;
	this->values = values;
	if (values != NULL)
	{
		set_values(values);		// if initialized with actual values, verify they are valid
	}
}

LatinSquare::LatinSquare(short order, short* values, short iso_class)
{
	this->order = order;
	this->o_sq = order * order;
	this->values = values;
	if (values != NULL)
	{
		set_values(values);		// if initialized with actual values, verify they are valid
	}
	this->iso_class = iso_class;
}

LatinSquare::LatinSquare(short order, short* values, short iso_class, short main_class)
{
	this->order = order;
	this->o_sq = order * order;
	this->values = values;
	if (values != NULL)
	{
		set_values(values);		// if initialized with actual values, verify they are valid
	}
	this->iso_class = iso_class;
	this->main_class = main_class;
}

LatinSquare::LatinSquare(const LatinSquare& cls)
{
	order = cls.order;
	o_sq = cls.o_sq;
	values = NULL;
	if (cls.values != NULL)
	{
		// deep copy the array
		values = new short[o_sq];
		memcpy(values, cls.values, sizeof(short) * o_sq);
	}
	iso_class = -1;
	main_class = -1;
}

string LatinSquare::tostring()
{
	string lsstr = "";
	if (values != NULL)
	{
		for (int i = 0; i < o_sq; i++)
		{
			if (i % order == 0 && i > 0)
				lsstr += "\n";
			lsstr += to_string(values[i]) + " ";
		}
	}
	else
	{
		lsstr = "An empty Latin square of order " + to_string(order) + ".\n";
	}
	return lsstr;
}

string LatinSquare::flatstring()
{
	string flatstr = "";
	if(values != NULL)
	{
		for(short i = 0; i < o_sq; i++)
		{
			flatstr += to_string(values[i]) + " ";
		}
	}
	else
	{
		flatstr = "An empty Latin square of order " + to_string(order);
	}
	return flatstr + "\n";
}

void LatinSquare::print()
{
	cout << tostring() << endl;
}

void LatinSquare::print_flat()
{
	if(values != NULL)
	{
		for(short i = 0; i < o_sq; i++)
		{
			cout << values[i];
		}
	}
	cout << endl;
}

void LatinSquare::output_values_space(ofstream &os)
{
	for (short i = 0; i < o_sq; i++)
		os << values[i] << " ";
	os << endl;
}

ostream& operator<<(ostream& os, const LatinSquare sq)
{
	if (sq.values != NULL)
	{
		for (int i = 0; i < sq.o_sq; i++)
		{
			if (i % sq.order == 0 && i > 0)
				os << "\n";
			os << to_string(sq.values[i]) + " ";
		}
	}
	else
	{
		os << to_string(sq.order);
	}
	return os;
}

bool LatinSquare::is_symmetric()
{
	if (!is_valid())
	{
		throw invalidExcept;
		return false;
	}

	short* rows = new short[o_sq];
	short* cols = new short[o_sq];
	short j = -1;	// start at -1 to prevent i>0 check everytime in the loop

	for (short i = 0; i < o_sq; i++)
	{
		if (i % order == 0)
			j++;
		rows[i] = values[i];
		cols[i] = values[((i*order) % o_sq) + j];	// hardest math here
	}

	for (short i = 0; i < o_sq; i++)
	{
		if (rows[i] != cols[i])		// if they ever differ, not symmetric
		{
			delete[] rows;
			delete[] cols;
			return false;
		}
	}

	delete[] rows;
	delete[] cols;
	return true;
}

bool LatinSquare::is_orthogonal(LatinSquare chk_sq)
{
	if (!is_valid() || !chk_sq.is_valid())
	{
		throw invalidExcept;
		return false;
	}
	if (order != chk_sq.order)	// must be the same size
		return false;

	// mark off the pairs in this table, if the table is marked the pair has been used
	// and the squares are not orthogonal.
	bool** pairs = new bool*[order];
	for (short i = 0; i < order; i++)
		pairs[i] = new bool[order];

	for (short i = 0; i < order; i++)
	{
		for (short j = 0; j < order; j++)
		{
			pairs[i][j] = false;
		}
	}

	for (short i = 0; i < o_sq; i++)
	{
		short idx1 = values[i] % order;
		short idx2 = chk_sq.values[i] % order;
		if (pairs[idx1][idx2])
		{
			delete[] pairs;
			return false;
		}
		else
		{
			pairs[idx1][idx2] = true;
		}
	}

	delete[] pairs;
	return true;
}

bool LatinSquare::is_valid()
{
	if (values == NULL)
	{
		return false;
	}
	// mark the row/col check matrices, each index should be marked once for each
	// row and column for the square to be valid.
	bool* row_chk = new bool[order];
	bool* col_chk = new bool[order];
	short j = -1;

	for (short i = 0; i < o_sq; i++)
	{
		// the check arrays should be filled every time we do 'order' number of checks
		// need to reset them to all false
		if (i % order == 0)
		{
			memset(row_chk, 0, order * sizeof(bool));
			memset(col_chk, 0, order * sizeof(bool));
			j++;
		}
		short row_idx = values[i] % order;
		short col_idx = values[((i * order) % o_sq) + j] % order;

		if (col_chk[col_idx] || row_chk[row_idx])	// same number twice (already marked)
		{
			delete[] row_chk;
			delete[] col_chk;
			return false;
		}
		else
		{
			col_chk[col_idx] = true;
			row_chk[row_idx] = true;
		}
	}

	delete[] row_chk;
	delete[] col_chk;
	return true;
}

void LatinSquare::set_values(short* sq_values)
{
	this->values = sq_values;
	if (!is_valid())
	{
		cout << "WARNING: Invalid values for latin square..." << endl;
		//this->print();
	}
}

void LatinSquare::permute_rows(short* new_rows)
{
	if (!is_valid())
	{
		throw invalidExcept;
		return;
	}

	short* new_values = new short[o_sq];
	for (short i = 0; i < order; i++)
	{
		move_row(i, new_rows[i], new_values);
	}
	delete[] values;
	set_values(new_values);
}

void LatinSquare::permute_cols(short *new_cols)
{
	if(!is_valid())
	{
		throw invalidExcept;
		return;
	}
	short *new_values = new short[o_sq];
	for(short i = 0; i < order; i++)
	{
		move_col(i, new_cols[i], new_values);
	}
	delete[] values;
	set_values(new_values);
}

void LatinSquare::rcs_permutation(short* rcs)
{
	// I'm sorry future me...

	// should be able to grab the current (R, C, S) triple and
	// index those values based on the current rcs permutation (param).
	// If the symbol becomes the row, then rcs[0] will be 2. This 2 will
	// index triple[2] which gives the row to be used in the new_value
	// index calculation.
	short j = -1;
	short trip[3];
	short* new_vals = new short[o_sq];

	for (short i = 0; i < o_sq; i++)
	{
		if (i % order == 0)
			j++;
		trip[1] = i % order;							// row
		trip[0] = j;									// col
		trip[2] = values[(j * order) + (i % order)];	// curr value
		new_vals[(trip[rcs[0]] * order) + trip[rcs[1]]] = trip[rcs[2]];
	}
	delete[] values;
	set_values(new_vals);
}

void LatinSquare::permute_symbols(short* syms)
{
	// index syms based on current square values to get updated values
	short* new_vals = new short[o_sq];
	for (short i = 0; i < o_sq; i++)
	{
		new_vals[i] = syms[values[i]];
	}
	delete[] values;
	set_values(new_vals);
}

/**
* Sets the values in the new array at row (new_row) to the values in the current
* array at row (curr_row).
**/
void LatinSquare::move_row(short curr_row, short new_row, short* new_values)
{
	for (short i = 0; i < order; i++)
	{
		new_values[curr_row * order + i] = values[new_row * order + i];
	}
}

void LatinSquare::move_col(short curr_col, short new_col, short *new_values)
{
	for (short i = 0; i < order; i++)
	{
		new_values[i * order + curr_col] = values[i * order + new_col];
	}
}

bool LatinSquare::operator==(const LatinSquare &chk_sq) const
{
	if (values == NULL || chk_sq.values == NULL)	// are empty, invalid squares not equal? should we compare the sizes?
	{
		throw invalidExcept;
		return false;
	}

	if (chk_sq.order != order)			// same order?
		return false;
	for (short i = 0; i < o_sq; i++)	// same values?
		if (values[i] != chk_sq.values[i])
			return false;
	return true;
}

bool LatinSquare::operator!=(const LatinSquare &chk_sq) const
{
	return !(*this == chk_sq);
}
