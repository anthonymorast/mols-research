#include "LatinSquare.h"

void LatinSquare::init(short order, short* values)
{
	// use order-1 since squares start at 0 (0 1 2 3 sum=6 vs 1 2 3 4 sum=10)
	expected_sum = ((order - 1) * order) / 2;
	this->order = order;
	o_sq = order * order;
	this->values = values;
	if (values != NULL)
		set_values(values);
	this->stringrepr.resize(o_sq);
}

LatinSquare::LatinSquare(short order) { init(order, NULL); }

LatinSquare::LatinSquare(short order, short* values)
{
	init(order, values);
}

LatinSquare::LatinSquare(short order, short* values, short iso_class)
{
	init(order, values);
	this->iso_class = iso_class;
}

LatinSquare::LatinSquare(short order, short* values, short iso_class, short main_class)
{
	init(order, values);
	this->iso_class = iso_class;
	this->main_class = main_class;
}

LatinSquare::LatinSquare(const LatinSquare& cls)
{
	expected_sum = cls.expected_sum;
	order = cls.order;
	o_sq = cls.o_sq;
	values = NULL;
	if (cls.values != NULL)
	{
		values = new short[o_sq];
		memcpy(values, cls.values, sizeof(short) * o_sq);
	}
	iso_class = -1;
	main_class = -1;
}

LatinSquare::~LatinSquare(){}

string LatinSquare::tostring() const
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

string LatinSquare::flatstring_no_space()
{
	if(values == NULL)
	{
		return "An empty Latin square of order " + to_string(order);
	}

	// Ensure stringrepr is sized (copy constructor doesn't copy it, so it may be empty)
	if (order <= 10)
	{
		if (stringrepr.size() != static_cast<size_t>(o_sq))
			stringrepr.resize(o_sq);
	}
	else
	{
		stringrepr.clear();
		stringrepr.reserve(o_sq*2); // overestimate by assuming 2-digits per value in square
	}

	for(short i = 0; i < o_sq; i++)
	{
		if(order <= 10) { // can go up to 10 due to 0-based indexing
			this->stringrepr[i] = '0' + values[i]; // more efficient if value is <= 9
			continue;
		}

		this->stringrepr += std::to_string(values[i]);
	}

	return this->stringrepr;
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

// probably doens't work
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

bool LatinSquare::is_normal()
{
	short* rows = new short[order];
	short* cols = new short[order];

	for (short i = 0; i < order; i++)
	{
		cols[i] = values[i];
		rows[i] = values[(i*order)];	// hardest math here
	}

	for (short i = 0; i < order; i++)
	{
		if (rows[i] != i || cols[i] != i)		// if they ever differ, not symmetric
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
	
	short row_sum {}, col_sum {}, j {};
	for (short i = 0; i < o_sq; i++)
	{
		// the check arrays should be filled every time we do 'order' number of checks
		// need to reset them to all false
		if (i % order == 0 && i > 0)
		{
			if (row_sum != this->expected_sum || col_sum != this->expected_sum)
			{
				cout << "Row sum: " << row_sum << ", Col sum: " << col_sum << endl;
				cout << "Expected sum: " << this->expected_sum << endl;
				cout << "Values: ";
				for (short k = 0; k < o_sq; k++)
				{
					cout << values[k] << " ";
				}
				cout << endl;
				return false;
			}
			row_sum = 0;
			col_sum = 0;
			j++;
		}
		row_sum += values[i];
		col_sum += values[((i * order) % o_sq) + j];
	}

	return true;
}

void LatinSquare::set_values(short* sq_values)
{
	this->values = sq_values;
	if (!is_valid())
	{
		cout << "WARNING: Invalid values for latin square..." << endl;
		this->print();
		exit(0);
	}
}

void LatinSquare::permute_rows(short* new_rows)
{
	if (!is_valid())
	{
		throw invalidExcept;
		return;
	}

	short temp_values[o_sq]; // create on stack to hold old values
	memcpy(temp_values, values, sizeof(short) * o_sq); // copy old values

	// modify values in-place rather than allocate, reset and delete
	for (short i = 0; i < order; i++)
	{
		for (short j = 0; j < order; j++) {
			values[i * order + j] = temp_values[new_rows[i] * order + j];
		}
	}
}

void LatinSquare::permute_cols(short *new_cols)
{
	if(!is_valid())
	{
		throw invalidExcept;
		return;
	}

	short temp_values[o_sq]; // create on stack to hold old values
	memcpy(temp_values, values, sizeof(short) * o_sq); // copy old values

	// edit in place rather than allocate, reset and delete
	for(short i = 0; i < order; i++) {
		for(auto j = 0; j < order; j++) { // move_cols logic inline here
			values[j * order + i] = temp_values[j * order + new_cols[i]];
		}
	}
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

void LatinSquare::normalize()
{
	short* perm = new short[order];
	for(int i = 0; i < order; i++)
	{
		perm[values[i]] = i;
	}
	permute_cols(perm);

	for(int i = 0; i < order; i++)
	{
		perm[values[i*order]] = i;
	}
	permute_rows(perm);
	delete[] perm;
}

void LatinSquare::reduce()
{
	short perm[order];
	for(int i = 0; i < order; i++)
	{
		perm[values[i]] = i;
	}
	permute_cols(perm);
}

void LatinSquare::permute_symbols(short* syms)
{
	// index syms based on current square values to get updated values
	short temp_values[o_sq];
	memcpy(temp_values, values, sizeof(short) * o_sq);
	for (short i = 0; i < o_sq; i++) {
		values[i] = syms[temp_values[i]];
	}
}

// NOTE (AM 3/1/26): These are not used but are more efficient than the ones above. To save
// allocations, static method operating on the same data were added to the GenerateSquares class.

// Duplicates permute_rows logic; writes directly from src to avoid extra copy (faster than copy+permute_rows).
// Caller must ensure src is valid.
void LatinSquare::permute_rows_from(const LatinSquare& src, short* new_rows)
{
	if (values != NULL) delete[] values;
	order = src.order;
	o_sq = src.o_sq;
	expected_sum = src.expected_sum;
	values = new short[o_sq];
	for (short i = 0; i < order; i++)
		for (short j = 0; j < order; j++)
			values[i * order + j] = src.values[new_rows[i] * order + j];
}

// Duplicates permute_cols logic; writes directly from src to avoid extra copy (faster than copy+permute_cols).
// Caller must ensure src is valid.
void LatinSquare::permute_cols_from(const LatinSquare& src, short* new_cols)
{
	if (values != NULL) delete[] values;
	order = src.order;
	o_sq = src.o_sq;
	expected_sum = src.expected_sum;
	values = new short[o_sq];
	for (short i = 0; i < order; i++)
		for (short j = 0; j < order; j++)
			values[j * order + i] = src.values[j * order + new_cols[i]];
}

// Duplicates permute_symbols logic; writes directly from src to avoid extra copy (faster than copy+permute_symbols).
// Caller must ensure src is valid.
void LatinSquare::permute_symbols_from(const LatinSquare& src, short* syms)
{
	if (values != NULL) delete[] values;
	order = src.order;
	o_sq = src.o_sq;
	expected_sum = src.expected_sum;
	values = new short[o_sq];
	for (short i = 0; i < o_sq; i++)
		values[i] = syms[src.values[i]];
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

bool LatinSquare::operator<(const LatinSquare &chk_sq) const
{
	// use the string representation of the squares to determine if two squares
	// are equal (essentially). A LS being less than another LS doesn't really
	// make sense so this is my way around that to use std::set objects in dealing
	// with squares.
	string thisStr = "";
	if(values != NULL)
	{
		for(short i = 0; i < o_sq; i++)
		{
			thisStr += to_string(values[i]);
		}
	}
	else
	{
		thisStr = "An empty Latin square of order " + to_string(order);
	}

	string checkStr = "";
	if(values != NULL)
	{
		for(short i = 0; i < o_sq; i++)
		{
			checkStr += to_string(chk_sq.values[i]);
		}
	}
	else
	{
		checkStr = "An empty Latin square of order " + to_string(order);
	}

	return thisStr < checkStr;
}
