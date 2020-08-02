#include "../LatinSquare/LatinSquare.h"
#include "utils/Utils.h"

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <unordered_map>
#include <unordered_set>

short* get_array_from_line(string line, int size)
{
	line.erase(remove(line.begin(), line.end(), ' '), line.end());
	short *vals = new short[size];
	const char* linearr = line.c_str();
	for(int i = 0; i < size; i++)
		vals[i] = linearr[i] - '0';
	return vals;
}

long my_factorial(long n)
{
    long product = 1;
    for(int i = 2; i <= n; i++)
    {
        product *= i;
    }
    return product;
}
