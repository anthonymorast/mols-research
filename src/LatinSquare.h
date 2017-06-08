#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <algorithm>
#include <vector>

using namespace std;

class LatinSquare 
{
    public:
    	LatinSquare (int order);
    	LatinSquare (int order, vector<int> values);
    	void Fill  (vector<int> values);
    	int GetOrder ();
    	bool IsOrthogonal (LatinSquare checkSq);
    	bool IsSameIsotopyClass (LatinSquare checkSq);
    	bool IsSameMainClass (LatinSquare checkSq);
    	int GetElementAtPosition (int row, int column);
    	LatinSquare PermuteRows (vector<int> newIndices);
		LatinSquare PermuteColumns (vector<int> newIndices);
		LatinSquare Normalize();
		bool IsSymmetric();
		bool IsEqual(LatinSquare compSq);
    	bool IsNormalized();
    	void Print();
        string ToString();

    protected:
    	int GetElement (int row, int col);
    	bool CheckValues (vector<int> valueList, string &error);
    	void SetValues (vector<int> valueList);
	 
    private:
    	string to_string(int value);
        bool Distinct (int rowOrCol[]);
    	int squareOrder;
    	vector<int> values;	
};

LatinSquare::LatinSquare (int order)
{
    squareOrder = order;
}

LatinSquare::LatinSquare (int order, vector<int> valueList)
{
    squareOrder = order;
    SetValues(valueList);
}

/*
    Sets the values of a Latin square to the values in valueList.
*/
void LatinSquare::Fill (vector<int> valueList)
{
    SetValues(valueList);
}

/*
    Returns the order of the Latin square.
*/
int LatinSquare::GetOrder()
{
    return squareOrder;
}

/*
    Determines if two Latin squares are Mutually orthogonal.
*/
bool LatinSquare::IsOrthogonal (LatinSquare checksq)
{
    vector<int> currentPair;
    vector< vector<int> > pairs;

    currentPair.push_back(0);
    currentPair.push_back(0);

    for (int i = 0; i < squareOrder; i++)
    {
		for (int j = 0; j < squareOrder; j++)
		{
	    	currentPair[0] = GetElementAtPosition(i+1, j+1);
	    	currentPair[1] = checksq.GetElementAtPosition(i+1, j+1);
 
 	    	if (!pairs.empty())
	    	{
				if (find(pairs.begin(), pairs.end(), currentPair) != pairs.end())
		    		return false;
				else 
		    		pairs.push_back(currentPair);
	    	}
	    	else 
				pairs.push_back(currentPair);
		}
    } 

    return true;
}

/*
    Determines if two Latin square were generated from the same Isotopy class.
*/
bool LatinSquare::IsSameIsotopyClass (LatinSquare checkSq)
{
    printf("IsSameIsotopyClass not yet implemented.");
    throw new exception;

    return false;
}

/*
    Determines if two squares came from the same main class.
*/
bool LatinSquare::IsSameMainClass (LatinSquare checkSq)
{
    printf ("IsSameIsotopyClass not yet implemented.");
    throw new exception;

    return false;
}

/*
    Returns the element at position row, col.
*/
int LatinSquare::GetElementAtPosition (int row, int column)
{    
    return GetElement(row, column);
}

/*
    Permutes (swaps) the rows of the Latin square based on the vector of new 
    positions. The newIndices vector should have the numbers 1 thru n where n
    is the order of the Latin square. The rows are moved in accordance to this vector, 
    e.g. row 1 in a square will be moved to row newIndices[0], row 2 will be moved to 
    rowInices[1], etc.
*/
LatinSquare LatinSquare::PermuteRows (vector<int> newIndices)
{
    if (newIndices.size() != squareOrder)
    {
		cout << "Not enough indices to swap in PermuteRows method." << endl;
		throw new exception;
    }
    
    vector< vector<int> > oldRows;
    vector< vector<int> > newRows;
    
    for (int i = 0; i < squareOrder; i++)
    {
    	vector<int> row;
    	for (int j = 0; j < squareOrder; j++)
    	    row.push_back(GetElementAtPosition(i+1, j+1));
    	
    	oldRows.push_back(row);
    }

    vector<int> newVals;
    for (int i = 0; i < squareOrder; i++)
    {
    	if (newIndices[i] > squareOrder || newIndices[i] < 1)
	    {
	       cout << "Invalid index in new indices list in PermuteRows method." << endl;
	       throw new exception;
        }

    	newRows.push_back(oldRows[newIndices[i] - 1]);
    	vector<int> current = newRows[i];
    	for (int j = 0; j < newRows[i].size(); j++)
    	    newVals.push_back(current[j]);
    }
   
    LatinSquare newSq (squareOrder, newVals);

    return newSq;
}

/*
    Permutes the columns of the Latin square in a similar fashon as is done in PermuteRows().
*/
LatinSquare LatinSquare::PermuteColumns (vector<int> newIndices) 
{
    if (newIndices.size() != squareOrder) 
    {
		cout << "Not enough indices to swap in PermuteColumns method.\nIndices supplied: ";
		cout << newIndices.size() << "\nIndices Required: " << squareOrder << endl;
		throw new exception;
    }

    vector<int> oldValues;
    vector<int> newValues(squareOrder*squareOrder);

    for (int i = 0; i < squareOrder; i++) 
    {
		for (int j = 0; j < squareOrder; j++) 
		{
	    	oldValues.push_back(GetElementAtPosition(i+1, j+1));
		}
    }

    for (int i = 0; i < squareOrder; i++) 
    {
		// The columns of the square are every nth element in an array of the values. Here we are using
		// the newIndices values as an offset to add onto the index to get to the correct column within
		// a set of n values. So if newIndices[i] = 2 then we will look at every (n+2)+(squareOrder*i) 
		// values. It is confusing, just write it down.
		for (int j = 0; j < squareOrder; j++) 	
		{
			newValues.at(i + (squareOrder * j)) = oldValues[(newIndices[i]-1) + (j * squareOrder)];
		}
    }

	LatinSquare newSq(squareOrder, newValues);
	return newSq;
}

/*
	Permutes the rows and columns of the square to normalize it.
*/
LatinSquare LatinSquare::Normalize() 
{
	vector<int> rowIndices;
	vector<int> colIndices;

	for (int i = 1; i <= squareOrder; i++) 
	{
		for (int j = 1; j <= squareOrder; j++) 
		{
			if (GetElementAtPosition(j, 1) == i) 
			{
				rowIndices.push_back(j);
				break;
			}
		}
	}

	*this = PermuteRows(rowIndices);

	for (int i = 1; i <= squareOrder; i++) 
	{
		for (int j = 1; j <= squareOrder; j++) 
		{
			if (GetElementAtPosition(1, j) == i)
			{
				colIndices.push_back(j);
				break;
			}
		}
	}

	*this = PermuteColumns(colIndices);

	return *this;
}

/*
	Determines if a Latin square is symmetric (rows = columns).
*/
bool LatinSquare::IsSymmetric() 
{
	vector<int> rowValues = values;
	vector<int> colValues;

	for (int i = 0; i < squareOrder; i++)
	{
		for (int j = 0; j < squareOrder; j++)
		{
			colValues.push_back(rowValues.at(i + (squareOrder * j)));
		}
	}

	if (rowValues == colValues) 
	{
		return true;
	}

	return false;
}

/*
	Determines if two Latin squares are equal.
*/
bool LatinSquare::IsEqual(LatinSquare compSquare) 
{
	for (int i = 0; i < squareOrder; i++) 
	{
		for (int j = 0; j < squareOrder; j++)
		{
			if (GetElementAtPosition(i+1, j+1) != compSquare.GetElementAtPosition(i+1, j+1))
			{
				return false;
			}
		}
	}
	return true;
}


/*
    Determines if the Latin square is normalized, i.e. the first row and column count up from
    1 to n.
*/
bool LatinSquare::IsNormalized()
{
	vector<int> testSqValues;
	for (int i = 0; i < squareOrder; i++)
	{
		for (int j = 0; j < squareOrder; j++)
		{
			testSqValues.push_back(GetElementAtPosition(i+1, j+1));
		}
	}
	LatinSquare testSq(squareOrder, testSqValues);
	if (IsEqual(testSq.Normalize())) 
	{
		return true;
	}

    return false;
}

/*
    Returns a particular element in a Latin square at position row, col.
*/
int LatinSquare::GetElement (int row, int col)
{
    if (row > squareOrder || col > squareOrder)
    {
		printf("Index %d,%d not in square of size %d,%d", row, col, squareOrder, squareOrder);
		throw new exception;
    }

    row--;
    col--;
    return values[(row*squareOrder) + col];
}

/*
    Determines if the list of values passed as the first parameter is of the right format to 
    create a Latin square. 
*/
bool LatinSquare::CheckValues (vector<int> valueList, string &error)
{
    int **rows = new int *[squareOrder];
    int **cols = new int *[squareOrder];    
    
    for (int i = 0; i < squareOrder; i++)
    {
		rows[i] = new int [squareOrder];
		cols[i] = new int [squareOrder];
    }
    
    for (int i = 0; i < squareOrder; i++)
    {
		for (int j = 0; j < squareOrder; j++)
		{
	    	int currentElement = valueList[(i*squareOrder) + j];
	    	if (currentElement > squareOrder || currentElement < 1)
	    	{
			error = "Element " + to_string(currentElement) + " in row " +
				to_string(i+1) + " and column " + to_string(j+1) + 
				" is outside the valid range of values. Latin squares should contain " +
				" elements between 1 and the order of the square.";
			return false;
	    	}
	    
	    	rows[i][j] = currentElement;
	    	cols[j][i] = currentElement;
		}	
    } 

    for (int i = 0; i < squareOrder; i++)
    {
		if (!Distinct(rows[i]))
		{
	    	error = "Row " + to_string(i+1) + " does not contain distinct elements.";
	    	return false;
		} 

        if (!Distinct(cols[i]))
		{
	    	error = "Col " + to_string(i+1) + " does not contain distinct elements.";
	    	return false;
        }
    }

    error = "";
    return true;
}

/*
    Sets the values of the Latin square. The value list must be of size order^2. The first n elements
    are used for the first row, the second n elements from the second row, etc.
*/
void LatinSquare::SetValues (vector<int> valueList)
{
    int size = valueList.size();
    string error = "";

    if (size != (squareOrder*squareOrder))
    {
		cout << "Incorrect number of values to fill Latin Square. Size: " << size << " Order: " << squareOrder << "."  << endl;
		throw new exception;
    }
    else if (CheckValues(valueList, error))
    {
		values = valueList;
    }
    else
    {
        cout << error << endl;
		throw new exception;
    }
}

string LatinSquare::to_string (int value)
{
    ostringstream oss;
    oss << value;
    return oss.str();    
}

bool LatinSquare::Distinct (int rowOrCol[])
{
    int size = (sizeof(rowOrCol)/sizeof(*rowOrCol));
    sort(rowOrCol, rowOrCol + size);

    for (int i = 0; i < size; i++)
    {
		if (rowOrCol[i] == rowOrCol[i+1])
	    	return false;
    }

    return true;
}

void LatinSquare::Print()
{
    cout << ToString() << endl;
}

string LatinSquare::ToString() 
{
    string printString = "";    

    if (values.size() > 0) 
    {
        for (int i = 0; i < squareOrder*squareOrder; i++)
        {
            if (i % squareOrder == 0 && i != 0)
                printString += "\n";

            printString += to_string(values[i]) + " ";
        }
    }
    else 
    {
        cout << "An empty Latin square of order " << squareOrder << "." << endl;
    }

    return printString;
}
