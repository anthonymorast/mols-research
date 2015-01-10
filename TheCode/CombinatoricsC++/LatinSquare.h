#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <algorithm>

using namespace std;

class LatinSquare 
{
    public:
	LatinSquare (int order);
	LatinSquare (int order, int values[]);
	void Fill  (int values[]);
	int GetOrder ();
	bool IsOrthogonal (LatinSquare checkSq);
	bool IsSameIsotopyClass (LatinSquare checkSq);
	bool IsSameMainClass (LatinSquare checkSq);
	int GetElementAtPosition (int row, int column);
	LatinSquare PermuteRows (int newIndices[]);
	bool IsNormal ();

    protected:
	int GetElement (int row, int col);
	bool CheckValues (int valueList[], string error);
	void SetValues (int valueList[]);
	 
    private:
	string to_string(int value);
        bool Distinct (int rowOrCol[]);
	int squareOrder;
	int *values;	
};

LatinSquare::LatinSquare (int order)
{
    squareOrder = order;
}

LatinSquare::LatinSquare (int order, int valueList[])
{
    squareOrder = order;
    SetValues(valueList);
}

void LatinSquare::Fill (int valueList[])
{
    SetValues(valueList);
}

int LatinSquare::GetOrder()
{
    return squareOrder;
}

bool LatinSquare::IsOrthogonal (LatinSquare checksq)
{
    return false;
}

bool LatinSquare::IsSameIsotopyClass (LatinSquare checkSq)
{
    return false;
}

bool LatinSquare::IsSameMainClass (LatinSquare checkSq)
{
    return false;
}

int LatinSquare::GetElementAtPosition (int row, int column)
{    
    return GetElement(row, column);
}

LatinSquare LatinSquare::PermuteRows (int newIndicies[])
{
    LatinSquare newSq(0);

    return newSq;
}

bool LatinSquare::IsNormal()
{
    return false;
}

int LatinSquare::GetElement (int row, int col)
{
    if (row > squareOrder || col > squareOrder)
    {
	printf("Index %d,%d not in square of size %d,%d", row, col, squareOrder, squareOrder);
	throw new exception;
    }

    return 0;
}

bool LatinSquare::CheckValues (int valueList[], string error)
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

void LatinSquare::SetValues (int valueList[])
{
    int size = (sizeof(valueList)/sizeof(*valueList));
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
