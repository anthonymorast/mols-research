#include <iostream>
#include <string>
#include <stdio.h>

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
    return false;
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
