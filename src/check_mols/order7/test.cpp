#include <vector>
#include <iostream>
#include "LatinSquare.h"

using namespace std;

void print(vector<int> vec);

int main(int argc, char*argv[]) 
{
	vector<int> test;
	test.push_back(1);
	test.push_back(2);
	test.push_back(3);
	test.push_back(4);
	
	test.push_back(2);
	test.push_back(1);
	test.push_back(4);
	test.push_back(3);
	
	test.push_back(3);
	test.push_back(4);
	test.push_back(2);
	test.push_back(1);
	
	test.push_back(4);
	test.push_back(3);
	test.push_back(1);
	test.push_back(2);
	
	vector<int> newIndices;
	newIndices.push_back(3);
	newIndices.push_back(1);
	newIndices.push_back(4);
	newIndices.push_back(2);

	LatinSquare og(4, test);
	LatinSquare nw = og.PermuteColumns(newIndices);

	cout << "nw before IsNormal() call" << endl;
	nw.Print();
	cout << endl;
	cout << "nw is normal: " << nw.IsNormalized() << endl;
	cout << "nw is symmetric: " << nw.IsSymmetric() << endl;
	cout << endl;
	LatinSquare ns = nw.Normalize();
	cout << "ns = nw.Normalize();" << endl;
	ns.Print();
	cout << "ns isnormal = " << ns.IsNormalized() << endl;
	cout << "ns is symmetric: " << ns.IsSymmetric() << endl;

	return 0;
}

void print(vector<int> vec) 
{
	for (vector<int>::iterator it = vec.begin(); it != vec.end(); it++) 
	{
		int val = *it;
		cout << val << endl;
	}
	cout << endl;
}
