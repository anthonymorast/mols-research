#pragma once
#include <iostream>
#include <exception>
using namespace std;

class InvalidSquareException : public exception
{
	virtual const char* what() const throw()
	{
		return "Attempting to operate on invalid square.";
	}
};
