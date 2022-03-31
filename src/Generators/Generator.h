/**
 * @file Generator.h
 * @author Anthony Morast (anthony.a.morast@gmail.com)
 * @brief 
 * The generator class and its subclasses are responsible for generating
 * all reduced Latin squares of a particular order and writing them to a
 * results file. This is going to be most of the algorithm.
 *
 * @date 2022-03-29
 * 
 * @copyright Copyright (c) 2022
 */

#ifndef GENERATOR
#define GENERATOR

#include <vector>
#include <unordered_set>
#include "../LatinSquare/LatinSquare.h"

namespace LS
{
    class GeneratorException : public std::runtime_error 
    {
        public:
            GeneratorException(const std::string msg) : std::runtime_error(msg) {}
            ~GeneratorException() {}
    };

    class Generator 
    {
        public:
            // takes a list of squares, applies permutations, and writes the squares to a file as they're generated
            virtual void generateAllSquares(std::vector<LatinSquare> squares) = 0; 

        protected:
            // use the LS -> R mapping to map each LS to a unique number, the number 
            // is stored in this unordered set. When the size of the set doesn't change
            // we are done generating. If a number is in this set, we have already checked/
            // generated this square, don't do anything more with it. This should save space.
            std::unordered_set<double> _generatedSquareKeys;
            int _order = 0;

            double _get_square_key(short* squareValues);
    };

    double Generator::_get_square_key(short* squareValues)
    {
        throw GeneratorException("Function _get_square_key() has not been implemented.");
    }
}

#endif