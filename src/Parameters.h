/**
 * @file Parameters.h
 * @author Anthony Morast (anthony.a.morast@gmail.com)
 * @brief 
 *  The parameters class handles the parameter parsing for the logic to 
 *  generate all Latin squares of a particular order. The class parses
 *  the command line parameters, reads in the appropriate permutations 
 *  use to generate the squares, and provides a method to read in the 
 *  initial Latin squares (the isotopy class representatives).
 * @date 2022-01-09
 * 
 * @copyright Copyright (c) 2022
 */

#ifndef PARAMETERS
#define PARAMETERS

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iostream>

#include "../LatinSquare/LatinSquare.h"

namespace LS
{
    class ParametersException : public std::runtime_error
    {
        public:
            ParametersException(const std::string msg) 
                : std::runtime_error(msg) {}
            ~ParametersException() {}
    };

    class Parameters
    {
        private:
            int _order;
            int _argc;
            char** _argv;
            std::vector<short*> _n_permutations;
            std::string _reps_filename;
            std::string _permutations_filename;

            void _verifyParameters();
            void _parse();
            bool _fileExists(std::string filename);
            short* _getArrayFromLine(std::string line, int size);

        public:
            Parameters(int argc, char* argv[]);

            std::vector<LatinSquare> getIsoReps();
            int getOrder() { return _order; }
            std::vector<short*> getPermutations() { return _n_permutations; }
    };

    Parameters::Parameters(int argc, char **argv)
    {
        _argc = argc;
        _argv = argv;

        _verifyParameters();
        _parse();
    }
 
    std::vector<LatinSquare> Parameters::getIsoReps() 
    {
        std::ifstream isofile;
        isofile.open(_reps_filename);
        std::string line;
        std::vector<LatinSquare> isoReps;
        while(std::getline(isofile, line)) 
        {
            LatinSquare sq(_order, _getArrayFromLine(line, _order*_order));
            isoReps.push_back(sq);
        }
        isofile.close();

        return isoReps;
    }

    bool Parameters::_fileExists(std::string filename) 
    {
        std::ifstream f(filename.c_str());
        return f.good();
    }

    void Parameters::_verifyParameters()
    {
        if (_argc < 3) 
            throw ParametersException("Usage: generate_squares <order> <iso reps filename>");

        _order = std::stoi(std::string(_argv[1]));
        _reps_filename = std::string(_argv[2]);
        _permutations_filename = "permutations/" + std::to_string(_order) + "_perm.dat";

        if(!_fileExists(_reps_filename)) 
            throw ParametersException("Isotopy class representives file does not exists: \"" + _reps_filename + "\"");
        if(!_fileExists(_permutations_filename)) 
            throw ParametersException("Permutation file does not exists: \"" + _permutations_filename + "\". You can use the utilities to generate it.");
    }

    short* Parameters::_getArrayFromLine(std::string line, int size) 
    {
        line.erase(std::remove(line.begin(), line.end(), ' '), line.end()); // trim white space
        short* vals = new short[size];
        const char* linearr = line.c_str();
        for(int i = 0; i < size; i++) 
            vals[i] = linearr[i] - '0';
        return vals;
    }

    void Parameters::_parse() 
    {
        std::string line;
        std::ifstream permfile;
        permfile.open(_permutations_filename);
        while(std::getline(permfile, line)) 
            _n_permutations.push_back(_getArrayFromLine(line, _order));
        permfile.close();
    }
}

#endif