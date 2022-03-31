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
#include <cctype>

#include "../LatinSquare/LatinSquare.h"

namespace LS
{
    class ParametersException : public std::runtime_error
    {
        public:
            ParametersException(const std::string msg) : std::runtime_error(msg) {}
            ~ParametersException() {}
    };
    enum GENERATOR_TYPE { SERIAL, CUDA, MPI };

    class Parameters
    {
        private:
            int _order;
            int _argc;
            char** _argv;
            std::vector<short*> _n_permutations;
            std::string _reps_filename;
            std::string _permutations_filename;
            GENERATOR_TYPE _generator_type = SERIAL;

            void _verifyParameters();
            void _parse();
            bool _fileExists(std::string filename);
            void _parseGeneratorType(std::string arg);
            short* _getArrayFromLine(std::string line, int size);

        public:
            Parameters(int argc, char* argv[]);

            std::vector<LatinSquare> getIsoReps();
            int getOrder() { return _order; }
            std::vector<short*> getPermutations() { return _n_permutations; }
            GENERATOR_TYPE getGeneratorType() { return _generator_type; }
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
            throw ParametersException("Usage: generate_squares <order> <iso reps filename> --<generator(default): cuda,(serial),mpi>");

        _order = std::stoi(std::string(_argv[1]));
        _reps_filename = std::string(_argv[2]);
        _permutations_filename = "permutations/" + std::to_string(_order) + "_perm.dat";

        if(!_fileExists(_reps_filename)) 
            throw ParametersException("Isotopy class representives file does not exists: \"" + _reps_filename + "\"");
        if(!_fileExists(_permutations_filename)) 
            throw ParametersException("Permutation file does not exists: \"" + _permutations_filename + "\". You can use the utilities to generate it.");
        
        if(_argc > 3)    // will need to expand this to be smarter if more parameters are added
            _parseGeneratorType(_argv[_argc-1]);
    }

    void Parameters::_parseGeneratorType(std::string arg)
    {
        arg = arg.substr(2);    // remove --
        std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) { return std::tolower(c); });
        
        // skip serial check as it is the default
        if(arg == "cuda") 
            _generator_type = CUDA;
        else if (arg == "mpi") 
            _generator_type = MPI;
        else 
            throw ParametersException("Invalid generator type \"" + arg + "\".");
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