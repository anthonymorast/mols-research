#ifndef MPI_GENERATOR
#define MPI_GENERATOR

#include "Generator.h"

namespace LS 
{
     class MPIGenerator : public Generator
    {
        public:
            MPIGenerator() { throw GeneratorException("MPI generator has not been implemented."); }
            void generateAllSquares(std::vector<LatinSquare> squares) {}
    };
}

#endif