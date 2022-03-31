#ifndef CUDA_GENERATOR
#define CUDA_GENERATOR

#include "Generator.h"

namespace LS 
{
    class CUDAGenerator : public Generator
    {
        public:
            void generateAllSquares(std::vector<LatinSquare> squares) {}
    };
}

#endif