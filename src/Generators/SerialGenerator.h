#ifndef SERIAL_GENERATOR
#define SERIAL_GENERATOR

#include <memory>

#include "Generator.h"
#include "../FileManager.h"

namespace LS 
{
    class SerialGenerator : public Generator
    {
        public:
            void generateAllSquares(std::vector<LatinSquare> squares);
        private:
            std::unique_ptr<FileManager> _fileManager = std::make_unique<FileManager>();
    };

    void SerialGenerator::generateAllSquares(std::vector<LatinSquare> squares) 
    {
        if(squares.size() == 0)
            return;

        
    }
}

#endif