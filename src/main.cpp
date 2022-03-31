#include <unordered_set>
#include <chrono>
#include <iostream>
#include <memory>

#include "Parameters.h"
#include "Generators/Include.h"

std::unique_ptr<LS::Generator> getGenerator(LS::GENERATOR_TYPE type);

class ProcessingException : std::runtime_error
{
    public:
        ~ProcessingException() {}
        ProcessingException(const std::string msg) : runtime_error(msg) {}
};

int main(int argc, char* argv[]) 
{
    auto start = std::chrono::system_clock::now();

    LS::Parameters params(argc, argv);
    std::unique_ptr<LS::Generator> generator = getGenerator(params.getGeneratorType());
    generator->generateAllSquares(params.getIsoReps());

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "Generation completed in " << elapsed.count() << " seconds." << std::endl;

    return 0;
}

std::unique_ptr<LS::Generator> getGenerator(LS::GENERATOR_TYPE type) 
{
    switch(type) 
    {
        case LS::GENERATOR_TYPE::SERIAL:
            return std::make_unique<LS::SerialGenerator>();
        case LS::GENERATOR_TYPE::MPI:
            return std::make_unique<LS::MPIGenerator>();
        case LS::GENERATOR_TYPE::CUDA:
            return std::make_unique<LS::CUDAGenerator>();
        default:
            throw ProcessingException("Could not create Latin square generator.");
    }
}