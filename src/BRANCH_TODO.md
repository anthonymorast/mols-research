# square_hash_dev

## Branch Tasks:
+ Use the LS-number formula to store generated squares rather than using the squares themselves
+ Create a file manager class to queue jobs (on a separate thread) to append generated squares to a squares file
    + Delete the squares file at the start of the run so we don't accidentally create huge files
+ Compare serial and cuda implementations and move common logic to 'common.h'
    + Might need to create classes for things like parameters (it will hold permutations, squares, etc.)
    + This will make the code much more maintainable
    + Consider making this more OOP-y by creating classes for the generators that are called from a 'main' class/function
+ Keep Makefile and other build tools up-to-date

## Example Structure

-main.cpp
-- creates a 'FileManager' class
-- creates a 'Parameters' class
-- -- 'Parameters' parses parameters provided command line arguments
-- creates a 'Generator' virtual class based on parameters
-- -- 'Generator' is a virtual class extended by MPI, CUDA, and Serial
-- -- -- 'SerialGenerator', 'CUDAGenerator', 'MPIGenerator'

## Questions
+ how will compilation work with CUDA generator and non-CUDA everything else?
    + can we use nvcc to compile everything?
        + probably a bad idea
    + can we compile the CUDA cpp files into .o's and link to them when using g++?
        + set up a test for this, if this works it is the correct way to do it.