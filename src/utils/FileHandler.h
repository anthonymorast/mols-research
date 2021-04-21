#ifndef FILE_HANDLER
#define FILE_HANDLER

#include "common.h"
#include <pthread.h>

class FileHandler
{
    public:
        FileHandler();

        // the queue of files to write
        void addToWriteQueue(std::unordered_set<std::string> squares, std::string filename);

        // return false if files are still being written
        bool synchronize(); 

    private:
        // filename -> set of squares
        std::unordered_map<std::string, std::unordered_set<std::string>> _queue;
        std::vector<int> _file_write_pids;  // process ids of files (for synchronization)

        // write a set of squares to a file - threaded, return pid
        int writeFile(std::unordered_set<std::string> squares, std::string filename);
};

// TODO: in the future, we are going to run out of RAM and not be able to process
// TODO: high-order squares. When this happens, the FileHandler class will need to 
// TODO: be extended to handle ALL of the square checking. The class will need to 
// TODO: be able to check if the square (or set of squares) exists already in a set
// TODO: of files (and write it to a file if not [could have a _current_file variable
// TODO: for that]), it will need to check if a square (or set of squares) is orthogonal
// TODO: to any of the squares in a set of files, etc. Will need to keep track of all
// TODO: of the valid files to look through (_file_names or _file_handles variable).
// TODO: This will move the processing from RAM to the disk.

#endif 
