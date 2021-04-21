#include "FileHandler.h"

FileHandler::FileHandler()
{
}

// the queue of files to write
void FileHandler::addToWriteQueue(std::unordered_set<std::string> squares, std::string filename)
{
}

// return false if files are still being written
bool FileHandler::synchronize()
{
    return false;
}
        
// write a set of squares to a file - threaded, return pid
int FileHandler::writeFile(std::unordered_set<std::string> squares, std::string filename)
{
    return 0;
}



/*
    private:
        std::unordered_map<std::string filename, std::unordered_set<std::string> squares> _queue;
        std::vector<int> _file_write_pids;  // process ids of files (for synchronization)
*/
