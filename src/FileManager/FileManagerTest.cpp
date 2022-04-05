#include <memory>
#include <random>
#include <vector>
#include <iostream>
#include <string>

#include "./FileManager.h"
#include "./PrintableInt.h"

using namespace FM;
using IntVec = std::vector<int>;

int main(int argc, char* argv[]) 
{
    FileManager fm;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1,100);

    // initialize some files in the filemanager
    std::string baseFilename = "./TestFiles/file_"; 
    std::string filename1 = baseFilename + "1.txt", filename2 = baseFilename + "2.txt", filename3 = baseFilename + "3.txt", filename4 = baseFilename + "4.txt";
    fm.initFile(filename1); fm.initFile(filename2); 
    fm.initFile(filename3); fm.initFile(filename4);

    // start the FileManager so it can process jobs as they are queued
    fm.start();

     // create 20 vectors w/ 100 random ints each to be written to files, queue them in the FileManager
    for(int i = 0; i < 20; i++)
    {  
        JobVector currVec;
        for(int j = 0; j < 100; j++)
            currVec.push_back(std::make_shared<PrintableInt>(dist(rng)));
        
        // spread the jobs across multiple files
        if(i < 5)
            fm.queueJob(filename1, currVec);
        else if(i < 10)
            fm.queueJob(filename2, currVec);
        else if(i < 15)
            fm.queueJob(filename3, currVec);
        else
            fm.queueJob(filename4, currVec);
    }

    // stop the FileManager, make sure all files are created as expected
    std::cout << "stopping file manager" << std::endl;
    fm.stop();
    std::cout << "file manager has stopped" << std::endl;

    return 0;
}