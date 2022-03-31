/**
 * @file FileManager.h
 * @author Anthony Morast (anthony.a.morast@gmail.com)
 * @brief 
 * The file manager will take care of writing the reduced squares to a file (or sequence
 * of files which will later be post-processed). Essentially, we need a queue that appends
 * a list of squares to a file if the file is currently not being written to.
 *
 * @date 2022-03-29
 * 
 * @copyright Copyright (c) 2022
 */

#ifndef FILE_MANAGER
#define FILE_MANAGER

#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <memory>

#include "../LatinSquare/LatinSquare.h"

namespace LS
{
    using JobVector = std::vector<std::shared_ptr<PrintableObject>>;    // a JobVector is a vector that defines the jobs (i.e. what is to be written)

    class FileManagerException : public std::runtime_error
    {
        public:
            ~FileManagerException() {}
            FileManagerException(const std::string msg) : runtime_error(msg) {}
    };

    class FileManager
    {
        public:
            void stop();                            // wraps up the queues and stops the manager PID
            void initFile(std::string filename);    // creates entries for the filename in the maps
            void start();                           // starts the manager PID
            void queueJob(std::string filename, JobVector job);

        private:        
            std::map<std::string, int>  _filename_to_pid;   // may need to update this to be string -> thread, depending on how we tell if a thread is running
            std::map<std::string, std::vector<JobVector>> _filename_to_jobs;    // store jobs for each filename
            int _manager_pid = 0;
            
            // start a separate thread to write the objects
            void _writeObjects(std::string filename, JobVector job, bool append=true);
            void _manager();    // thread used to manage the queues
    };

    void FileManager::start()
    {

    }

    void FileManager::initFile(std::string filename) 
    {
        //TODO: might throw an error here if filename already initalized, won't hurt anything if that case is ignored
        if(_filename_to_jobs.find(filename) == _filename_to_jobs.end())
        {
            std::vector<JobVector> jobs;
            _filename_to_jobs.insert(std::pair<std::string, std::vector<JobVector>>(filename, jobs));
        }
        
        if(_filename_to_pid.find(filename) == _filename_to_pid.end())
            _filename_to_pid.insert(std::pair<std::string, int>(filename, 0));
    }

    void FileManager::queueJob(std::string filename, JobVector job) 
    {
        if(_filename_to_jobs.find(filename) == _filename_to_jobs.end())
            throw FileManagerException("Filename \"" + filename + "\" has not been initalized in the FileManager.");
        _filename_to_jobs[filename].push_back(job);
    }

    void FileManager::stop()
    {
        int runningCount;
        do {
            runningCount = 0;
            // sleep a couple of seconds

            int runningCount = 0;
            for(auto it = _filename_to_pid.begin(); it != _filename_to_pid.end(); it++) 
                if((*it).second != 0) 
                    runningCount++;

        } while(runningCount > 0);

        // kill _manager_pid;
    }

    void FileManager::_writeObjects(std::string filename, JobVector job, bool append)
    {
        // start a thread to write the printable objects to a file
        // will need to check if a thread is currently writing to a filename
        // is there a way to check if a process is currently running? If so will need a map (filename -> PID)
        // essentially want to create a queue for writing.
    }
}

#endif