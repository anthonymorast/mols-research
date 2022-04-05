/**
 * @file FileManager.h
 * @author Anthony Morast (anthony.a.morast@gmail.com)
 * @brief 
 * The file manager will take care of writing the PrintableObjects to a file (or set
 * of files). Essentially, we need a queue that appends printable objects to a file 
 * if the file is currently not being written to.
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
#include <chrono>
#include <pthread.h>
#include <thread>
#include <algorithm>

#include "./PrintableObject.h"

namespace FM
{
    // a JobVector is a vector that defines the jobs (i.e. the objects to be written)
    // sometimes you get tired of typing std::vector<std::shared_ptr<PrintableObject>> and managing the carats
    using JobVector = std::vector<std::shared_ptr<PrintableObject>>;

    class FileManagerException : public std::runtime_error
    {
        public:
            ~FileManagerException() {}
            FileManagerException(const std::string msg) : runtime_error(msg) {}
    };

    class FileManager
    {
        public:
            FileManager(bool append=false, int wait_timeout=120) : _append(append), _max_stop_wait(wait_timeout) {}
            void stop();                            // wraps up the queues and stops the manager PID
            void start();                           // starts the manager PID
            void initFile(std::string filename);    // creates entries for the filename in the maps
            void queueJob(std::string filename, JobVector job);

        private:        
            std::map<std::string, std::shared_ptr<std::ofstream>> _filename_to_filehandle;
            std::map<std::string, std::vector<JobVector>> _filename_to_jobs;    // store jobs for each filename
            std::vector<std::string> _filenames;

            pthread_t _manager_thread_handle;
            bool _append;
            int _max_stop_wait;
            
            void _manager();    // thread used to manage the queues
    };

    void FileManager::start()
    {
        std::thread manager_thread(&FileManager::_manager, this);
        _manager_thread_handle = manager_thread.native_handle();    // store the pthread handle to be 'cancelled' later
        manager_thread.detach();
    }

    void FileManager::_manager() 
    {
        while(true)
        {
            // assumes all three maps have the same keys (this SHOULD be the case)
            for(auto it = _filenames.begin(); it != _filenames.end(); it++) 
            {
                std::string key = (*it);

                // process a job
                if(_filename_to_jobs[key].size() > 0) 
                {
                    // get job and create worker thread
                    auto job = _filename_to_jobs[key].back();
                    for(auto jobIt = job.begin(); jobIt != job.end(); jobIt++) 
                        (*_filename_to_filehandle[key]) << (*jobIt)->getPrintString() << std::endl;
                    _filename_to_jobs[key].pop_back();
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));   // wait before checking again
        }
    }

    void FileManager::initFile(std::string filename) 
    {
        if(std::find(_filenames.begin(), _filenames.end(), filename) == _filenames.end())
            _filenames.push_back(filename);

        if(_filename_to_jobs.find(filename) == _filename_to_jobs.end())
        {
            std::vector<JobVector> jobs;
            _filename_to_jobs.insert(std::pair<std::string, std::vector<JobVector>>(filename, jobs));
        }

        if(_filename_to_filehandle.find(filename) == _filename_to_filehandle.end())
        {
            std::shared_ptr<std::ofstream> fout = std::make_shared<std::ofstream>();
            if(_append)
                 fout->open(filename, std::fstream::out | std::fstream::app);
             else
                 fout->open(filename, std::fstream::out);
            if(!fout->is_open())
                throw FileManagerException("Unable to open file \"" + filename + "\" for writing (append=" + (_append ? "true" : "false") + ").");

            _filename_to_filehandle.insert(std::pair<std::string, std::shared_ptr<std::ofstream>>(filename, fout));
        }
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
            for(auto it = _filename_to_jobs.begin(); it != _filename_to_jobs.end(); it++) 
                if((*it).second.size() != 0) 
                    runningCount++;
            
            // only sleep if necessary
            std::this_thread::sleep_for(std::chrono::seconds(runningCount > 0 ? 2 : 0));
        } while(runningCount > 0);

        // close the files
        for(auto it = _filename_to_filehandle.begin(); it != _filename_to_filehandle.end(); it++)
            it->second->close();

        pthread_cancel(_manager_thread_handle);
    }
}

#endif