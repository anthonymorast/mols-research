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
#include <chrono>
#include <pthread.h>
#include <thread>
#include <iostream>     // !! debug only

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
            std::map<std::string, bool>  _filename_to_running;                  // true if the thread is running for the filename
            std::map<std::string, pthread_t> _filename_to_thread_handle;
            std::map<std::string, std::vector<JobVector>> _filename_to_jobs;    // store jobs for each filename
            pthread_t _manager_thread_handle;
            bool _append;
            int _max_stop_wait;
            
            void _write_objects(std::string filename, const std::vector<std::string> job); // start a separate thread to write the objects
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
            for(auto it = _filename_to_running.begin(); it != _filename_to_running.end(); it++) 
            {
                std::string key = it->first;
                bool running = it->second;
                pthread_t threadHandle = _filename_to_thread_handle[key];
                if(!running && threadHandle != 0L)  // job finished, stop the thread if need be
                {
                    pthread_cancel(threadHandle);
                    _filename_to_thread_handle[key] = 0L;
                }

                // start the next job on its own thread
                // std::cout << "here: " << key << std::endl;
                if(_filename_to_jobs[key].size() != 0 && !_filename_to_running[key]) 
                {
                    // get job and create worker thread
                    auto job = _filename_to_jobs[key].back();
                    // std::cout << "pop me " << key << " " << _filename_to_jobs[key].size() << " " << job.size() <<  std::endl;
                    job = _filename_to_jobs[key].back();
                    // std::cout << "create job vector: " << key << std::endl;
                    std::vector<std::string> jobValues;
                    for(auto jobIt = job.begin(); jobIt != job.end(); jobIt++) 
                    {
                        // std::cout << (**jobIt).getPrintString() << std::endl;
                        jobValues.push_back((**jobIt).getPrintString());
                    }
                    // std::cout << "before creating worker: " << key << std::endl;
                    std::thread worker(&FileManager::_write_objects, this, key, jobValues);
                    _filename_to_thread_handle[key] = worker.native_handle();
                    worker.detach();
                    // std::cout << "detach job: " << key << std::endl;

                    // set running to true
                    _filename_to_running[key] = true;
                    _filename_to_jobs[key].pop_back();
                    // std::cout << "vector hpdated: " << key << std::endl;
                }
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));   // wait for a second and check again
        }
    }

    void FileManager::initFile(std::string filename) 
    {
        if(_filename_to_jobs.find(filename) == _filename_to_jobs.end())
        {
            std::vector<JobVector> jobs;
            _filename_to_jobs.insert(std::pair<std::string, std::vector<JobVector>>(filename, jobs));
        }

        if(_filename_to_running.find(filename) == _filename_to_running.end())
            _filename_to_running.insert(std::pair<std::string, bool>(filename, false));

        if(_filename_to_thread_handle.find(filename) == _filename_to_thread_handle.end())
            _filename_to_thread_handle.insert(std::pair<std::string, pthread_t>(filename, 0L));
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
        pthread_cancel(_manager_thread_handle);
    }

    void FileManager::_write_objects(std::string filename, const std::vector<std::string> job)
    {
        // open the file
        std::fstream fout;
        if(_append)
            fout.open(filename, std::fstream::out | std::fstream::app);
        else
            fout.open(filename, std::fstream::out);

        if(!fout.is_open())
            throw FileManagerException("Unable to open file \"" + filename + "\" for writing (append=" + (_append ? "true" : "false") + ").");
        
        // write the contents
        // std::cout << filename << " " << job.size() << std::endl;
        for(auto it = job.begin(); it != job.end(); it++)
        {
            // std::cout << filename << " " << (*it) << std::endl;
            // std::cout << (*it)->getPrintString() << std::endl;
            fout << (*it) << std::endl;
        }
        fout.close();

        // set flag in _filename_to_running and thread to 0
        // std::cout << "finishing" << std::endl;
        _filename_to_running[filename] = false;
        _filename_to_thread_handle[filename] = 0L;
    }
}

#endif