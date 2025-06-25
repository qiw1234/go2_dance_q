/*!
 * @file PeriodicTask.h
 * @brief Implementation of a periodic function running in a separate thread.
 * Periodic tasks have a task manager, which measure how long they take to run.
 */

#ifndef PROJECT_PERIODICTASK_H
#define PROJECT_PERIODICTASK_H

#include <string>
#include <thread>
#include <vector>
#include <pthread.h>
#include "utilities/PeriodicTaskManager.h"

class PeriodicTaskManager;

class PeriodicTask {
public:
    PeriodicTask(PeriodicTaskManager* taskManager, float period, std::string name);
    ~PeriodicTask() {stop();}
    void start();
    void stop();
    void printStatus();
    void clearMax();
    bool isSlow();
    virtual void init() = 0;
    virtual void run() = 0;
    virtual void cleanup() = 0;
    std::string getname() {return name;}
    float getPeriod() { return period; }
    float getRuntime() { return lastRuntime; }
    double getRealTime() {return realRunTime;}
    float getMaxPeriod() { return maxPeriod; }
    float getMaxRuntime() { return maxRuntime; }
    float getLastPeriodTime() { return lastPeriodTime; }
//private:
    void loopFunction();
    float period;
    volatile bool running = false;
    float lastRuntime = 0;
    float lastPeriodTime = 0;
    float maxPeriod = 0;
    float maxRuntime = 0;
    double realRunTime = 0;
    std::string name;
    std::thread _thread;
    pthread_attr_t thread_attr;
    struct sched_param schedule_param;
    pthread_t loop = 0;
};

#endif
