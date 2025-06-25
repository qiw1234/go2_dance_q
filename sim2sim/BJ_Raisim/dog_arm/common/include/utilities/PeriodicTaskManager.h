#ifndef PERIODICTASKMANAGER_H
#define PERIODICTASKMANAGER_H

#include <string>
#include <thread>
#include <vector>
#include "utilities/PeriodicTask.h"

#define PERIODICTASKMANAGER_DBG 0
#if PERIODICTASKMANAGER_DBG
    #define TaskManager_Log(format,...)  printf("[TM Log Info]" format, ##__VA_ARGS__)
#else
    #define TaskManager_Log(format,...)
#endif

#define DRIV_TASK_NAME std::string("driv_task")
#define CTRL_TASK_NAME std::string("ctrl_task")
#define CTRL_INTERFACE_TASK_NAME std::string("ctrl_interface_task")

class PeriodicTask;

class PeriodicTaskManager
{
    public:
    static PeriodicTaskManager *get_instance();
    PeriodicTaskManager() = default;
    ~PeriodicTaskManager();
    void addTask(PeriodicTask* task);
    PeriodicTask* getTask();
    PeriodicTask* getTask(const std::string name);
    void printStatus();
    void printStatusOfSlowTasks();
    void stopAll();
    private:
    std::vector<PeriodicTask*> tasks;
};

#endif
