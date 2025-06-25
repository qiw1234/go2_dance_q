#include <unistd.h>
#include <cmath>

#include "utilities/PeriodicTaskManager.h"

PeriodicTaskManager *PeriodicTaskManager::get_instance()
{
    static PeriodicTaskManager *ins = 0;
    if(!ins)
    {
        ins = new PeriodicTaskManager();
    }
    return ins;
}

PeriodicTaskManager::~PeriodicTaskManager() {}

/*!
 * Add a new task to a task manager
 */
void PeriodicTaskManager::addTask(PeriodicTask* task)
{
    tasks.push_back(task);
    task->getname();
}

PeriodicTask* PeriodicTaskManager::getTask()
{
    return tasks.at(0);
}

PeriodicTask* PeriodicTaskManager::getTask(const std::string name)
{
    for(auto iter = tasks.begin(); iter != tasks.end(); iter++)
    {
        if(name.compare((*iter)->getname())==0)
        {
            return *iter;
        }
    }
    printf("getTask failed\n");
    exit(0);
}

/*!
 * Print the status of all tasks and rest max statistics
 */
void PeriodicTaskManager::printStatus()
{
  printf("\n----------------------------TASKS----------------------------\n");
  printf("|%-20s|%-6s|%-6s|%-6s|%-6s|%-6s\n", "name", "rt", "rt-max", "T-des",
         "T-act", "T-max");
  printf("-----------------------------------------------------------\n");
  for (auto& task : tasks)
  {
    task->printStatus();
    task->clearMax();
  }
  printf("-------------------------------------------------------------\n\n");
}

/*!
 * Print only the slow tasks
 */
void PeriodicTaskManager::printStatusOfSlowTasks()
{
  for (auto& task : tasks)
  {
    if (task->isSlow())
    {
      task->printStatus();
      task->clearMax();
    }
  }
}

/*!
 * Stop all tasks
 */
void PeriodicTaskManager::stopAll()
{
  for (auto& task : tasks)
  {
    task->stop();
  }
}
