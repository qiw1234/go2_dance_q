#include <unistd.h>
#include <cmath>
#include "utilities/PeriodicTask.h"
#include "utilities/Timer.h"
#include <sys/timerfd.h>
#include <iostream>
#include <string.h>

using namespace std;

//void loopFunction_test()
//{
//    Timer tim;
//    PeriodicTask _periodicTask = new PeriodicTask();
//    int seconds = (int)_periodicTask.period;
//    int nanoseconds = (int)(1e9 * std::fmod(_periodicTask.period, 1.f));
//    //printf("nanoseconds = %d\n", nanoseconds);

//    unsigned long long missed = 0;
//    auto timerFd = timerfd_create(CLOCK_MONOTONIC, 0);
//    itimerspec timerSpec;
//    timerSpec.it_interval.tv_sec = seconds;
//    timerSpec.it_value.tv_sec = seconds;
//    timerSpec.it_value.tv_nsec = nanoseconds;
//    timerSpec.it_interval.tv_nsec = nanoseconds;
//    timerfd_settime(timerFd, 0, &timerSpec, nullptr);


//    while (_periodicTask.running)
//    {
//        _periodicTask.lastPeriodTime = (float)tim.getSeconds();
//        tim.start();
//        if(_periodicTask.running == false)break;
//        _periodicTask.run();
//        _periodicTask.lastRuntime = (float)tim.getSeconds();
//        _periodicTask.realRunTime = tim.getRealTime();
//        //printf("%s (P:%f, R:%f)\n", name.c_str(), lastPeriodTime, lastRuntime);
//        //TaskManager_Log("%s (P:%f, R:%f)\n", name.c_str(), lastPeriodTime, lastRuntime);
//        int m = read(timerFd, &missed, sizeof(missed));
//        (void)m;
//        _periodicTask.maxPeriod  = std::max(_periodicTask.maxPeriod, _periodicTask.lastPeriodTime);
//        _periodicTask.maxRuntime = std::max(_periodicTask.maxRuntime, _periodicTask.lastRuntime);
//    }
//    printf("loopFunction close\n");
//}

void *test(void *arg)
{
    PeriodicTask *_periodicTask = (PeriodicTask*)arg;
    Timer tim;

//    cout << "arg = " << arg << endl;
//    cout << "_periodicTask = " << _periodicTask << endl;
//    cout << "running = " << (*_periodicTask).running << endl;
    int seconds = (int)_periodicTask->period;
    int nanoseconds = (int)(1e9 * std::fmod(_periodicTask->period, 1.f));
    //printf("nanoseconds = %d\n", nanoseconds);

    unsigned long long missed = 0;
    auto timerFd = timerfd_create(CLOCK_MONOTONIC, 0);
    itimerspec timerSpec;
    timerSpec.it_interval.tv_sec = seconds;
    timerSpec.it_value.tv_sec = seconds;
    timerSpec.it_value.tv_nsec = nanoseconds;
    timerSpec.it_interval.tv_nsec = nanoseconds;
    timerfd_settime(timerFd, 0, &timerSpec, nullptr);

    //cout << _periodicTask->running << endl;

    while (_periodicTask->running)
    {
        _periodicTask->lastPeriodTime = (float)tim.getSeconds();
        tim.start();
        if(_periodicTask->running == false)break;

        _periodicTask->run();
        _periodicTask->lastRuntime = (float)tim.getSeconds();
        _periodicTask->realRunTime = tim.getRealTime();
        //printf("%s (P:%f, R:%f)\n", name.c_str(), lastPeriodTime, lastRuntime);
        //TaskManager_Log("%s (P:%f, R:%f)\n", name.c_str(), lastPeriodTime, lastRuntime);
        int m = read(timerFd, &missed, sizeof(missed));
        (void)m;
        _periodicTask->maxPeriod  = std::max(_periodicTask->maxPeriod, _periodicTask->lastPeriodTime);
        _periodicTask->maxRuntime = std::max(_periodicTask->maxRuntime, _periodicTask->lastRuntime);
    }

//    memset(&_periodicTask->schedule_param, 0, sizeof(_periodicTask->schedule_param));
//    cout << "schedule_param = " << _periodicTask->schedule_param.__sched_priority << endl;
    pthread_attr_getschedparam(&_periodicTask->thread_attr, &_periodicTask->schedule_param);
    //cout << "schedule_param = " << _periodicTask->schedule_param.__sched_priority << endl;
    printf("schedule_param = %d\n", _periodicTask->schedule_param);
    printf("loopFunction close\n");
}
void *func(void* p)
{

    PeriodicTask *_task = (PeriodicTask *)p;
    static long long count = 0,count_all = 0;
    //printf("loopfunction\n");
    Timer tim;

    int seconds = (int)_task->period;
    int nanoseconds = (int)(1e9 * std::fmod(_task->period, 1.f));
    //printf("nanoseconds = %d\n", nanoseconds);

    unsigned long long missed = 0;
    auto timerFd = timerfd_create(CLOCK_MONOTONIC, 0);
    itimerspec timerSpec;
    timerSpec.it_interval.tv_sec = seconds;
    timerSpec.it_value.tv_sec = seconds;
    timerSpec.it_value.tv_nsec = nanoseconds;
    timerSpec.it_interval.tv_nsec = nanoseconds;
    timerfd_settime(timerFd, 0, &timerSpec, nullptr);
    while (_task->running)
    {
        _task->lastPeriodTime = (float)tim.getSeconds();
        tim.start();
        if(_task->running == false)break;

        _task->run();


        count_all++;

        //printf("%s (P:%f, R:%f)\n", name.c_str(), lastPeriodTime, lastRuntime);
        //TaskManager_Log("%s (P:%f, R:%f)\n", name.c_str(), lastPeriodTime, lastRuntime);
        int m = read(timerFd, &missed, sizeof(missed));
        (void)m;
        _task->lastRuntime = (float)tim.getSeconds();
        _task->realRunTime = tim.getRealTime();
        if(fabs(_task->lastRuntime - _task->period) >  _task->period*0.2)
        {
            count++;
            printf("[%lld]lastRuntime: %f  overpersent: %f%% \n",count_all,_task->lastRuntime*1000.0,(double)count/(double)count_all*100.0);
        }
        _task->maxPeriod  = std::max(_task->maxPeriod, _task->lastPeriodTime);
        _task->maxRuntime = std::max(_task->maxRuntime, _task->lastRuntime);
    }

}

PeriodicTask::PeriodicTask(PeriodicTaskManager* taskManager, float period, std::string name) : period(period), name(name)
{
    taskManager->addTask(this);
}

void PeriodicTask::start()
{
    if (running)
    {
        printf("[PeriodicTask] Tried to start %s but it was already running!\n", name.c_str());
        return;
    }
    init();
    running = true;
    int hr = 0;
    //_thread = std::thread(&PeriodicTask::loopFunction, this);

    pthread_create(&loop, NULL, &func, this);
//    pthread_attr_init(&thread_attr);
//    schedule_param.sched_priority = 99;
//    //schedule_param.__sched_priority = 62;
//    pthread_attr_setinheritsched(&thread_attr, PTHREAD_EXPLICIT_SCHED);
//    pthread_attr_setschedpolicy(&thread_attr, SCHED_RR);
//    hr = pthread_attr_setschedparam(&thread_attr, &schedule_param);
//    //printf("pthread_attr_setschedparam = %d\n", hr);

//    //pthread_create(&loop, &thread_attr, test, NULL);
//    pthread_create(&loop, &thread_attr, test, this);

//    printf("pthread_create\n");
//    memset(&schedule_param, 0, sizeof(schedule_param));
//    pthread_attr_getschedparam(&thread_attr, &schedule_param);
//    cout << "schedule_param = " << schedule_param.__sched_priority << endl;

}

void PeriodicTask::stop()
{
    if (!running)
    {
        printf("[PeriodicTask] Tried to stop %s but it wasn't running!\n", name.c_str());
        return;
    }
    running = false;
    printf("running = false\n");
    cleanup();
}

bool PeriodicTask::isSlow()
{
    return maxPeriod > period * 1.3f || maxRuntime > period;
}

void PeriodicTask::clearMax()
{
    maxPeriod = 0;
    maxRuntime = 0;
}

void PeriodicTask::printStatus()
{
    if (!running) return;
    if (isSlow())
    {
        printf("|%-20s|%6.4f|%6.4f|%6.4f|%6.4f|%6.4f\n",name.c_str(),
        lastRuntime, maxRuntime, period,lastPeriodTime, maxPeriod);
    }
    else
    {
        printf("|%-20s|%6.4f|%6.4f|%6.4f|%6.4f|%6.4f\n", name.c_str(),
        lastRuntime, maxRuntime, period, lastPeriodTime, maxPeriod);
    }
}

void PeriodicTask::loopFunction()
{
    Timer tim;

    int seconds = (int)period;
    int nanoseconds = (int)(1e9 * std::fmod(period, 1.f));
    //printf("nanoseconds = %d\n", nanoseconds);

    unsigned long long missed = 0;
    auto timerFd = timerfd_create(CLOCK_MONOTONIC, 0);
    itimerspec timerSpec;
    timerSpec.it_interval.tv_sec = seconds;
    timerSpec.it_value.tv_sec = seconds;
    timerSpec.it_value.tv_nsec = nanoseconds;
    timerSpec.it_interval.tv_nsec = nanoseconds;
    timerfd_settime(timerFd, 0, &timerSpec, nullptr);


    while (running)
    {
        lastPeriodTime = (float)tim.getSeconds();
        tim.start();
        if(running == false)break;

        run();
        lastRuntime = (float)tim.getSeconds();
        realRunTime = tim.getRealTime();
        //printf("%s (P:%f, R:%f)\n", name.c_str(), lastPeriodTime, lastRuntime);
        //TaskManager_Log("%s (P:%f, R:%f)\n", name.c_str(), lastPeriodTime, lastRuntime);
        int m = read(timerFd, &missed, sizeof(missed));
        (void)m;
        maxPeriod  = std::max(maxPeriod, lastPeriodTime);
        maxRuntime = std::max(maxRuntime, lastRuntime);
    }

    printf("loopFunction close\n");
}
