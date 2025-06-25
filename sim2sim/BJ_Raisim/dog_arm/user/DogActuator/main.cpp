#include <iostream>
#include <fstream>

#include <pthread.h>
#include <stdio.h>
#include <sched.h>
#include <stdlib.h>
#include <unistd.h>
//#include <robot_controller_task.h>
#include <robot_drive_task.h>
#include <utilities/PeriodicTask.h>
#include "sys/wait.h"
#include "sys/types.h"
#include "time.h"


using namespace std;

#include <cppTypes.h>
#include <utilities/SharedMemory.h>
#define AUTO_FLAG 1

int ch_test = 0;

xpRobotDriTask *xp_driv_task;

void Stop(int sig)
{
    printf("xp_driv_task stop\n");
    xp_driv_task->stop();
}

int main(int argc, char* argv[])
{
    PeriodicTaskManager *taskManager = PeriodicTaskManager::get_instance();
    printf("********###start_test_task*************\n");

    xp_driv_task = new xpRobotDriTask(taskManager,0.002,DRIV_TASK_NAME);//unit seconds

    //signal(SIGINT, Stop);

    xp_driv_task->start();
    pthread_join(xp_driv_task->loop, NULL);
#if !AUTO_FLAG
    while(1)
    {
#if _NO_SIM == 1
        printf("ON\n");
#endif

        char c = getchar();
        if ( c == 'q' )
        {
            break;
        }
    }
#endif
    xp_driv_task->stop();


//    int act_cnt = 0;
//    EP_CAN_MULTI_OBJ m_recvObj[100] = {0};
//    xpCan *_xp_can = xpCan::get_instance();
//    while(1)
//    {
//        hr = EphCAN_ReceiveMulti(_xp_can->cardNum, 0, 32, m_recvObj, &act_cnt);
//        usleep(1000);
//    }

    return 0;
}
