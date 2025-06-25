#include <utilities/SharedMemory.h>

ShareMemory *ShareMemory::get_instance(const char *sharememory_name)
{
    static ShareMemory *ins = NULL;
    if(!ins)
    {
        ins = new ShareMemory(sharememory_name);
        printf("ShareInfo size %d \n",sizeof(ins->shareinfo));
        printf("ocu_package size %d \n",sizeof(ins->shareinfo.ocu_package));
        printf("servo_package_socket size %d \n",sizeof(ins->shareinfo.servo_package_socket));
        printf("send_package_socket size %d \n",sizeof(ins->shareinfo.send_package_socket));
    }
    return ins;
}

ShareMemory::ShareMemory(const char *sharememory_name)
{
    m_sem = new sem_com(creat_keyid("xpShareMemory", IPC_PROJ_ID + 2));
    SemLock();
    m_shareMemory = CreateShareMemory(sharememory_name, SHAREMEMORY_MAX_SIZE);
    printf("sizeof ShareInfo : %zu \n",sizeof(shareinfo));
    // m_shareMemory = CreateShareMemory("xpEDogShareMemory",2*1024*1024);
    SemUnLock();
    if(m_shareMemory == NULL)
    {
        printf("ShareMemory Create Failed\n");
    }
    else
    {
        printf("ShareMemory Create OK\n");
    }
}

int ShareMemory::PutToShareMem(void *src_buffer, unsigned int length)
{
    unsigned int offset = GetOffset(src_buffer);
    if(offset + length > SHAREMEMORY_MAX_SIZE)
	{
        printf("Input length[%u] exceeds memory length[%u] error", offset + length, SHAREMEMORY_MAX_SIZE);
		return -1;
	}
    if((m_shareMemory == NULL)||(src_buffer == NULL))
    {
        perror("put failed memory is null\n");
        return -1;
    }
    SemLock();
    char *m2_shareMemory = new char[sizeof(shareinfo)];
    memcpy((char*)m2_shareMemory, (char*)m_shareMemory, sizeof(shareinfo));
    memcpy((char*)m2_shareMemory + offset, src_buffer, length);
    memcpy((char*)m_shareMemory,(char*)m2_shareMemory,sizeof(shareinfo));
    delete m2_shareMemory;
    SemUnLock();
	return length;
}

int ShareMemory::GetFromShareMem(void *dts_buffer, unsigned int length)
{	
    unsigned int offset = GetOffset(dts_buffer);
    if(offset + length > SHAREMEMORY_MAX_SIZE)
	{
        printf("Input length[%u] exceeds memory length[%u] error", offset + length, SHAREMEMORY_MAX_SIZE);
        return -1;
	}
    if(m_shareMemory == NULL)
    {
        //perror("get failed memory is null\n");
        return -1;
    }
    SemLock();
    memcpy(dts_buffer, (char*)m_shareMemory + offset, length);
    SemUnLock();
    return length;
}

void *ShareMemory::CreateShareMemory(const char *name, unsigned int size)//("xpEDogShareMemory",2*1024*1024)
{
	void *memory;
	struct shmid_ds buf;

    shmid = shmget(creat_keyid(name, IPC_PROJ_ID), size, 0666 | IPC_CREAT);
    printf("sharememory shmid: %d  key: %x \n",shmid,creat_keyid(name, IPC_PROJ_ID));
//     `shmget` 函数用于获取共享内存的标识符，或者创建一个新的共享内存区域。
//             shmid = shmget(key, size, 0666 | IPC_CREAT);
//      `shmget` 函数接受三个参数：
//          - `key`：这是一个由 `ftok` 函数生成的 IPC 键值，用于标识共享内存区域。它是一个 `key_t` 类型的值。
//          - `size`：这是要创建的共享内存的大小，以字节为单位。如果是创建新的共享内存区域，这个参数指定了需要分配的内存大小。
//          - `0666 | IPC_CREAT`：这是一个位掩码，用于设置共享内存的权限和标志。`0666` 表示权限设置为用户、组和其他用户都具有读写权限，`IPC_CREAT` 表示如果共享内存不存在，则创建一个新的共享内存区域。
//             如果共享内存区域已经存在，则 `shmget` 函数将返回该共享内存区域的标识符。
//             如果共享内存区域不存在，并且指定了 `IPC_CREAT` 标志，则 `shmget` 函数将创建一个新的共享内存区域，并返回其标识符。如果出现错误，`shmget` 函数将返回 -1，并设置 `errno` 来指示错误的原因。
	if (-1 == shmid)
    {
        perror("shmget err");
        return NULL;
    }
    //printf("shmid:%d \n", shmid);
	memory = shmat(shmid, NULL, 0);
//     `shmat` 函数用于将共享内存连接到当前进程的地址空间中，以便进程可以访问共享内存中存储的数据。
//             memory = shmat(shmid, NULL, 0);
//         在这个代码片段中，`shmat` 函数接受三个参数：
//         - `shmid`：这是共享内存的标识符，由之前调用 `shmget` 函数返回。它是一个整数值。
//         - `NULL`：这个参数通常设置为 `NULL`，表示让系统自动选择一个适当的地址来连接共享内存。如果你想要指定连接的地址，可以传递一个非空的指针，但是这种情况下需要保证指定的地址是合法且未被占用的。
//         - `0`：这个参数通常设置为 `0`，表示对共享内存区域的操作没有特殊要求。
//            如果连接成功，`shmat` 函数将返回一个指向共享内存区域的指针，即 `memory`。如果出现错误，它将返回 `(void *)-1`，并设置 `errno` 来指示错误的原因。
//            需要注意的是，一旦共享内存被连接到进程的地址空间中，进程就可以直接访问共享内存中存储的数据。因此，在操作完共享内存后，应该调用 `shmdt` 函数将其从进程的地址空间中分离，以避免内存泄漏或其他问题。
	if ((void *)-1 == memory)
    {
        perror("shmget err");
        return NULL;
    }
	
	shmctl(shmid, IPC_STAT, &buf);
	if (buf.shm_nattch == 1)
	{
		memset(memory, 0, size);
	}	
	
	return memory;
}

void ShareMemory::ShareMemClose(bool bDestroyShm)
{
    if(m_shareMemory)
    {
        shmdt(m_shareMemory);
        if(bDestroyShm)
        {
            shmctl(shmid, IPC_RMID, 0);
        }
        m_sem->~sem_com();
        m_shareMemory = NULL;
        printf("ShareMemory close\n");
    }
}



