/*************************************************
  Copyright (C), 2004-2021, Enpht Tech. Co., Ltd.
  File name:     EphCAN_32.h  // 文件名

  Description:   此文件包含各个API函数的声明及功能注释
  
  History:        // 历史修改记录
      <author>  <time>   <version >   <desc>
      Yb		2014-12   1.0		  build this moudle 
      Yb		2021-4    1.1		  提速，增加一次读取多个通道数据。

*************************************************/
#ifndef _EphCANAPI_H_
#define _EphCANAPI_H_

#include "visatype.h"

#ifdef EPH_LINUX
#include <libusb-1.0/libusb.h>
#endif

//锟斤拷锟斤拷锟斤拷
#define		BAUDRATE_COUNT	10

#define		RATE_1000KHZ	0	//1000KHz		
#define		RATE_800KHz		1	//800KHz		
#define		RATE_500KHZ		2	//500KHz		
#define		RATE_250KHZ		3	//250KHz		
#define		RATE_125KHZ		4	//1250KHz		
#define		RATE_100KHZ		5	//100KHz		
#define		RATE_50KHZ		6	//50KHz			
#define		RATE_20KHZ		7	//20KHz			
#define		RATE_12KHZ		8	//12KHz			
#define		RATE_10KHZ		9	//10KHz			

//CAN 芯片 工作模式
#define		WM_NORMAL		0		//正常模式		
#define		WM_SLEEP		1		//休眠模式		
#define		WM_LOOP			2		//环回模式		
#define		WM_LISTEN		3		//仅监听模式	
#define		WM_CONFIG		4		//配置模式		

//单次发送，正常发送
#define		TRANS_NORMAL	0	//重复
#define		TRANS_ONCE		1	//单次

//接收滤波模式
#define		FILTER_NULL		0	//关闭滤波
#define		FILTER_STANDARD	1	//标准帧滤波
#define		FILTER_EXTERN	2	//扩展帧滤波

//帧标记
#define		STANDARD_FLAG	0	//发送接收标准帧标记
#define		EXTERN_FLAG		1	//发送接收扩展帧标记

// EP_CAN_INIT结构体定义了初始化CAN的配置，结构体将在Eph_InitCAN函数中被填充
typedef struct _EP_CAN_INIT_
{
	unsigned int	AC[6];				// 验收代码
	unsigned int	AM;					// 验收屏蔽代码
	unsigned int 	FilterType;         // 验收屏蔽类型，0=关闭，1=标准，2=扩展
	unsigned int	bitRate;			// 传输速率
	unsigned int 	SFType;				// 发送帧类型，0时为正常发送，1时为单次发送
} EP_CAN_INIT, *EP_CAN_INIT_LP;

// EP_CAN_OBJ结构体在EphCAN_Transmit和EphCAN_Receive中被用来传送CAN信息帧。
typedef struct _EP_CAN_OBJ_
{
	unsigned int	ID			: 29;	// 报文ID
	unsigned int	TimeStamp		;	// 接收到信息帧时的时间标示，从CAN控制器初始化开始计时
	unsigned char	TimeFlag	:  1;	// 是否使用时间标识，为1时TimeStamp有效，TimeFlag和TimeStamp只在此帧为接收帧时有意义。
	unsigned char	RemoteFlag	:  1;	// 是否是远程帧  0:数据;CAN控制器将发送数据帧,1:远程;CAN控制器讲发送远程帧
	unsigned char	ExternFlag	:  1;	// 是否是扩展帧 0：标准帧  1：扩展帧
	unsigned char	DataLen		:  4;	// 数据长度(<=8)，即Data的长度。
	unsigned char	Data[8]			;	// 报文的数据
} EP_CAN_OBJ, *EP_CAN_OBJ_PTR;

// 2021.4.26
typedef struct _EP_CAN_OBJ_MULTI_
{
	unsigned int	ch; // 通道号
	unsigned int	ID			;	// 报文ID
	unsigned int	TimeStamp	;	// 接收到信息帧时的时间标示，从CAN控制器初始化开始计时
	unsigned char	TimeFlag	;	// 是否使用时间标识，为1时TimeStamp有效，TimeFlag和TimeStamp只在此帧为接收帧时有意义。
	unsigned char	RemoteFlag	;	// 是否是远程帧  0:数据;CAN控制器将发送数据帧,1:远程;CAN控制器讲发送远程帧
	unsigned char	ExternFlag	;	// 是否是扩展帧 0：标准帧  1：扩展帧
	unsigned char	DataLen		;	// 数据长度(<=8)，即Data的长度。
	unsigned char	Data[8]		;	// 报文的数据	
} EP_CAN_MULTI_OBJ, *EP_CAN_MULTI_OBJ_PTR;

#ifdef __cplusplus
extern "C"{
#endif

/*-----------------------------------------------------
* Function:     EphCan_USB_AutoConnectToFirst
* Description:	自动连接到系统找到的第一张CAN模块
* Parameters: 
*			    cardnum	[out]	: 返回已连接到的模块句柄
* Return:       0: 表示成功 非0: 表示错误
------------------------------------------------------*/
ViStatus  _VI_FUNC EphCan_USB_AutoConnectToFirst(
	ViUInt32	*cardnum);

/*-----------------------------------------------------
* Function:     EphCan_USB_AutoConnectToUsbAddr
* Description:	根据USB设备地址连接到指定的CAN模块，USB设备地址由系统分配
* Parameters: 
*			    UsbAddr	[in]	: 模块总线号
*			    cardnum	[out]	: 返回已连接到的模块句柄
* Return: 0: 表示成功  非0: 表示失败
------------------------------------------------------*/
ViStatus _VI_FUNC EphCan_USB_AutoConnectToUsbAddr(
	ViUInt32	UsbAddr,
	ViUInt32	*cardnum);

/**********************************************************
* Function:		EphCAN_Reset
* Description:	软件复位，所有通道停止发送，清空发送、接收缓存。所有通道速率设置为1000KHz
* Parameters: 
*				cardnum	[in]	: 模块句柄
* Return:		0:表示成功 非0时:使用EphCAN_StatusGetString()获取返回值信息描述
**********************************************************/
ViStatus _VI_FUNC EphCAN_Reset(
	ViUInt32	cardnum);

/**********************************************************
* Function:		EphCAN_GetManuID
* Description:	获取厂商ID
* Parameters: 
*				cardnum	[in]	: 模块句柄
*				manuID	[out]	: 获取厂商ID，低16位有效，为0x41F8
* Return:		0: 表示成功 非0时: 使用EphCAN_StatusGetString()获取返回值信息描述
**********************************************************/
ViStatus _VI_FUNC EphCAN_GetManuID(
	ViUInt32	cardnum,
	ViUInt32	*manuID);

/**********************************************************
* Function:		EphCAN_GetDevID
* Description:	获取模块ID
* Parameters: 
*				cardnum	[in]	: 模块句柄
*				devID	[out]	: 返回模块ID
* Return:		0: 表示成功 非0: 表示错误
**********************************************************/
ViStatus _VI_FUNC EphCAN_GetDevID(
	ViUInt32	cardnum,
	ViUInt32	*devID);

/**********************************************************
* Function:		EphCAN_GetVersion
* Description:	获取模块版本号
* Parameters: 
*				cardnum	[in]	: 模块句柄
*				Version	[out]	: 返回模块版本号，如0x0100，表示V1.00
				sVersion[out]	: 返回API版本号，如0x0100，表示V1.00
			
* Return:		0: 表示成功 非0: 表示错误
**********************************************************/
ViStatus _VI_FUNC EphCAN_GetVersion(
    ViUInt32	cardnum,
	ViUInt32	*Version,
	ViUInt32	*sVersion);

/**********************************************************
* Function:		EphCAN_Close
* Description:	自动关闭一个模块的连接
*				已关闭的模块句柄不能再使用，只有重新连接后才可以使用
*				在退出应用软件前必须调用该函数关闭所有的连接
* Parameters: 
*				cardnum	[in]	: 模块句柄
* Return:		0: 表示成功 非0: 表示错误
**********************************************************/
ViStatus _VI_FUNC EphCAN_Close(
	ViUInt32	cardnum);

/**********************************************************
* Function:		EphCAN_StatusGetString
* Description:	将其他API函数的返回值转换为字符串
*			   
*			   
* Parameters: 
*				hr		[in]	: 返回值
				pString	[out]	: 返回值信息，256字节
* Return:		固定返回 API_SUCCESS
**********************************************************/
ViStatus _VI_FUNC EphCAN_StatusGetString(
	ViStatus	hr,  
	char		*pString);

/********************************************************************
* 函数名称： EphCAN_Transmit
* 函数功能： 向指定通道发送指定数量的CAN数据
* 参数：	
*			cardnum		[in]	: 设备句柄
*			ch			[in]	: 通道号
*			len			[in]	: 发送CAN帧数
*			pSend		[in]	: 发送CAN数据
* 返回值：  0 = 指定通道没有接收到数据， 1 = 有效数据
********************************************************************/
ViStatus _VI_FUNC EphCAN_Transmit(
	ViUInt32	cardnum,
	ViUInt16	ch,
	ViUInt32	len,
	EP_CAN_OBJ	*pSend);

/********************************************************************
* 函数名称： 拷EphCAN_Receive
* 函数功能： 从指定通道处读取指定数据的CAN帧数据
* 参数：
*			cardnum		[in]	: 设备句柄
*			ch			[in]	: 通道号
*			max_num		[in]	: 读取CAN帧数
* 			pReceive	[out]	: CAN数据对象数组
*			act_num		[out]	: 实际接收到的CAN帧数
* 返回值：  0 = 成功，非0 = 失败
********************************************************************/
ViStatus _VI_FUNC  EphCAN_Receive(
	ViUInt32	cardnum,	
	ViUInt16	ch,
	ViUInt32	max_num,
	EP_CAN_OBJ	*pReceive,
	ViUInt32	*act_num);

/********************************************************************
* 函数名称：  EphCAN_InitCAN
* 函数功能： 设置通道配置信息
* 参数：     cardnum		[in] : 设备句柄
*			ch			   [in]	: 通道号
*			obj			   [in]	: 通道配置信息
* 返回值：  0=成功，非0 = 失败
********************************************************************/
ViStatus _VI_FUNC EphCAN_InitCAN(
	ViUInt32	cardnum,
	ViUInt16	ch,
	EP_CAN_INIT *pEpCanInit);

/********************************************************************
* 函数名称： EphCAN_GetChConfig
* 函数功能： 获取通道配置信息
* 参数：     cardnum	 [in]	 : 设备句柄
*			ch			[in]	: 通道号
*			pEpCanInit	[out]	: 通道配置信息
* 返回值：  0=成功，非0 = 失败
********************************************************************/
ViStatus _VI_FUNC EphCAN_GetChConfig(
	ViUInt32	cardnum,
	ViUInt16	ch,
	EP_CAN_INIT *pEpCanInit);

/********************************************************************
* 函数名称： EphCAN_TimeGetString
* 函数功能： 将CAN帧时间转换为字符串
* 参数       curtime	[in]	: CAN帧中时间信息
*			string		[out]	: 包含天、小时、分、秒、毫秒、微秒
* 返回值： 0 = 成功
********************************************************************/
ViStatus _VI_FUNC EphCAN_TimeGetString(
	ViUInt32	curtime,
	char		*string);

/********************************************************************
* 函数名称： EphCAN_GetFrameCount
* 函数功能： CAN数据帧读取函数
* 参数：
*			cardnum		[in]	: 设备句柄
*			ch			[in]	: 通道号
* 			frameCount	[out]	: 有多少帧未读出
* 返回值：  0=成功，非0=失败
********************************************************************/
ViStatus _VI_FUNC EphCAN_GetFrameCount(
	ViUInt32	cardnum,
	ViUInt16	ch,
	ViUInt32	*frameCount);

/********************************************************************
* 函数名称：     EphCAN_SetWorkMode
* 函数功能：     设置指定通道工作模式
* 参数：
*			    cardnum	[in]	: 模块句柄
				ch		[in]	: 通道号
				WorkMode[in]	: WM_NORMAL——正常模式  WM_SLEEP——休眠模式 WM_LOOP——环回模式  
									WM_LISTEN——监听模式  WM_CONFIG——配置模式
* 返回值：     0：表示成功，非0：表示错误
********************************************************************/
ViStatus _VI_FUNC EphCAN_SetWorkMode(
	ViUInt32	cardnum,
	ViUInt16	ch,
	ViUInt32	WorkMode);

/**********************************************************
* Function:		EphCAN_GetAllChTransStatus
* Description:	
* Parameters: 
*				cardnum	[in]	: 模块句柄
*				status	[out]	: 所有通道通信状态
* Return:		0: 表示成功，非0：表示错误
**********************************************************/
ViStatus _VI_FUNC EphCAN_GetAllChTransStatus(
	ViUInt32	cardnum,
	ViUInt32	*status);

/**********************************************************
* Function: EphCAN_GetChanelCount
* Description: 获取CAN模块通道个数
* Parameters: 
*			cardnum[in]: 模块句柄
*			chanelCount[out]: 获取通道个数
* Return: 0: 表示成功  非0：表示错误
**********************************************************/
ViStatus _VI_FUNC EphCAN_GetChanelCount(
	ViUInt32	cardnum,
	ViUInt32	*chanelCount);

// 2021.4.26
/**********************************************************
* Function: EphCAN_TransmitMulti
* Description: 一次发送多条CAN帧。
* Parameters: 
*			cardnum[in]: 模块句柄
*			count[in]: 发送CAN帧，一次不能大于32帧。
*			pSend[in]: 发送的CAN帧数据。
* Return: 0: 表示成功  非0：表示错误
**********************************************************/
ViStatus _VI_FUNC EphCAN_TransmitMulti (
	ViUInt32	cardnum,
	ViUInt32	count,
	EP_CAN_MULTI_OBJ	*pSend);

/**********************************************************
* Function: EphCAN_ReceiveMulti
* Description: 一次接收多条CAN帧。
* Parameters: 
*			cardnum[in]: 模块句柄
*			mode[in]: // 0=读通道0~3，1=读通道4~7
*			max_num[in]: 保留。接收CAN帧，一次不能大于8帧。
*			pReceive[out]: 接收的CAN帧数据。
*			act_num[out]: 实际接收的CAN帧，一次不能大于32帧。
* Return: 0: 表示成功  非0：表示错误
**********************************************************/
ViStatus _VI_FUNC EphCAN_ReceiveMulti (
	ViUInt32	cardnum,	
	ViUInt32	mode, 
	ViUInt32	max_num,
	EP_CAN_MULTI_OBJ	*pReceive,
	ViUInt32	*act_num);

#ifdef __cplusplus
}
#endif

#endif
