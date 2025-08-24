/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: adapter层重构，rts common接口
 */

#ifndef HCCL_INC_ADAPTER_RTS_COMMON_H
#define HCCL_INC_ADAPTER_RTS_COMMON_H

#include "hccl_common.h"
#include "dtype_common.h"
#include "runtime/kernel.h"

#if T_DESC("stream管理", true)
HcclResult hcclStreamSynchronize(HcclRtStream stream);
HcclResult hrtStreamSetMode(HcclRtStream stream, const uint64_t stmMode);
HcclResult hrtStreamGetMode(HcclRtStream const stream, uint64_t *const stmMode);
HcclResult hrtGetStreamId(HcclRtStream stream, s32 &streamId);
HcclResult hrtStreamActive(HcclRtStream activeStream, HcclRtStream stream);
#endif

HcclResult hrtCtxGetCurrent(HcclRtContext *ctx);
HcclResult hrtCtxSetCurrent(HcclRtContext ctx);
HcclResult hrtEventCreateWithFlag(HcclRtEvent *evt);
HcclResult hrtGetEventID(HcclRtEvent event, uint32_t *eventId);
HcclResult hrtNotifyGetPhyInfo(HcclRtNotify notify, uint32_t *phyDevId, uint32_t *tsId);
HcclResult hrtGetNotifyID(HcclRtNotify signal, u32 *notifyID);
HcclResult hrtNotifyReset(rtNotify_t notify);

#if T_DESC("Device管理", true)
HcclResult hrtResetDevice(s32 deviceLogicId);
HcclResult hrtSetDevice(s32 deviceLogicId);
HcclResult hrtGetPairDeviceLinkType(u32 phyDevId, u32 otherPhyDevId, LinkTypeInServer &linkType);

enum class HcclReduceType {
    HCCL_INLINE_REDUCE = 0,
    HCCL_TBE_REDUCE
};

enum class HcclRtMemcpyKind {
    HCCL_RT_MEMCPY_KIND_HOST_TO_HOST = 0, /**< host to host */
    HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE,   /**< host to device */
    HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST,   /**< device to host */
    HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, /**< device to device */
    HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, /**< Level-2 address copy, device to device */
    HCCL_RT_MEMCPY_KIND_RESERVED,
};

enum class HcclRtDeviceModuleType {
    HCCL_RT_MODULE_TYPE_SYSTEM = 0,  /**< system info*/
    HCCL_RT_MODULE_TYPE_AICORE,      /**< AI CORE info*/
    HCCL_RT_MODULE_TYPE_VECTOR_CORE, /**< VECTOR CORE info*/
    HCCL_RT_DEVICE_MOUDLE_RESERVED,
};

enum class HcclRtDeviceInfoType {
    HCCL_INFO_TYPE_CORE_NUM,
    HCCL_INFO_TYPE_PHY_CHIP_ID,
    HCCL_INFO_TYPE_SDID,
    HCCL_INFO_TYPE_SERVER_ID,
    HCCL_INFO_TYPE_SUPER_POD_ID,
    HCCL_RT_DEVICE_INFO_RESERVED,
};


#ifdef __cplusplus
extern "C" {
#endif
HcclResult hrtGetDevice(s32 *deviceLogicId);
HcclResult hrtGetDeviceRefresh(s32 *deviceLogicId);
HcclResult hrtGetDeviceCount(s32 *count);
HcclResult hrtGetDeviceInfo(u32 deviceId, HcclRtDeviceModuleType hcclModuleType,
    HcclRtDeviceInfoType hcclInfoType, s64 &val);
HcclResult hrtGetDeviceType(DevType &devType);
HcclResult hrtSetlocalDeviceType(DevType devType);
HcclResult hrtSetlocalDevice(s32 deviceLogicId);
HcclResult hrtSetLocalDeviceSatMode(rtFloatOverflowMode_t floatOverflowMode);
HcclResult hrtSetWorkModeAicpu(bool workModeAicpu);
HcclResult hrtGetDeviceSatMode(rtFloatOverflowMode_t *floatOverflowMode);
HcclResult hrtGetDevicePhyIdByIndex(u32 deviceLogicId, u32 &devicePhyId, bool isRefresh = false);
HcclResult hrtGetDeviceIndexByPhyId(u32 devicePhyId, u32 &deviceLogicId);
HcclResult hrtGetPairDevicePhyId(u32 localDevPhyId, u32 &pairDevPhyId);
HcclResult PrintMemoryAttr(const void *memAddr);
HcclResult hrtCtxGetOverflowAddr(void **overflowAddr);

HcclResult hrtEventDestroy(HcclRtEvent event);
HcclResult hrtMalloc(void **devPtr, u64 size, bool level2Address = false);
HcclResult hrtFree(void *devPtr);
HcclResult hrtMemSet(void *dst, uint64_t destMax, uint64_t count);
HcclResult hrtMemSyncCopy(void *dst, uint64_t destMax, const void *src, uint64_t count, HcclRtMemcpyKind kind);
HcclResult hrtMemAsyncCopyByQos(void *dst, uint64_t destMax, const void *src, uint64_t count,
    HcclRtMemcpyKind kind, rtStream_t stream, uint32_t qosCfg);
HcclResult hrtMemAsyncCopy(void *dst, uint64_t destMax, const void *src, uint64_t count,
    HcclRtMemcpyKind kind, rtStream_t stream);
HcclResult hrtMemcpyAddrAsync(void *dst, uint64_t destMax, uint64_t destOffset, const void *src, uint64_t count,
    uint64_t srcOffset, rtStream_t stream);

HcclResult hrtAicpuKernelLaunchExWithArgs(uint32_t kernelType, const char *opName, uint32_t blockDim,
    const rtAicpuArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
    rtStream_t stream, uint32_t flags);
#if T_DESC("RtsTaskCallBack", true)
HcclResult hrtSubscribeReport(u64 threadId, rtStream_t &stream);
HcclResult hrtProcessReport(s32 timeout);
HcclResult hrtTaskAbortHandleCallback(rtTaskAbortCallBack callback, void *args);
HcclResult hrtResourceClean(int32_t devId, rtIdType_t type);
HcclResult hrtGetHccsPortNum(u32 deviceLogicId, s32 &num);
#endif

HcclResult hrtGetTaskIdAndStreamID(u32 &taskId, u32 &streamId);
 
#if T_DESC("RtsTaskExceptionHandler", true)
HcclResult hrtRegTaskFailCallbackByModule(rtTaskFailCallback callback);
HcclResult hrtGetMaxStreamAndTask(u32 &maxStrCount, u32 &maxTaskCount);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
#endif