/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: lookup device capacity
 */

#ifndef HCCL_DEVICE_CAPACITY_H
#define HCCL_DEVICE_CAPACITY_H

#include "hccl/base.h"
#include "dtype_common.h"
#include "hccl_common.h"


namespace hccl {
// 节点间RDMA发送数据单个WQE支持的最大数据量
const u64 RDMA_SEND_MAX_SIZE = 0x80000000;
// 节点内单个SDMA任务发送数据支持的最大数据量
const u64 SDMA_SEND_MAX_SIZE = 0x100000000;
    bool IsSupportAIVCopy(HcclDataType dataType);
    bool IsSupportAIVReduce(HcclDataType dataType, HcclReduceOp op);
    bool IsSupportSDMAReduce(const void *inputPtr, const void *outputPtr, HcclDataType dataType, HcclReduceOp op);
    bool IsSupportRDMAReduce(HcclDataType dataType, HcclReduceOp op);
    HcclResult GetBandWidthPerNPU(u32 level, u32 userRankSize, u32 deviceNumPerAggregation, float &bandWidth);
    HcclResult CheckDeviceType(const DevType deviceType);
    bool IsOverFlowInfNanMode();
    bool Is310PDevice();
    bool IsUseSdidForDeviceId(const u32 superDeviceId = INVALID_UINT);  // deprecated
    HcclResult IsSuperPodMode(bool &useSuperPodMode);
    bool IsSupportRDMALite(const s32 deviceLogicId);                    // 是否支持rdma lite
    HcclResult GetMemBlockNum(const u32 devicePhyId, u32& memBlockNum);
    HcclResult IsSupportAicpuNormalQP(const u32& devicePhyId, bool &isSupportNormalQP); // 是否支持AICPU的Normal QP
}

#endif // end HCCL_DEVICE_CAPACITY_H
