/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: nic port context.
 */

#ifndef HCCL_NETWORK_PUB_H
#define HCCL_NETWORK_PUB_H

#include "hccl_common.h"
#include "hccl_ip_address.h"

using HcclNetDevCtx = void *;

struct HcclNetDevInfo {
    s32 devicePhyId;
    s32 deviceLogicId;
    u32 superDeviceId;
    u32 rsvd;
};

enum class NicType {
    VNIC_TYPE = 0,
    DEVICE_NIC_TYPE,
    HOST_NIC_TYPE
};

HcclResult HcclNetInit(NICDeployment nicDeploy, s32 devicePhyId, s32 deviceLogicId, 
    bool enableWhitelistFlag, bool hasBackup = false);
HcclResult HcclNetDeInit(NICDeployment nicDeploy, s32 devicePhyId, s32 deviceLogicId,
    bool hasBackup = false);

HcclResult HcclNetOpenDev(
    HcclNetDevCtx *netDevCtx, NicType nicType, s32 devicePhyId, s32 deviceLogicId, hccl::HcclIpAddress localIp, 
    hccl::HcclIpAddress backupIp = hccl::HcclIpAddress(0));
void HcclNetCloseDev(HcclNetDevCtx netDevCtx);

HcclResult HcclNetDevGetNicType(HcclNetDevCtx netDevCtx, NicType *nicType);
HcclResult HcclNetDevGetLocalIp(HcclNetDevCtx netDevCtx, hccl::HcclIpAddress &localIp);
HcclResult HcclNetDevGetPortStatus(HcclNetDevCtx netDevCtx, bool &portStatus);
#endif
