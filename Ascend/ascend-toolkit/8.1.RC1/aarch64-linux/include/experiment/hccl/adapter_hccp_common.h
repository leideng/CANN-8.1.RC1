/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: adapter层重构，hccp common接口
 */

#ifndef HCCL_INC_ADAPTER_HCCP_COMMON_H
#define HCCL_INC_ADAPTER_HCCP_COMMON_H

#include "hccl_common.h"
#include "hccl_ip_address.h"

#if T_DESC("ip管理", true)

enum DeviceIdType {
    DEVICE_ID_TYPE_PHY_ID = 0,
    DEVICE_ID_TYPE_SDID
};

#define SOCK_CONN_TAG_SIZE 192
struct SocketWlistInfo {
    union hccl::HcclInAddr remoteIp; /**< IP address of remote */
    unsigned int connLimit; /**< limit of whilte list */
    char tag[SOCK_CONN_TAG_SIZE]; /**< tag used for whitelist must ended by '\0' */
};

struct SocketEventInfo {
    u32 event;
    FdHandle fdHandle;
};

enum class HcclEpollEvent {
    HCCL_EPOLLIN = 0,
    HCCL_EPOLLOUT,
    HCCL_EPOLLPRI,
    HCCL_EPOLLERR,
    HCCL_EPOLLHUP,
    HCCL_EPOLLET,
    HCCL_EPOLLONESHOT,
    HCCL_EPOLLOUT_LET_ONESHOT,
    HCCL_EPOLLINVALD
};

HcclResult hrtRaGetSingleSocketVnicIpInfo(u32 phy_id, DeviceIdType deviceIdType, u32 deviceId,
    hccl::HcclIpAddress &vnicIP);
HcclResult hrtGetHostIf(
    std::vector<std::pair<std::string, hccl::HcclIpAddress>> &hostIfs, u32 devPhyId = 0); // key: if name, value ip addr
HcclResult hrtRaGetDeviceIP(u32 devicePhyId, std::vector<hccl::HcclIpAddress> &ipAddr);
HcclResult hrtRaGetDeviceAllNicIP(std::vector<std::vector<hccl::HcclIpAddress>> &ipAddr);
HcclResult GetIsSupSockBatchCloseImmed(u32 phyId, bool& isSupportBatchClose);

HcclResult hrtRaCreateEventHandle(s32 &eventHandle);
HcclResult hrtRaCtlEventHandle(s32 eventHandle, const FdHandle fdHandle, int opCode, HcclEpollEvent event);

HcclResult hrtRaWaitEventHandle(s32 eventHandle, std::vector<SocketEventInfo> &eventInfos, s32 timeOut,
    u32 maxEvents, u32 &eventsNum);

HcclResult hrtRaDestroyEventHandle(s32 &eventHandle);
#endif

#endif