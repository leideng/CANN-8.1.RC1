/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: remote_access.h
 */

#ifndef REMOTE_ACCESS_H
#define REMOTE_ACCESS_H

#include <memory>
#include <vector>
#include "base.h"
#include "hccl_common.h"

namespace hccl {
class RemoteAccessImpl;

// RemoteAccess的ranktable格式
using RmaRankTable = struct tagRmaRankTable {
    u32 serverNum { 0 };                    // 集群内服务器总数
    u32 rankNum { 0 };                      // 通信域内的rank总数
    s32 devicePhyId { 0 };                  // 服务器内device唯一标识
    std::vector<std::vector<HcclIpAddress>> deviceIps;    // 所有rank的device 对应的网卡ip
};

class RemoteAccess {
public:
    explicit RemoteAccess();
    virtual ~RemoteAccess();
    HcclResult Init(u32 rank, const std::vector<MemRegisterAddr>& addrInfos, const RmaRankTable &rankTable);
    HcclResult RemoteRead(const std::vector<HcomRemoteAccessAddrInfo>& addrInfos, HcclRtStream stream);
    HcclResult RemoteWrite(const std::vector<HcomRemoteAccessAddrInfo>& addrInfos, HcclRtStream stream);
private:
    std::unique_ptr<RemoteAccessImpl> impl_;
};
}

#endif  // REMOTE_ACCESS_H