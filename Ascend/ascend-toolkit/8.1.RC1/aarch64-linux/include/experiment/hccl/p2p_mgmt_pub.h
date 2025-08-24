/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description:
 */

#ifndef P2P_MGMT_PUB_H
#define P2P_MGMT_PUB_H

#include <map>
#include <mutex>
#include <vector>
#include <atomic>
#include "hccl/hccl_types.h"

namespace hccl {
enum class P2PStatus {
    P2P_STATUS_DISABLED = 0,
    P2P_STATUS_ENABLING,
    P2P_STATUS_ENABLED
};

using P2PConnectionInfo = struct P2PConnectionInfoDef {
    uint32_t reference = 0;
    P2PStatus status = P2PStatus::P2P_STATUS_DISABLED;
};

class P2PMgmt;
class P2PMgmtPub {
public:
    static HcclResult EnableP2P(std::vector<uint32_t> remoteDevices);
    static HcclResult DisableP2P(std::vector<uint32_t> remoteDevices);
    static HcclResult WaitP2PEnabled(std::vector<uint32_t> remoteDevices,
        std::function<bool()> needStop = []() { return false; });
};
} // namespace hccl

#endif // P2P_MGMT_H