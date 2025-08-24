/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TRANSPORT_COMMON_H
#define TRANSPORT_COMMON_H

#include <vector>
#include <map>
#include "transport_pub.h"
#include "hccl_common.h"

namespace hccl {
constexpr u32 HCCL_RANK_SIZE_EQ_ONE = 1;
using RankId = u32;

enum class TransportStatus {
    INIT,
    READY,
    STOP
};

enum TransportMemType {
    CCL_INPUT = 0,
    CCL_OUTPUT,
    SCRATCH,
    PARAM_INPUT,
    PARAM_OUTPUT,
    AIV_INPUT,
    AIV_OUTPUT,
    RESERVED
};

typedef enum {
    COMM_LEVEL0 = 0,    // 一级通信域(server内)
    COMM_LEVEL0_ANYPATH_RDMA,  // anypath特性使用
    COMM_LEVEL1,        // 二级通信域(server间)
    COMM_LEVEL1_ANYPATH_RDMA, // anypath特性使用
    COMM_LEVEL1_AHC,    // AHC 二级通信域(server间)
    COMM_LEVEL2,        // 三级通信域(超节点间)
    COMM_MESH_L0,       // mesh内
    COMM_MESH_L1,       // mesh间
    COMM_COMBINE,       // 打平通信域，大ring环
    COMM_COMBINE_ORDER, // 打平通信域，按rank排序
    COMM_LEVEL0_ANYPATH_SDMA,  // anypath特性使用
    COMM_LEVEL1_ANYPATH_SDMA, // anypath特性使用
    COMM_LEVEL_RESERVED,
} CommPlane;

enum class CommType {
    COMM_TAG_RING_INNER = 0,
    COMM_TAG_RING_COMBINED,
    COMM_TAG_HALVING_DOUBLING,
    COMM_TAG_STAR,
    COMM_TAG_NONUNIFORM_HIERARCHICAL_RING,
    COMM_TAG_WHOLE_NHR,
    COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1,
    COMM_TAG_WHOLE_NHR_V1,
    COMM_TAG_ASYMMETRIC_HIERARCHICAL_CONCATENATE,
    COMM_TAG_WHOLE_AHC,
    COMM_TAG_ASYMMETRIC_HIERARCHICAL_CONCATENATE_BROKE,
    COMM_TAG_WHOLE_AHC_BROKE,
    COMM_TAG_NONUNIFORM_BRUCK,
    COMM_TAG_WHOLE_NB,
    COMM_TAG_MESH_COMBINED,
    COMM_TAG_MESH,
    COMM_TAG_P2P,
    COMM_TAG_PARTIAL_MESH_COMBINED,
    COMM_TAG_MAX,
};

struct TransportRequest {
    bool isValid = false;
    RankId localUserRank = 0;
    RankId remoteUserRank = 0;
    TransportMemType inputMemType = TransportMemType::RESERVED;
    TransportMemType outputMemType = TransportMemType::RESERVED;
    bool isUsedRdma = false;
    u32 notifyNum = 0;
};

struct SingleSubCommTransport {
    std::vector<TransportRequest> transportRequests;
    std::vector<LINK> links;
    std::vector<TransportStatus> status; // 代表该transport是否ready, stop后为stop, 建链后为ready
    u64 taskNum = 0;
    std::map<u32, u32> userRank2subCommRank;
    std::map<u32, u32> subCommRank2UserRank;
    bool supportDataReceivedAck = false;
    LinkMode linkMode = LinkMode::LINK_DUPLEX_MODE;
    bool enableUseOneDoorbell = false;
    bool needVirtualLink =false; // for alltoall 多线程性能提升使用
    std::vector<LINK> virtualLinks; // for alltoall 多线程性能提升使用
};

// 通信域建链信息
struct CommParaInfo {
    CommPlane commPlane = COMM_LEVEL_RESERVED;
    CommType commType = CommType::COMM_TAG_MAX;
    u32 root = INVALID_VALUE_RANKID;
    u32 peerUserRank = INVALID_VALUE_RANKID;
    bool isAicpuModeEn = false;
    bool meshSinglePlane = false;
    std::set<u32> batchSendRecvtargetRanks;
    bool forceRdma = false;

    CommParaInfo() {}
    CommParaInfo (CommPlane commPlane, CommType commType, u32 root = INVALID_VALUE_RANKID,
        u32 peerUserRank = INVALID_VALUE_RANKID, bool isAicpuModeEn = false, bool meshSinglePlane = false,
        std::set<u32> batchSendRecvtargetRanks = std::set<u32>(), bool forceRdma = false)
        : commPlane(commPlane), commType(commType), root(root), peerUserRank(peerUserRank),
        isAicpuModeEn(isAicpuModeEn), meshSinglePlane(meshSinglePlane),
        batchSendRecvtargetRanks(batchSendRecvtargetRanks), forceRdma(forceRdma)
    {
    }
};
}
#endif