/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_impl.h
 * \brief
 */
#ifndef IMPL_HCCL_HCCL_IMPL_H
#define IMPL_HCCL_HCCL_IMPL_H
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
#include "hccl_v220_impl.h"
#endif  // IMPL_HCCL_HCCL_V220_IMPL_H
#include "hccl_impl_dfx.h"
namespace AscendC {
template <HcclServerType serverType, const auto &config>
template <bool commit>
__aicore__ inline HcclHandle Hccl<serverType, config>::AllReduce(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t count,
                                                         HcclDataType dataType, HcclReduceOp op, uint8_t repeat)
{
    DFX_MAKE_GUARD(HcclApiOperType::ALL_REDUCE_PREPARE, GetRankDim(), repeat, count, dataType,
                   sendBuf, 0U, recvBuf, 0U);
    return impl_.template AllReduce<commit>(sendBuf, recvBuf, count, dataType, op, repeat);
}

template <HcclServerType serverType, const auto &config>
template <bool commit>
__aicore__ inline HcclHandle Hccl<serverType, config>::AllGather(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t sendCount,
                                                         HcclDataType dataType, uint64_t strideCount, uint8_t repeat)
{
    DFX_MAKE_GUARD(HcclApiOperType::ALL_GATHER_PREPARE, GetRankDim(), repeat, sendCount, dataType,
                   sendBuf, 0U, recvBuf, strideCount);
    return impl_.template AllGather<commit>(sendBuf, recvBuf, sendCount, dataType, strideCount, repeat);
}

template <HcclServerType serverType, const auto &config>
template <bool commit>
__aicore__ inline HcclHandle Hccl<serverType, config>::AlltoAll(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t dataCount,
                                                        HcclDataType dataType, uint64_t strideCount, uint8_t repeat)
{
    DFX_MAKE_GUARD(HcclApiOperType::ALL_TO_ALL_PREPARE, GetRankDim(), repeat, dataCount, dataType,
                   sendBuf, strideCount, recvBuf, strideCount);
    return impl_.template AlltoAll<commit>(sendBuf, recvBuf, dataCount, dataType, strideCount, repeat);
}

template <HcclServerType serverType, const auto &config>
template <bool commit>
__aicore__ inline HcclHandle
Hccl<serverType, config>::AlltoAllV(GM_ADDR sendBuf, void *sendCounts, void *sdispls, HcclDataType sendType,
                            GM_ADDR recvBuf, void *recvCounts, void *rdispls, HcclDataType recvType,
                            uint8_t repeat)
{
    DFX_MAKE_GUARD(HcclApiOperType::ALL_TO_ALL_V_PREPARE, GetRankDim(), repeat,
                   sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType);
    return impl_.template AlltoAllV<commit>(sendBuf, sendCounts, sdispls, sendType,
                                            recvBuf, recvCounts, rdispls, recvType, repeat);
}

template <HcclServerType serverType, const auto &config>
template <bool commit>
__aicore__ inline HcclHandle
Hccl<serverType, config>::ReduceScatter(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t recvCount, HcclDataType dataType,
                                HcclReduceOp op, uint64_t strideCount, uint8_t repeat)
{
    DFX_MAKE_GUARD(HcclApiOperType::REDUCE_SCATTER_PREPARE, GetRankDim(), repeat, recvCount, dataType,
                   sendBuf, strideCount, recvBuf, 0U);
    return impl_.template ReduceScatter<commit>(sendBuf, recvBuf, recvCount, dataType, op, strideCount,
                                                repeat);
}

template <HcclServerType serverType, const auto &config>
template <bool commit>
__aicore__ inline HcclHandle Hccl<serverType, config>::BatchWrite(GM_ADDR batchWriteInfo, uint32_t itemNum)
{
    DFX_MAKE_GUARD(HcclApiOperType::BATCH_WRITE_PREPARE);
    return impl_.template BatchWrite<commit>(batchWriteInfo, itemNum);
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline void Hccl<serverType, config>::Init(GM_ADDR context, __gm__ void *initTiling)
{
    DFX_MAKE_GUARD(HcclApiOperType::INIT);
    impl_.Init(context, initTiling);
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline int32_t Hccl<serverType, config>::SetCcTiling(__gm__ void *ccOpTilingData)
{
    DFX_MAKE_GUARD(HcclApiOperType::SET_CCTILING);
    return impl_.SetCcTiling(ccOpTilingData);
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline void Hccl<serverType, config>::Commit(HcclHandle handleId)
{
    DFX_MAKE_GUARD(HcclApiOperType::COMMIT);
    impl_.Commit(handleId);
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline int32_t Hccl<serverType, config>::Wait(HcclHandle handleId)
{
    DFX_MAKE_GUARD(HcclApiOperType::WAIT);
    return impl_.Wait(handleId);
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline int32_t Hccl<serverType, config>::Query(HcclHandle handleId)
{
    DFX_MAKE_GUARD(HcclApiOperType::QUERY);
    return impl_.Query(handleId);
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline void Hccl<serverType, config>::InterHcclGroupSync(int8_t srcGroupID, HcclHandle srcHandleID)
{
    DFX_MAKE_GUARD(HcclApiOperType::GROUP_SYNC);
    impl_.InterHcclGroupSync(srcGroupID, srcHandleID);
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline void Hccl<serverType, config>::Finalize()
{
    DFX_MAKE_GUARD(HcclApiOperType::FINALIZE);
    impl_.Finalize();
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline GM_ADDR Hccl<serverType, config>::GetWindowsInAddr(uint32_t rankId)
{
    return impl_.GetWindowsInAddr(rankId);
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline GM_ADDR Hccl<serverType, config>::GetWindowsOutAddr(uint32_t rankId)
{
    return impl_.GetWindowsOutAddr(rankId);
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline uint32_t Hccl<serverType, config>::GetRankId()
{
    return impl_.GetRankId();
}

template <HcclServerType serverType, const auto &config>
__aicore__ inline uint32_t Hccl<serverType, config>::GetRankDim()
{
    return impl_.GetRankDim();
}
}  // namespace AscendC

#endif  // IMPL_HCCL_HCCL_IMPL_H
