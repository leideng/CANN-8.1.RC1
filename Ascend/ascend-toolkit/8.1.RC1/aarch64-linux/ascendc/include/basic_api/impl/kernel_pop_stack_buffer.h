/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file kernel_pop_stack_buffer.h
 * \brief
 */
#ifndef TIKCFW_IMPL_KERNEL_POP_STACK_BUFFER_H
#define TIKCFW_IMPL_KERNEL_POP_STACK_BUFFER_H

#include "kernel_tpipe_impl.h"
namespace AscendC {
template <TPosition pos> __aicore__ inline uint64_t GetEndAddress()
{
    Hardware hardType = GetPhyType(pos);
    ASCENDC_ASSERT((hardType == Hardware::UB), { KERNEL_LOG(KERNEL_ERROR, "hardType should be UB"); });
    // the last 64B reserved for ub kfc msg send
#if __CCE_AICORE__ == 220
    return TOTAL_UB_SIZE - sizeof(KfcMsg);
#else
    return TOTAL_UB_SIZE;
#endif
}

template <typename T, TPosition pos> __aicore__ inline bool PopStackBuffer(LocalTensor<T>& popLocal)
{
    TBuffAddr addr;
    addr.logicPos = (int8_t)pos;
    ASCENDC_ASSERT((GetTPipePtr() != nullptr), { KERNEL_LOG(KERNEL_ERROR, "GetTPipePtr is nullptr"); });
    uint64_t endAddress = GetEndAddress<pos>();
    uint64_t queEndAddress = GetTPipePtr()->GetQueueEndAddress<pos>();
    ASCENDC_ASSERT((queEndAddress % ONE_BLK_SIZE == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "queEndAddress is %lu, which must be 32B aligned", queEndAddress); });
    addr.dataLen = (uint32_t)(endAddress - queEndAddress);
    addr.bufferAddr = queEndAddress;
#ifdef ASCENDC_CPU_DEBUG
    auto absAddr = GetTPipePtr()->GetBaseAddr(static_cast<int8_t>(pos));
    addr.absAddr = absAddr + addr.bufferAddr;
    AscendCBufInit(static_cast<uint8_t>(pos), static_cast<uint8_t>(1), static_cast<uint8_t>(1), (uint64_t)addr.absAddr,
        static_cast<uint64_t>(addr.dataLen));
    AscendCBufInit(static_cast<uint8_t>(pos), static_cast<uint8_t>(1), static_cast<uint8_t>(1),
        (uint64_t)(absAddr + endAddress), static_cast<uint64_t>(ONE_BLK_SIZE << 1));
#endif // ASCENDC_CPU_DEBUG
    popLocal.SetAddr(addr);
    return true;
}

template <TPosition pos> __aicore__ inline bool PopStackBuffer(TBuf<pos>& popBuffer, TBufType& bufStart)
{
    uint64_t endAddress = GetEndAddress<pos>();
    uint64_t queEndAddress = GetTPipePtr()->GetQueueEndAddress<pos>();
    ASCENDC_ASSERT((queEndAddress % ONE_BLK_SIZE == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "queEndAddress is %lu, which must be 32B aligned", queEndAddress); });
    uint32_t dataLen = (uint32_t)(endAddress - queEndAddress);
    bufStart.address = queEndAddress;
    bufStart.dataLen = dataLen;
    popBuffer.SetTpipeBuf(&bufStart, dataLen);
#ifdef ASCENDC_CPU_DEBUG
    auto absAddr = GetTPipePtr()->GetBaseAddr(static_cast<int8_t>(pos));
    AscendCBufInit(static_cast<uint8_t>(pos), static_cast<uint8_t>(1), static_cast<uint8_t>(1),
        (uint64_t)(absAddr + queEndAddress), static_cast<uint64_t>(dataLen));
    AscendCBufInit(static_cast<uint8_t>(pos), static_cast<uint8_t>(1), static_cast<uint8_t>(1),
        (uint64_t)(absAddr + endAddress), static_cast<uint64_t>(ONE_BLK_SIZE << 1));
#endif
    return true;
}
} // namespace AscendC
#endif
