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
 * \file kernel_tpipe_base.h
 * \brief
 */
#ifndef ASCENDC_MODULE_TPIPE_BASE_H
#define ASCENDC_MODULE_TPIPE_BASE_H
#include "kernel_tensor_impl.h"
#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_common_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_common_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_common_impl.h"
#include "dav_c220/kfc/kfc_comm.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_common_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m300/kernel_operator_common_impl.h"
#endif

namespace AscendC {
// begin base define of tquebind
template <int depth>
struct TBufHandleAux {
    using T = TBufHandle[depth];
};

template <>
struct TBufHandleAux<1> {
    using T = TBufHandle;
};
constexpr TEventID INVALID_TEVENTID = (static_cast<TEventID>(-1));

// begin base define of tpipe
struct TEventPool {
    uint64_t eventOccupy;
};

struct TPipeBufPool {
    uint32_t maxAddr;
};

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
struct BufPoolExtra {
    uint8_t* absAddr;
    uint32_t phySpace;
};
#endif

struct TShareBuf {
    enum class ShareHard : uint8_t {  // Redefine to save resources
        L1 = 0,
        L0C = 1,
        UB = 2,
        MAX,
    };
    int32_t start[static_cast<uint8_t>(ShareHard::MAX)];
    int32_t maxAddr[static_cast<uint8_t>(ShareHard::MAX)];
    DEBUG_CODE(uint32_t length[static_cast<uint8_t>(ShareHard::MAX)]);
};

struct SpmInfo {
    uint64_t spmAddr;
    int32_t spmBuffSize;
    uint8_t spmBufType;
};

struct TPipeImpl {
    struct TEventPool eventPool_[EVENT_NUM];
    struct TPipeBufPool bufPool_[static_cast<uint8_t>(Hardware::MAX)];
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    BufPoolExtra bufPoolBaseAddr_[(uint8_t)Hardware::MAX];
#endif
    struct TBufType buf_[QBUF_MAX_LEN];
    TShareBuf shareBufPool_;
    SpmInfo spmInfo_;
    // the tscm buffer addr
    uint32_t tscmBufferPtr_;
    uint8_t curBufSize_;
    bool isDestroy;
};

constexpr uint32_t defaultBufIDSize = 4;

template <uint32_t bufIDSize = defaultBufIDSize>
struct TBufPoolImpl {
    struct TBufType buf_[bufIDSize];
    uint32_t startAddr_;
    uint32_t maxAddr_;
    uint32_t maxLen_;
    uint8_t curBufSize_;
    uint8_t isReset_;
};

class TPipeBase {
public:
    __aicore__ inline void InitShareBufStart(uint32_t mode, uint32_t* shareLens, uint32_t lens, uint8_t subBlockIdx);
    __aicore__ inline void InitShareBufEnd();

protected:
    TPipeImpl g_tpipeImpl;
    __aicore__ inline void AuxShareBufStart(uint32_t mode, uint32_t* shareLens, uint8_t pos, Hardware hard,
                                            uint8_t subBlockIdx);
};

__aicore__ inline void TPipeBase::InitShareBufStart(uint32_t mode, uint32_t* shareLens, uint32_t lens,
                                                    uint8_t subBlockIdx)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASCENDC_ASSERT((lens == static_cast<uint32_t>(TShareBuf::ShareHard::MAX)), {
        KERNEL_LOG(KERNEL_ERROR, "lens is %u, which should be %u", lens,
                   static_cast<uint32_t>(TShareBuf::ShareHard::MAX));
    });
#else
    (void)(lens);
#endif

    ASCENDC_ASSERT((subBlockIdx == 0 || subBlockIdx == 1),
                   { KERNEL_LOG(KERNEL_ERROR, "subBlockIdx is %d, which should only be 0/1", subBlockIdx); });
    AuxShareBufStart(mode, shareLens, static_cast<uint8_t>(TShareBuf::ShareHard::L1), Hardware::L1, subBlockIdx);
    AuxShareBufStart(mode, shareLens, static_cast<uint8_t>(TShareBuf::ShareHard::L0C), Hardware::L0C, subBlockIdx);
#if __CCE_AICORE__ < 220
    AuxShareBufStart(mode, shareLens, static_cast<uint8_t>(TShareBuf::ShareHard::UB), Hardware::UB, subBlockIdx);
#endif
    this->g_tpipeImpl.bufPool_[static_cast<uint8_t>(Hardware::L0A)].maxAddr = 0;
    this->g_tpipeImpl.bufPool_[static_cast<uint8_t>(Hardware::L0B)].maxAddr = 0;
    // v100 Shouldn't Use Bias Table
    this->g_tpipeImpl.bufPool_[static_cast<uint8_t>(Hardware::BIAS)].maxAddr = 0;

    return;
}

__aicore__ inline void TPipeBase::InitShareBufEnd()
{
    // debug methods need to be added.
    this->g_tpipeImpl.bufPool_[static_cast<uint8_t>(Hardware::L1)].maxAddr =
        g_tpipeImpl.shareBufPool_.maxAddr[static_cast<uint8_t>(TShareBuf::ShareHard::L1)];
    this->g_tpipeImpl.bufPool_[static_cast<uint8_t>(Hardware::L0C)].maxAddr =
        g_tpipeImpl.shareBufPool_.maxAddr[static_cast<uint8_t>(TShareBuf::ShareHard::L0C)];
#if __CCE_AICORE__ < 220
    this->g_tpipeImpl.bufPool_[static_cast<uint8_t>(Hardware::UB)].maxAddr =
        g_tpipeImpl.shareBufPool_.maxAddr[static_cast<uint8_t>(TShareBuf::ShareHard::UB)];
#endif

    return;
}

__aicore__ inline void TPipeBase::AuxShareBufStart(uint32_t mode, uint32_t* shareLens, uint8_t pos, Hardware hard,
                                                   uint8_t subBlockIdx)
{
    uint8_t hardU8 = static_cast<uint8_t>(hard);
    if (unlikely(g_tpipeImpl.shareBufPool_.start[pos] == -1)) {  // The address has not been initialized.
        // Record the maximum allocated address.
        g_tpipeImpl.shareBufPool_.start[pos] = this->g_tpipeImpl.bufPool_[hardU8].maxAddr;
        g_tpipeImpl.shareBufPool_.maxAddr[pos] = g_tpipeImpl.shareBufPool_.start[pos] + shareLens[pos];
        DEBUG_CODE(g_tpipeImpl.shareBufPool_.length[pos] = shareLens[pos]);
    } else {
        DEBUG_CODE(g_tpipeImpl.shareBufPool_.length[pos] = g_tpipeImpl.shareBufPool_.length[pos] > shareLens[pos] ?
                                                               g_tpipeImpl.shareBufPool_.length[pos] :
                                                               shareLens[pos]);
        // Record the maximum allocated address.
        g_tpipeImpl.shareBufPool_.maxAddr[pos] = this->g_tpipeImpl.bufPool_[hardU8].maxAddr;
        g_tpipeImpl.bufPool_[hardU8].maxAddr = g_tpipeImpl.shareBufPool_.start[pos];  // Reset resource start position.
    }

    if (mode == 1 && subBlockIdx == 1) {
        this->g_tpipeImpl.bufPool_[hardU8].maxAddr += shareLens[pos] / HALF_FACTOR;  // Reset resource start position.
    }

    ASCENDC_ASSERT((g_tpipeImpl.shareBufPool_.length[pos] >= shareLens[pos]), {
        KERNEL_LOG(KERNEL_ERROR, "share buf addr is %u, exceed limits %u", shareLens[pos],
                   g_tpipeImpl.shareBufPool_.length[pos]);
    });
}

}  // namespace AscendC
#endif  // ASCENDC_MODULE_TPIPE_BASE_H