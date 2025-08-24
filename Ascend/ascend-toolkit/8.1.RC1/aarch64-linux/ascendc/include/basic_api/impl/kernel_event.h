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
 * \file kernel_event.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_EVENT_IMPL_H
#define ASCENDC_KERNEL_EVENT_IMPL_H

#include "kernel_macros.h"
#include "kernel_log.h"
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include <cstdint>
#include "stub_def.h"
#include "stub_fun.h"
#endif

namespace AscendC {
enum class TPosition : uint8_t {
    GM,
    A1,
    A2,
    B1,
    B2,
    C1,
    C2,
    CO1,
    CO2,
    VECIN,
    VECOUT,
    VECCALC,
    LCM = VECCALC,
    SPM,
    SHM = SPM,
    TSCM,
    C2PIPE2GM,
    C2PIPE2LOCAL,
    MAX,
};

using QuePosition = TPosition;
enum class Hardware : uint8_t { GM, UB, L1, L0A, L0B, L0C, BIAS, FIXBUF, MAX };

enum class HardEvent : uint8_t {
    // src_dst
    MTE2_MTE1,
    MTE1_MTE2,
    MTE1_M,
    M_MTE1,
    MTE2_V,
    V_MTE2,
    MTE3_V,
    V_MTE3,
    M_V,
    V_M,
    V_V,
    MTE3_MTE1,
    MTE1_MTE3,
    MTE1_V,
    MTE2_M,
    M_MTE2,
    V_MTE1,
    M_FIX,
    FIX_M,
    MTE3_MTE2,
    MTE2_MTE3,
    S_V,
    V_S,
    S_MTE2,
    MTE2_S,
    S_MTE3,
    MTE3_S,
    MTE2_FIX,
    FIX_MTE2,
    FIX_S,
    M_S,
    FIX_MTE3,
    MTE1_FIX,
    FIX_MTE1,
    FIX_FIX,
    MAX,
};

enum class HardEventAic : uint8_t {
    // src_dst
    MTE2_MTE1,
    MTE1_MTE2,
    MTE1_M,
    M_MTE1,
    MTE3_MTE1,
    MTE1_MTE3,
    MTE2_M,
    M_MTE2,
    M_FIX,
    FIX_M,
    MTE3_MTE2,
    MTE2_MTE3,
    S_MTE2,
    MTE2_S,
    S_MTE3,
    MTE3_S,
    MTE2_FIX,
    FIX_MTE2,
    FIX_S,
    M_S,
    FIX_MTE3,
    MTE1_FIX,
    FIX_MTE1,
    FIX_FIX,
    MAX,
};

enum class HardEventAiv : uint8_t {
    // src_dst
    MTE2_V,
    V_MTE2,
    MTE3_V,
    V_MTE3,
    V_V,
    MTE3_MTE2,
    MTE2_MTE3,
    S_V,
    V_S,
    S_MTE2,
    MTE2_S,
    S_MTE3,
    MTE3_S,
    MAX,
};

enum class MemoryT : uint8_t { L1 = 0, L0A, L0B, L0C, UB, BIAS };

enum class MemDsbT : uint8_t { ALL = 0, DDR, UB, SEQ };

#if defined(ASCENDC_CPU_DEBUG) || (__CCE_AICORE__ != 220)
constexpr uint8_t EVENT_NUM = static_cast<uint8_t>(HardEvent::MAX);
#else
#ifdef __DAV_C220_CUBE__
constexpr uint8_t EVENT_NUM = static_cast<uint8_t>(HardEventAic::MAX);
#else
constexpr uint8_t EVENT_NUM = static_cast<uint8_t>(HardEventAiv::MAX);
#endif
#endif

__aicore__ constexpr uint8_t EventToIndexAic(HardEvent evt)
{
    // in v220 aic, only 21 events is usefull, so convert evt to index in event pool
    // in other chip version, all events are valid;
    if (evt == HardEvent::MTE2_MTE1) {
        return static_cast<uint8_t>(HardEventAic::MTE2_MTE1);
    } else if (evt == HardEvent::MTE1_MTE2) {
        return static_cast<uint8_t>(HardEventAic::MTE1_MTE2);
    } else if (evt == HardEvent::MTE1_M) {
        return static_cast<uint8_t>(HardEventAic::MTE1_M);
    } else if (evt == HardEvent::M_MTE1) {
        return static_cast<uint8_t>(HardEventAic::M_MTE1);
    } else if (evt == HardEvent::MTE3_MTE1) {
        return static_cast<uint8_t>(HardEventAic::MTE3_MTE1);
    } else if (evt == HardEvent::MTE1_MTE3) {
        return static_cast<uint8_t>(HardEventAic::MTE1_MTE3);
    } else if (evt == HardEvent::MTE2_M) {
        return static_cast<uint8_t>(HardEventAic::MTE2_M);
    } else if (evt == HardEvent::M_MTE2) {
        return static_cast<uint8_t>(HardEventAic::M_MTE2);
    } else if (evt == HardEvent::M_FIX) {
        return static_cast<uint8_t>(HardEventAic::M_FIX);
    } else if (evt == HardEvent::FIX_M) {
        return static_cast<uint8_t>(HardEventAic::FIX_M);
    } else if (evt == HardEvent::MTE3_MTE2) {
        return static_cast<uint8_t>(HardEventAic::MTE3_MTE2);
    } else if (evt == HardEvent::MTE2_MTE3) {
        return static_cast<uint8_t>(HardEventAic::MTE2_MTE3);
    } else if (evt == HardEvent::S_MTE2) {
        return static_cast<uint8_t>(HardEventAic::S_MTE2);
    } else if (evt == HardEvent::MTE2_S) {
        return static_cast<uint8_t>(HardEventAic::MTE2_S);
    } else if (evt == HardEvent::S_MTE3) {
        return static_cast<uint8_t>(HardEventAic::S_MTE3);
    } else if (evt == HardEvent::MTE3_S) {
        return static_cast<uint8_t>(HardEventAic::MTE3_S);
    } else if (evt == HardEvent::MTE2_FIX) {
        return static_cast<uint8_t>(HardEventAic::MTE2_FIX);
    } else if (evt == HardEvent::FIX_MTE2) {
        return static_cast<uint8_t>(HardEventAic::FIX_MTE2);
    } else if (evt == HardEvent::FIX_S) {
        return static_cast<uint8_t>(HardEventAic::FIX_S);
    } else if (evt == HardEvent::M_S) {
        return static_cast<uint8_t>(HardEventAic::M_S);
    } else if (evt == HardEvent::FIX_MTE3) {
        return static_cast<uint8_t>(HardEventAic::FIX_MTE3);
    } else if (evt == HardEvent::FIX_MTE1) {
        return static_cast<uint8_t>(HardEventAic::FIX_MTE1);
    } else if (evt == HardEvent::MTE1_FIX) {
        return static_cast<uint8_t>(HardEventAic::MTE1_FIX);
    } else if (evt == HardEvent::FIX_FIX) {
        return static_cast<uint8_t>(HardEventAic::FIX_FIX);
    } else {
        return static_cast<uint8_t>(HardEventAic::MAX);
    }
}

__aicore__ constexpr uint8_t EventToIndexAiv(HardEvent evt)
{
    // in v220 aiv, only 13 events is usefull, so convert evt to index in event pool
    // in other chip version, all events are valid;
    if (evt == HardEvent::MTE2_V) {
        return static_cast<uint8_t>(HardEventAiv::MTE2_V);
    } else if (evt == HardEvent::V_MTE2) {
        return static_cast<uint8_t>(HardEventAiv::V_MTE2);
    } else if (evt == HardEvent::MTE3_V) {
        return static_cast<uint8_t>(HardEventAiv::MTE3_V);
    } else if (evt == HardEvent::V_MTE3) {
        return static_cast<uint8_t>(HardEventAiv::V_MTE3);
    } else if (evt == HardEvent::V_V) {
        return static_cast<uint8_t>(HardEventAiv::V_V);
    } else if (evt == HardEvent::MTE3_MTE2) {
        return static_cast<uint8_t>(HardEventAiv::MTE3_MTE2);
    } else if (evt == HardEvent::MTE2_MTE3) {
        return static_cast<uint8_t>(HardEventAiv::MTE2_MTE3);
    } else if (evt == HardEvent::S_V) {
        return static_cast<uint8_t>(HardEventAiv::S_V);
    } else if (evt == HardEvent::V_S) {
        return static_cast<uint8_t>(HardEventAiv::V_S);
    } else if (evt == HardEvent::S_MTE2) {
        return static_cast<uint8_t>(HardEventAiv::S_MTE2);
    } else if (evt == HardEvent::MTE2_S) {
        return static_cast<uint8_t>(HardEventAiv::MTE2_S);
    } else if (evt == HardEvent::S_MTE3) {
        return static_cast<uint8_t>(HardEventAiv::S_MTE3);
    } else if (evt == HardEvent::MTE3_S) {
        return static_cast<uint8_t>(HardEventAiv::MTE3_S);
    } else {
        return static_cast<uint8_t>(HardEventAiv::MAX);
    }
}

__aicore__ constexpr uint8_t EventToIndex(HardEvent evt)
{
#if defined(ASCENDC_CPU_DEBUG) || (__CCE_AICORE__ != 220)
    return static_cast<uint8_t>(evt);
#elif defined(__DAV_C220_CUBE__)
    return EventToIndexAic(evt);
#else
    return EventToIndexAiv(evt);
#endif
}

#if (__CCE_AICORE__ <= 200)
constexpr int32_t PIPE_NUM = 6;
constexpr pipe_t SUPPORTED_PIPE[PIPE_NUM] = { PIPE_S, PIPE_V, PIPE_M, PIPE_MTE1, PIPE_MTE2, PIPE_MTE3 };
#else
constexpr int32_t PIPE_NUM = 7;
constexpr pipe_t SUPPORTED_PIPE[PIPE_NUM] = { PIPE_S, PIPE_V, PIPE_M, PIPE_MTE1, PIPE_MTE2, PIPE_MTE3, PIPE_FIX };
#endif

__aicore__ constexpr bool IsSupportedPipe(pipe_t pipe)
{
    for (int i = 0; i < PIPE_NUM; i++) {
        if (pipe == SUPPORTED_PIPE[i]) {
            return true;
        }
    }
    return false;
}

__aicore__ constexpr Hardware GetPhyType(TPosition pos)
{
    ASSERT(pos != TPosition::MAX);
    Hardware hard = Hardware::UB;
    if (pos == TPosition::GM) {
        hard = Hardware::GM;
    } else if (pos == TPosition::A1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::A2) {
        hard = Hardware::L0A;
    } else if (pos == TPosition::B1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::B2) {
        hard = Hardware::L0B;
#if (__CCE_AICORE__ <= 200)
    } else if (pos == TPosition::C1) {
        hard = Hardware::UB;
    } else if (pos == TPosition::C2) {
        hard = Hardware::L0C;
    } else if (pos == TPosition::CO2) {
        hard = Hardware::UB;
#elif (__CCE_AICORE__ == 220)
    } else if (pos == TPosition::C1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::C2) {
        hard = Hardware::BIAS;
    } else if (pos == TPosition::CO2) {
        hard = Hardware::GM;
    } else if (pos == TPosition::C2PIPE2GM) {
        hard = Hardware::FIXBUF;
#elif (__CCE_AICORE__ == 300)
    } else if (pos == TPosition::C1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::C2) {
        hard = Hardware::BIAS;
    } else if (pos == TPosition::C2PIPE2GM) {
        hard = Hardware::FIXBUF;
#elif defined(__DAV_M310__)
    } else if (pos == TPosition::C1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::C2) {
        hard = Hardware::BIAS;
#endif
    } else if (pos == TPosition::CO1) {
        hard = Hardware::L0C;
    } else if (pos == TPosition::SHM) {
        hard = Hardware::L1;
    } else if (pos == TPosition::TSCM) {
        hard = Hardware::L1;
    }
    return hard;
}

__aicore__ constexpr TPosition GetPosition(TPosition srcPos, TPosition dstPos)
{
    // unsupported data stream
    ASSERT(!((srcPos == TPosition::CO2) && (dstPos == TPosition::SHM)));
    ASSERT(!((srcPos == TPosition::VECOUT) && (dstPos == TPosition::SHM)));
#if (__CCE_AICORE__ <= 200)
    if (dstPos == TPosition::GM || ((dstPos == TPosition::CO2) && (srcPos == TPosition::CO1))) {
        return srcPos;
    }
#elif (__CCE_AICORE__ >= 220)
    if ((dstPos == TPosition::GM) || (dstPos == TPosition::CO2)) {
        return srcPos;
    }
#endif
    return dstPos;
}

__aicore__ constexpr Hardware GetBufferPos(TPosition srcPos, TPosition dstPos)
{
    // unsupported data stream
    ASSERT(!((srcPos == TPosition::CO2) && (dstPos == TPosition::SHM)));
    ASSERT(!((srcPos == TPosition::VECOUT) && (dstPos == TPosition::SHM)));
#if (__CCE_AICORE__ <= 200)
    if ((dstPos == TPosition::GM) || ((dstPos == TPosition::CO2) && (srcPos == TPosition::CO1))) {
        return GetPhyType(srcPos);
    }
#elif (__CCE_AICORE__ >= 220)
    if ((dstPos == TPosition::GM) || (dstPos == TPosition::CO2)) {
        return GetPhyType(srcPos);
    }
#endif
    return GetPhyType(dstPos);
}

__aicore__ constexpr TPosition GetBufferLogicPos(TPosition pos, bool isSrc)
{
    ASSERT(pos != TPosition::GM);
    ASSERT(pos != TPosition::VECCALC);
    ASSERT(pos != TPosition::MAX);
    if (pos == TPosition::A1) {
        return isSrc ? TPosition::GM : TPosition::A1;
    } else if (pos == TPosition::B1) {
        return isSrc ? TPosition::GM : TPosition::B1;
    } else if (pos == TPosition::C1) {
        return isSrc ? TPosition::GM : TPosition::C1;
    } else if (pos == TPosition::A2) {
        return isSrc ? TPosition::A1 : TPosition::A2;
    } else if (pos == TPosition::B2) {
        return isSrc ? TPosition::B1 : TPosition::B2;
    } else if (pos == TPosition::C2) {
        return isSrc ? TPosition::C1 : TPosition::C2;
    } else if (pos == TPosition::CO1) {
        return isSrc ? TPosition::CO1 : TPosition::CO2;
    } else if (pos == TPosition::CO2) {
        return isSrc ? TPosition::CO2 : TPosition::GM;
    } else if (pos == TPosition::VECIN) {
        return isSrc ? TPosition::GM : TPosition::VECIN;
    } else if (pos == TPosition::VECOUT) {
        return isSrc ? TPosition::VECOUT : TPosition::GM;
    } else if (pos == TPosition::SPM) {
        return isSrc ? TPosition::VECOUT : TPosition::GM;
    } else if (pos == TPosition::C2PIPE2GM) {
        return isSrc ? TPosition::B1 : TPosition::C2PIPE2GM;
    }
    return TPosition::MAX;
}

__aicore__ constexpr HardEvent GetQueEvt(Hardware src, Hardware dst, bool fwdDirect, bool nd2nz = false,
                                         bool nz2nd = false)
{
    (void)(nz2nd);
    ASSERT((src == Hardware::GM) || (src == Hardware::UB) || (src == Hardware::L1) || (src == Hardware::L0A) ||
           (src == Hardware::L0B) || (src == Hardware::L0C));
    ASSERT(src != Hardware::MAX);
    ASSERT(dst != Hardware::MAX);
    if (src == Hardware::GM) {  // MTE2
        ASSERT(dst != Hardware::GM);
        ASSERT(dst != Hardware::L0C);
        ASSERT(dst != Hardware::BIAS);
        ASSERT(dst != Hardware::FIXBUF);
        if (dst == Hardware::UB) {  // MTE3
            return fwdDirect ? HardEvent::MTE2_V : HardEvent::V_MTE2;
        } else if (dst == Hardware::L1) {  // MTE1
#if (__CCE_AICORE__ <= 200)
            // in v100/v200, nd2nz was simulated with vector intrins, so event changed event to mte3
            if (nd2nz) {
                return fwdDirect ? HardEvent::MTE3_MTE1 : HardEvent::MTE1_MTE3;
            }
#else
            (void)(nd2nz);
#endif
            return fwdDirect ? HardEvent::MTE2_MTE1 : HardEvent::MTE1_MTE2;
        } else if (dst == Hardware::L0A) {
            return fwdDirect ? HardEvent::MTE2_M : HardEvent::M_MTE2;
        } else if (dst == Hardware::L0B) {
            return fwdDirect ? HardEvent::MTE2_M : HardEvent::M_MTE2;
        }
    } else if (src == Hardware::UB) {  // MTE3
        ASSERT(dst != Hardware::L0A);
        ASSERT(dst != Hardware::L0B);
        ASSERT(dst != Hardware::BIAS);
        ASSERT(dst != Hardware::FIXBUF);
        if (dst == Hardware::GM) {
            return fwdDirect ? HardEvent::V_MTE3 : HardEvent::MTE3_V;
        } else if (dst == Hardware::L1) {  // MTE1
            return fwdDirect ? HardEvent::MTE3_MTE1 : HardEvent::MTE1_MTE3;
        } else if (dst == Hardware::L0C) {
            return fwdDirect ? HardEvent::V_V : HardEvent::MAX;  // HardEvent::M_V
        } else if (dst == Hardware::UB) {
            return fwdDirect ? HardEvent::MTE2_MTE3 : HardEvent::MTE3_MTE2;
        }
    } else if (src == Hardware::L1) {  // MTE1
        ASSERT(dst != Hardware::GM);
        ASSERT(dst != Hardware::L1);
        ASSERT(dst != Hardware::L0C);
#if (__CCE_AICORE__ <= 200)
        ASSERT(dst != Hardware::BIAS);
        ASSERT(dst != Hardware::FIXBUF);
#endif
        if (dst == Hardware::UB) {
            return fwdDirect ? HardEvent::MTE1_V : HardEvent::V_MTE1;
        } else if (dst == Hardware::L0A) {
            return fwdDirect ? HardEvent::MTE1_M : HardEvent::M_MTE1;
        } else if (dst == Hardware::L0B) {
            return fwdDirect ? HardEvent::MTE1_M : HardEvent::M_MTE1;
        } else if (dst == Hardware::FIXBUF) {
            return fwdDirect ? HardEvent::MTE1_FIX : HardEvent::FIX_MTE1;
        } else if (dst == Hardware::BIAS) {
            return fwdDirect ? HardEvent::MTE1_M : HardEvent::M_MTE1;
        }
    } else if (src == Hardware::L0A) {
        ASSERT(dst == Hardware::L0C);
        return fwdDirect ? HardEvent::M_V : HardEvent::V_M;
    } else if (src == Hardware::L0B) {
        ASSERT(dst == Hardware::L0C);
        return fwdDirect ? HardEvent::M_V : HardEvent::V_M;
#if (__CCE_AICORE__ <= 200)
    } else if (src == Hardware::L0C) {
        ASSERT(dst == Hardware::UB);
        return fwdDirect ? HardEvent::M_V : HardEvent::V_M;
    }
#elif (__CCE_AICORE__ == 220)
    } else if (src == Hardware::L0C) {
        ASSERT(dst == Hardware::GM);
        return fwdDirect ? HardEvent::M_FIX : HardEvent::FIX_M;
    }
#elif (__CCE_AICORE__ == 300)
    } else if (src == Hardware::L0C) {
        ASSERT(dst == Hardware::GM || dst == Hardware::UB);
        return fwdDirect ? HardEvent::M_FIX : HardEvent::FIX_M;
    }
#elif defined(__DAV_M310__)
    } else if (src == Hardware::L0C) {
        ASSERT(dst == Hardware::GM || dst == Hardware::UB);
        return fwdDirect ? HardEvent::M_FIX : HardEvent::FIX_M;
    }
#else
    }
#endif
    return HardEvent::MAX;
}

#if __CCE_AICORE__ >= 220
template <MemDsbT arg>
__aicore__ inline void DataSyncBarrierImpl()
{
    dsb((mem_dsb_t)arg);
}

template <HardEvent event, MemoryT memT, bool isVirtual>
__aicore__ inline void HSetFlagImpl(int32_t eventID)
{
    ASCENDC_ASSERT((eventID >= 0 && eventID < QUE_MAX_EVENT),
        { KERNEL_LOG(KERNEL_ERROR, "For HSetFlag, eventID %d should be in range [0, %d)", eventID, QUE_MAX_EVENT); });
    static_assert(((int32_t)memT >= 0 && memT <= MemoryT::BIAS && memT != MemoryT::UB && memT != MemoryT::L1),
        "For HSetFlag, memT only support L0A, L0B, L0C, BIAS.");

    event_t e = static_cast<event_t>(eventID);

    switch (event) {
        case HardEvent::MTE1_M:
            ASCENDC_ASSERT((memT != MemoryT::L1 && memT != MemoryT::L0C),
                           "memT only support L0A, L0B, BIAS in MTE1_M.");
            hset_flag(PIPE_MTE1, PIPE_M, e, (mem_t)memT, isVirtual);
            break;
        case HardEvent::M_MTE1:
            ASCENDC_ASSERT((memT != MemoryT::L1 && memT != MemoryT::L0C),
                           "memT only support L0A, L0B, BIAS in M_MTE1.");
            hset_flag(PIPE_M, PIPE_MTE1, e, (mem_t)memT, isVirtual);
            break;
        case HardEvent::M_FIX:
            ASCENDC_ASSERT((memT == MemoryT::L0C), "memT only support L0C in M_FIX.");
            hset_flag(PIPE_M, PIPE_FIX, e, (mem_t)memT, isVirtual);
            break;
        case HardEvent::FIX_M:
            ASCENDC_ASSERT((memT == MemoryT::L0C), "memT only support L0C in FIX_M.");
            hset_flag(PIPE_FIX, PIPE_M, e, (mem_t)memT, isVirtual);
            break;
        default:
            ASCENDC_ASSERT((0), KERNEL_LOG(KERNEL_ERROR, "invalid event %d", static_cast<int32_t>(event)););
            break;
    }
}

template <HardEvent event, MemoryT memT, bool isVirtual>
__aicore__ inline void HWaitFlagImpl(int32_t eventID)
{
    ASCENDC_ASSERT((eventID >= 0 && eventID < QUE_MAX_EVENT),
                   { KERNEL_LOG(KERNEL_ERROR, "For HWaitFlag, eventID %d should be in range [0, %d)", eventID, QUE_MAX_EVENT); });
    static_assert(((int32_t)memT >= 0 && memT <= MemoryT::BIAS && memT != MemoryT::UB && memT != MemoryT::L1),
                  "For HWaitFlag, memT only support L0A, L0B, L0C, BIAS.");

    event_t e = static_cast<event_t>(eventID);

    switch (event) {
        case HardEvent::MTE1_M:
            ASCENDC_ASSERT((memT != MemoryT::L1 && memT != MemoryT::L0C),
                           "memT only support L0A, L0B, BIAS in MTE1_M.");
            hwait_flag(PIPE_MTE1, PIPE_M, e, (mem_t)memT, isVirtual);
            break;
        case HardEvent::M_MTE1:
            ASCENDC_ASSERT((memT != MemoryT::L1 && memT != MemoryT::L0C),
                           "memT only support L0A, L0B, BIAS in M_MTE1.");
            hwait_flag(PIPE_M, PIPE_MTE1, e, (mem_t)memT, isVirtual);
            break;
        case HardEvent::M_FIX:
            ASCENDC_ASSERT((memT == MemoryT::L0C), "memT only support L0C in M_FIX.");
            hwait_flag(PIPE_M, PIPE_FIX, e, (mem_t)memT, isVirtual);
            break;
        case HardEvent::FIX_M:
            ASCENDC_ASSERT((memT == MemoryT::L0C), "memT only support L0C in FIX_M.");
            hwait_flag(PIPE_FIX, PIPE_M, e, (mem_t)memT, isVirtual);
            break;
        default:
            ASCENDC_ASSERT((0), KERNEL_LOG(KERNEL_ERROR, "invalid event %d", static_cast<int32_t>(event)););
            break;
    }
}
#endif

template <HardEvent event>
__aicore__ inline void SetFlagImpl(int32_t eventID)
{
    ASCENDC_ASSERT((eventID >= 0 && eventID < QUE_MAX_EVENT),
                   { KERNEL_LOG(KERNEL_ERROR, "eventID %d should be in range [0, %d)", eventID, QUE_MAX_EVENT); });
    event_t e = static_cast<event_t>(eventID);
    switch (event) {
        case HardEvent::MTE2_MTE1:
            set_flag(PIPE_MTE2, PIPE_MTE1, e);
            break;
        case HardEvent::MTE1_MTE2:
            set_flag(PIPE_MTE1, PIPE_MTE2, e);
            break;
        case HardEvent::MTE2_MTE3:
            set_flag(PIPE_MTE2, PIPE_MTE3, e);
            break;
        case HardEvent::MTE3_MTE2:
            set_flag(PIPE_MTE3, PIPE_MTE2, e);
            break;
        case HardEvent::MTE1_M:
            set_flag(PIPE_MTE1, PIPE_M, e);
            break;
        case HardEvent::M_MTE1:
            set_flag(PIPE_M, PIPE_MTE1, e);
            break;
        case HardEvent::MTE2_V:
            set_flag(PIPE_MTE2, PIPE_V, e);
            break;
        case HardEvent::V_MTE2:
            set_flag(PIPE_V, PIPE_MTE2, e);
            break;
        case HardEvent::MTE3_V:
            set_flag(PIPE_MTE3, PIPE_V, e);
            break;
        case HardEvent::V_MTE3:
            set_flag(PIPE_V, PIPE_MTE3, e);
            break;
        case HardEvent::M_V:
            set_flag(PIPE_M, PIPE_V, e);
            break;
        case HardEvent::M_S:
            set_flag(PIPE_M, PIPE_S, e);
            break;
        case HardEvent::V_M:
            set_flag(PIPE_V, PIPE_M, e);
            break;
        case HardEvent::S_V:
            set_flag(PIPE_S, PIPE_V, e);
            break;
        case HardEvent::V_S:
            set_flag(PIPE_V, PIPE_S, e);
            break;
#if (__CCE_AICORE__ != 300)
        case HardEvent::V_V:
            pipe_barrier(PIPE_V);
            return;
#endif
        case HardEvent::MTE3_MTE1:
            set_flag(PIPE_MTE3, PIPE_MTE1, e);
            break;
        case HardEvent::MTE1_MTE3:
            set_flag(PIPE_MTE1, PIPE_MTE3, e);
            break;
        case HardEvent::MTE1_V:
            set_flag(PIPE_MTE1, PIPE_V, e);
            break;
        case HardEvent::MTE2_M:
            set_flag(PIPE_MTE2, PIPE_M, e);
            break;
        case HardEvent::M_MTE2:
            set_flag(PIPE_M, PIPE_MTE2, e);
            break;
        case HardEvent::S_MTE2:
            set_flag(PIPE_S, PIPE_MTE2, e);
            break;
        case HardEvent::MTE2_S:
            set_flag(PIPE_MTE2, PIPE_S, e);
            break;
        case HardEvent::V_MTE1:
            set_flag(PIPE_V, PIPE_MTE1, e);
            break;
        case HardEvent::S_MTE3:
            set_flag(PIPE_S, PIPE_MTE3, e);
            break;
        case HardEvent::MTE3_S:
            set_flag(PIPE_MTE3, PIPE_S, e);
            break;
#if (__CCE_AICORE__ >= 220)
        case HardEvent::M_FIX:
            set_flag(PIPE_M, PIPE_FIX, e);
            break;
        case HardEvent::FIX_M:
            set_flag(PIPE_FIX, PIPE_M, e);
            break;
        case HardEvent::FIX_MTE3:
            set_flag(PIPE_FIX, PIPE_MTE3, e);
            break;
        case HardEvent::FIX_MTE2:
            set_flag(PIPE_FIX, PIPE_MTE2, e);
            break;
        case HardEvent::MTE2_FIX:
            set_flag(PIPE_MTE2, PIPE_FIX, e);
            break;
        case HardEvent::FIX_S:
            set_flag(PIPE_FIX, PIPE_S, e);
            break;
        case HardEvent::MTE1_FIX:
            set_flag(PIPE_MTE1, PIPE_FIX, e);
            break;
        case HardEvent::FIX_MTE1:
            set_flag(PIPE_FIX, PIPE_MTE1, e);
            break;
        case HardEvent::FIX_FIX:
            pipe_barrier(PIPE_FIX);
            break;
#endif
        case HardEvent::MAX:
            break;
        default:
            ASCENDC_ASSERT((0), { KERNEL_LOG(KERNEL_ERROR, "invalid event %d", static_cast<int32_t>(event)); });
            break;
    }
}

__aicore__ inline void WaitFlagImpl(const HardEvent event, int32_t eventID)
{
    ASCENDC_ASSERT((eventID >= 0 && eventID < QUE_MAX_EVENT),
                   { KERNEL_LOG(KERNEL_ERROR, "eventID %d should be in range [0, %d)", eventID, QUE_MAX_EVENT); });
    event_t e = static_cast<event_t>(eventID);
    switch (event) {
#ifndef __DAV_C220_VEC__  // CUBE core
        case HardEvent::MTE2_MTE1:
            wait_flag(PIPE_MTE2, PIPE_MTE1, e);
            break;
        case HardEvent::MTE1_MTE2:
            wait_flag(PIPE_MTE1, PIPE_MTE2, e);
            break;
        case HardEvent::MTE1_M:
            wait_flag(PIPE_MTE1, PIPE_M, e);
            break;
        case HardEvent::M_MTE1:
            wait_flag(PIPE_M, PIPE_MTE1, e);
            break;
        case HardEvent::MTE3_MTE1:
            wait_flag(PIPE_MTE3, PIPE_MTE1, e);
            break;
        case HardEvent::MTE1_MTE3:
            wait_flag(PIPE_MTE1, PIPE_MTE3, e);
            break;
        case HardEvent::MTE2_M:
            wait_flag(PIPE_MTE2, PIPE_M, e);
            break;
        case HardEvent::M_MTE2:
            wait_flag(PIPE_M, PIPE_MTE2, e);
            break;
#endif
#ifndef __DAV_C220_CUBE__  // VECTOR core
        case HardEvent::MTE2_V:
            wait_flag(PIPE_MTE2, PIPE_V, e);
            break;
        case HardEvent::V_MTE2:
            wait_flag(PIPE_V, PIPE_MTE2, e);
            break;
        case HardEvent::MTE3_V:
            wait_flag(PIPE_MTE3, PIPE_V, e);
            break;
        case HardEvent::V_MTE3:
            wait_flag(PIPE_V, PIPE_MTE3, e);
            break;
#endif
#if __CCE_AICORE__ != 220
        case HardEvent::M_V:
            wait_flag(PIPE_M, PIPE_V, e);
            break;
        case HardEvent::V_M:
            wait_flag(PIPE_V, PIPE_M, e);
            break;
        case HardEvent::MTE1_V:
            wait_flag(PIPE_MTE1, PIPE_V, e);
            break;
        case HardEvent::V_MTE1:
            wait_flag(PIPE_V, PIPE_MTE1, e);
            break;
#endif
#if (__CCE_AICORE__ >= 220)
        case HardEvent::FIX_M:
            wait_flag(PIPE_FIX, PIPE_M, e);
            break;
        case HardEvent::M_FIX:
            wait_flag(PIPE_M, PIPE_FIX, e);
            break;
        case HardEvent::MTE2_FIX:
            wait_flag(PIPE_MTE2, PIPE_FIX, e);
            break;
        case HardEvent::FIX_MTE2:
            wait_flag(PIPE_FIX, PIPE_MTE2, e);
            break;
        case HardEvent::FIX_S:
            wait_flag(PIPE_FIX, PIPE_S, e);
            break;
        case HardEvent::FIX_MTE3:
            wait_flag(PIPE_FIX, PIPE_MTE3, e);
            break;
        case HardEvent::MTE1_FIX:
            wait_flag(PIPE_MTE1, PIPE_FIX, e);
            break;
        case HardEvent::FIX_MTE1:
            wait_flag(PIPE_FIX, PIPE_MTE1, e);
            break;
        case HardEvent::FIX_FIX:
            pipe_barrier(PIPE_FIX);
            break;
#endif
        case HardEvent::MTE3_MTE2:
            wait_flag(PIPE_MTE3, PIPE_MTE2, e);
            break;
        case HardEvent::MTE2_MTE3:
            wait_flag(PIPE_MTE2, PIPE_MTE3, e);
            break;
        case HardEvent::S_MTE2:
            wait_flag(PIPE_S, PIPE_MTE2, e);
            break;
        case HardEvent::MTE2_S:
            wait_flag(PIPE_MTE2, PIPE_S, e);
            break;
        case HardEvent::S_MTE3:
            wait_flag(PIPE_S, PIPE_MTE3, e);
            break;
        case HardEvent::MTE3_S:
            wait_flag(PIPE_MTE3, PIPE_S, e);
            break;
        case HardEvent::M_S:
            wait_flag(PIPE_M, PIPE_S, e);
            break;
        case HardEvent::S_V:
            wait_flag(PIPE_S, PIPE_V, e);
            break;
        case HardEvent::V_S:
            wait_flag(PIPE_V, PIPE_S, e);
            break;
        case HardEvent::V_V:
            return;
        case HardEvent::MAX:
            break;
        default:
            break;
    }
    return;
}
}  // namespace AscendC

#endif  // ASCENDC_KERNEL_EVENT_IMPL_H