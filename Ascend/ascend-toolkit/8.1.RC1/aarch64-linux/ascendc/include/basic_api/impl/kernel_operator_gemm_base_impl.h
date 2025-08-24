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
 * \file kernel_operator_gemm_base_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_GEMM_BASE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_GEMM_BASE_IMPL_H
#if ASCENDC_CPU_DEBUG
#include <unordered_set>
#endif
#include "kernel_tensor.h"
#include "kernel_operator_mm_base_impl.h"
#include "kernel_struct_conv2d.h"
#include "kernel_struct_mm.h"

namespace AscendC {
#if ASCENDC_CPU_DEBUG

const std::unordered_set<std::string> MATMUL_SUPPORT_TYPE { "s8s8s32", "f16f16f32", "f16f16f16" };

template <typename T> __aicore__ inline std::string GetTypeStr(const LocalTensor<T>& tensor)
{
    if (std::is_same<PrimT<T>, uint8_t>::value) {
        return "u8";
    } else if (std::is_same<PrimT<T>, int8_t>::value) {
        return "s8";
    } else if (std::is_same<PrimT<T>, half>::value) {
        return "f16";
    } else if (std::is_same<PrimT<T>, float>::value) {
        return "f32";
    } else if (std::is_same<PrimT<T>, int32_t>::value) {
        return "s32";
    } else {
        return "None";
    }
}

__aicore__ inline bool CheckRange(std::pair<uint32_t, uint32_t>& range, const uint32_t num)
{
    if (num < range.first || num > range.second) {
        return false;
    } else {
        return true;
    }
}

template <typename dst_T, typename src0_T, typename src1_T>
__aicore__ inline bool CheckOverflow(const LocalTensor<dst_T>& dstLocal, const LocalTensor<src0_T>& src0Local,
    const LocalTensor<src1_T>& src1Local, const uint32_t m, const uint32_t k, const uint32_t n, GemmTiling& tilling)
{
    // check l0c
    uint32_t roundM = DivCeil(m, tilling.blockSize) * tilling.blockSize;
    uint32_t roundN = DivCeil(n, tilling.blockSize) * tilling.blockSize;
    uint32_t roundK = DivCeil(k, tilling.c0Size) * tilling.c0Size;

    uint32_t needElementLoc = roundM * roundN * sizeof(uint32_t);
    if (needElementLoc > TOTAL_L0C_SIZE) {
        return false;
    }

    // check l0a:
    uint32_t needElementL0a = roundM * roundK * sizeof(PrimT<src0_T>);

    // check l0b:
    uint32_t needElementL0b = roundN * roundK * sizeof(PrimT<src1_T>);
    if ((needElementL0b + needElementL0a) > TOTAL_L1_SIZE) {
        return false;
    }

    return true;
}

template <typename dst_T, typename src0_T, typename src1_T>
__aicore__ inline bool CheckParams(const LocalTensor<dst_T>& dstLocal, const LocalTensor<src0_T>& src0Local,
    const LocalTensor<src1_T>& src1Local, const uint32_t m, const uint32_t k, const uint32_t n, GemmTiling& tilling)
{
    // check c0Size
    if (tilling.c0Size != 16 && tilling.c0Size != 32) {
        return false;
    }
    // check scope
    const Hardware src0Scope = GetPhyType((TPosition)src0Local.GetPosition());
    const Hardware src1Scope = GetPhyType((TPosition)src1Local.GetPosition());
    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    if (src0Scope != Hardware::L1 || src1Scope != Hardware::L1) {
        return false;
    }
    if (dstScope != Hardware::UB && dstScope != Hardware::L0C) {
        return false;
    }

    // check dtype
    std::string dtypeStr = GetTypeStr(src0Local) + GetTypeStr(src1Local) + GetTypeStr(dstLocal);
    if (MATMUL_SUPPORT_TYPE.find(dtypeStr) == MATMUL_SUPPORT_TYPE.end()) {
        return false;
    }

    // check m/k/n range
    std::pair<uint32_t, uint32_t> mRange(1, 4096);
    std::pair<uint32_t, uint32_t> nRange = mRange;
    std::pair<uint32_t, uint32_t> kRange;
    if (std::is_same<src0_T, half>::value) {
        kRange = std::make_pair(1, 32768);
    } else {
        kRange = std::make_pair(1, 16384);
    }

    if (!CheckRange(mRange, m) || !CheckRange(nRange, n) || !CheckRange(kRange, k)) {
        return false;
    }

    // check overflow
    if (!CheckOverflow(dstLocal, src0Local, src1Local, m, k, n, tilling)) {
        return false;
    }

    return true;
}
#endif

__aicore__ inline void CalculateGemmTiling(GemmTiling& tilling)
{
    tilling.mIterNum = 1;
    tilling.nIterNum = 1;
    tilling.kIterNum = DivCeil(tilling.kBlockNum, tilling.kTileBlock);

    tilling.mTileBlock = DivCeil(tilling.mBlockNum, tilling.mIterNum);
    tilling.nTileBlock = DivCeil(tilling.nBlockNum, tilling.nIterNum);

    tilling.kTailBlock = tilling.kBlockNum - (tilling.kIterNum - 1) * tilling.kTileBlock;
    tilling.mTailBlock = tilling.mBlockNum - (tilling.mIterNum - 1) * tilling.mTileBlock; // mTailBlock <= mBlockNum
    tilling.nTailBlock = tilling.nBlockNum - (tilling.nIterNum - 1) * tilling.nTileBlock;

    tilling.kHasTail = tilling.kTailBlock != tilling.kTileBlock;
    tilling.kHasTailEle = tilling.roundK != tilling.kNum;
    tilling.kTailEle = tilling.kNum % (tilling.kTileBlock * tilling.c0Size);

    if (tilling.mNum != tilling.mTileBlock * tilling.blockSize) {
        tilling.mHasTail = true;
    } else {
        tilling.mHasTail = false;
    }
    tilling.nHasTail = tilling.nTileBlock != tilling.nTailBlock;
}

template <typename T>
__aicore__ inline void LoadL0B(uint32_t kBlocks, uint32_t nBlocks, GemmTiling tilling, uint32_t i, uint32_t j,
    const LocalTensor<T>& src1Local, const LocalTensor<T>& L0b)
{
    if (tilling.nIterNum == 1) {
        uint32_t wSize = tilling.blockSize * tilling.c0Size;
        uint32_t wIdx = (i * tilling.kTileBlock * tilling.nBlockNum + j * tilling.nTileBlock) * wSize;
        LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = kBlocks * nBlocks;
        params.srcStride = 1;
        LoadDataImpl(L0b, src1Local[wIdx], params);
    } else {
        // load data row by row
        for (size_t index = 0; index < kBlocks; ++index) {
            uint32_t wSize = j * tilling.nTileBlock * tilling.blockSize * tilling.c0Size;
            uint32_t wIdx =
                (i * tilling.kTileBlock + index) * tilling.nBlockNum * tilling.blockSize * tilling.c0Size + wSize;
            uint32_t l0bIdx = index * nBlocks * tilling.blockSize * tilling.c0Size;
            LoadData2DParams params;
            params.startIndex = 0;
            params.repeatTimes = nBlocks;
            params.srcStride = 1;
            LoadDataImpl(L0b[l0bIdx], src1Local[wIdx], params);
        }
    }
}

template <typename T>
__aicore__ inline void LoadL0A(uint32_t kBlocks, uint32_t mBlocks, GemmTiling tilling, uint32_t i, uint32_t t,
    const LocalTensor<T>& src0Local, const LocalTensor<T>& L0a)
{
    if (kBlocks == 1) {
        uint32_t l1aSize = i * tilling.kTileBlock * tilling.mBlockNum * tilling.blockSize * tilling.c0Size;
        uint32_t l1aOffset = t * tilling.mTileBlock * tilling.blockSize * tilling.c0Size + l1aSize;
        LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = mBlocks;
        params.srcStride = 1;
        LoadDataImpl(L0a, src0Local[l1aOffset], params);
    } else {
        // load data row by row
        for (size_t index = 0; index < mBlocks; index++) {
            uint32_t l0aOffset = index * kBlocks * tilling.blockSize * tilling.c0Size;
            uint32_t l1aOffset = (t * tilling.mTileBlock + index) * tilling.blockSize * tilling.c0Size +
                i * tilling.kTileBlock * tilling.mBlockNum * tilling.blockSize * tilling.c0Size;
            LoadData2DParams params;
            params.startIndex = 0;
            params.repeatTimes = kBlocks;
            params.srcStride = tilling.mBlockNum;
            LoadDataImpl(L0a[l0aOffset], src0Local[l1aOffset], params);
        }
    }
}

template <typename dst_T, typename src0_T, typename src1_T>
__aicore__ inline void MmadFunc(const LocalTensor<src0_T>& L0a, const LocalTensor<src1_T>& L0b,
    const LocalTensor<dst_T>& L0c, int32_t initValue, GemmTiling tilling, size_t i)
{
    MmadParams mmadParams;
    mmadParams.m = tilling.mTileBlock * tilling.blockSize;
    mmadParams.n = tilling.nTileBlock * tilling.blockSize;
    mmadParams.isBias = 1;

    if (tilling.kIterNum == 1) {
        mmadParams.k = tilling.kNum;
        mmadParams.isBias = initValue;
    } else if (initValue == 1 && tilling.kHasTailEle) {
        if (i == tilling.kIterNum - 1) {
            mmadParams.k = tilling.kTailEle;
        } else {
            mmadParams.k = tilling.kTileBlock * tilling.c0Size;
        }
    } else if (initValue != 1 && tilling.kHasTailEle) {
        if (i == 0) {
            mmadParams.k = tilling.kTileBlock * tilling.c0Size;
            mmadParams.isBias = 0;
        } else if (i == tilling.kIterNum - 1) {
            mmadParams.k = tilling.kTailEle;
        } else {
            mmadParams.k = tilling.kTileBlock * tilling.c0Size;
        }
    } else if (initValue == 1 && !tilling.kHasTailEle) {
        if (i == tilling.kIterNum - 1) {
            mmadParams.k = tilling.kTailBlock * tilling.c0Size;
        } else {
            mmadParams.k = tilling.kTileBlock * tilling.c0Size;
        }
    } else {
        if (i == 0) {
            mmadParams.k = tilling.kTileBlock * tilling.c0Size;
            mmadParams.isBias = 0;
        } else if (i == tilling.kIterNum - 1) {
            mmadParams.k = tilling.kTailBlock * tilling.c0Size;
        } else {
            mmadParams.k = tilling.kTileBlock * tilling.c0Size;
        }
    }
    MmadImpl(L0c, L0a, L0b, mmadParams);
}

template <typename src0_T, typename src1_T>
__aicore__ inline void GetPingPongBuffer(LocalTensor<src0_T>& L0aPing, LocalTensor<src0_T>& L0aPong,
    LocalTensor<src1_T>& L0bPing, LocalTensor<src1_T>& L0bPong)
{
    // L0Abuffer
    TBuffAddr tbufaPing;
    tbufaPing.logicPos = (uint8_t)TPosition::A2;
    L0aPing.SetAddr(tbufaPing);
    L0aPing.InitBuffer(0, TOTAL_L0A_SIZE / 2 / sizeof(PrimT<src0_T>));

    TBuffAddr tbufaPong;
    tbufaPong.logicPos = (uint8_t)TPosition::A2;
    L0aPong.SetAddr(tbufaPong);
    L0aPong.InitBuffer(TOTAL_L0A_SIZE / 2, TOTAL_L0A_SIZE / 2 / sizeof(PrimT<src0_T>));

    // L0Bbuffer
    TBuffAddr tbufbPing;
    tbufbPing.logicPos = (uint8_t)TPosition::B2;
    L0bPing.SetAddr(tbufbPing);
    L0bPing.InitBuffer(0, TOTAL_L0B_SIZE / 2 / sizeof(PrimT<src1_T>));

    TBuffAddr tbufbPong;
    tbufbPong.logicPos = (uint8_t)TPosition::B2;
    L0bPong.SetAddr(tbufbPong);
    L0bPong.InitBuffer(TOTAL_L0B_SIZE / 2, TOTAL_L0B_SIZE / 2 / sizeof(PrimT<src1_T>));
    return;
}

template <typename src0_T, typename src1_T>
__aicore__ inline void GetSingleThreadBuffer(LocalTensor<src0_T>& L0a, LocalTensor<src1_T>& L0b)
{
    // L0Abuffer
    TBuffAddr tbufa;
    tbufa.logicPos = (uint8_t)TPosition::A2;
    L0a.SetAddr(tbufa);
    L0a.InitBuffer(0, TOTAL_L0A_SIZE / sizeof(PrimT<src0_T>));

    // L0Bbuffer
    TBuffAddr tbufb;
    tbufb.logicPos = (uint8_t)TPosition::B2;
    L0b.SetAddr(tbufb);
    L0b.InitBuffer(0, TOTAL_L0B_SIZE / sizeof(PrimT<src1_T>));
    return;
}

template <typename dst_T, typename src0_T, typename src1_T>
__aicore__ inline void GemmExecNmNopingpong(const LocalTensor<dst_T>& L0c, const LocalTensor<src0_T>& src0Local,
    const LocalTensor<src1_T>& src1Local, GemmTiling tilling, const int32_t initValue)
{
    LocalTensor<src0_T> L0a;
    LocalTensor<src1_T> L0b;
    GetSingleThreadBuffer(L0a, L0b);
    event_t eventIdMToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
    SetFlag<HardEvent::M_MTE1>(eventIdMToMte1);
    for (size_t indexK = 0; indexK < tilling.kIterNum; indexK++) {
        uint32_t kBlocks = tilling.kTileBlock;
        if (indexK == tilling.kIterNum - 1) {
            kBlocks = tilling.kTailBlock;
        }
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1);
        for (size_t indexN = 0; indexN < tilling.nIterNum; indexN++) {
            // load data from l1 to l0b
            LoadL0B(kBlocks, tilling.nTileBlock, tilling, indexK, indexN, src1Local, L0b);
            for (size_t indexM = 0; indexM < tilling.mIterNum; indexM++) {
                // load data from l1 to l0a
                LoadL0A(kBlocks, tilling.mTileBlock, tilling, indexK, indexM, src0Local, L0a);
                event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
                SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                PipeBarrier<PIPE_M>();
                MmadFunc(L0a, L0b, L0c, initValue, tilling, indexK);
            }
        }
        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1);
    }
    WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1);
}

template <typename dst_T, typename src0_T, typename src1_T>
__aicore__ inline void GemmExecNmPingPong(const LocalTensor<dst_T>& L0c, const LocalTensor<src0_T>& src0Local,
    const LocalTensor<src1_T>& src1Local, GemmTiling tilling, const int32_t initValue)
{
    uint32_t ping = 1;
    LocalTensor<src0_T> L0aPing;
    LocalTensor<src0_T> L0aPong;
    LocalTensor<src1_T> L0bPing;
    LocalTensor<src1_T> L0bPong;
    GetPingPongBuffer(L0aPing, L0aPong, L0bPing, L0bPong);

    event_t eventId0 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
    event_t eventId1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
    SetFlag<HardEvent::M_MTE1>(eventId0);
    SetFlag<HardEvent::M_MTE1>(eventId1);

    for (size_t i = 0; i < tilling.kIterNum; i++) {
        uint32_t kBlocks = tilling.kTileBlock;
        if (i == tilling.kIterNum - 1) {
            kBlocks = tilling.kTailBlock;
        }
        if (ping == 1) {
            WaitFlag<HardEvent::M_MTE1>(eventId0);
            for (size_t indexN = 0; indexN < tilling.nIterNum; indexN++) {
                // load data from l1 to l0b
                LoadL0B(kBlocks, tilling.nTileBlock, tilling, i, indexN, src1Local, L0bPing);
                for (size_t indexM = 0; indexM < tilling.mIterNum; indexM++) {
                    // load data from l1 to l0a
                    LoadL0A(kBlocks, tilling.mTileBlock, tilling, i, indexM, src0Local, L0aPing);
                    event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
                    SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                    WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                    PipeBarrier<PIPE_M>();
                    MmadFunc(L0aPing, L0bPing, L0c, initValue, tilling, i);
                }
            }
            SetFlag<HardEvent::M_MTE1>(eventId0);
        } else {
            WaitFlag<HardEvent::M_MTE1>(eventId1);
            for (size_t indexN = 0; indexN < tilling.nIterNum; indexN++) {
                // load data from l1 to l0b
                LoadL0B(kBlocks, tilling.nTileBlock, tilling, i, indexN, src1Local, L0bPong);
                for (size_t indexM = 0; indexM < tilling.mIterNum; indexM++) {
                    // load data from l1 to l0a
                    LoadL0A(kBlocks, tilling.mTileBlock, tilling, i, indexM, src0Local, L0aPong);
                    event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
                    SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                    WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                    PipeBarrier<PIPE_M>();
                    MmadFunc(L0aPong, L0bPong, L0c, initValue, tilling, i);
                }
            }
            SetFlag<HardEvent::M_MTE1>(eventId1);
        }
        ping = 1 - ping;
    }

#if __CCE_AICORE__ == 220
    WaitFlag<HardEvent::M_MTE1>(eventId0);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventId0);
    WaitFlag<HardEvent::M_MTE1>(eventId1);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventId1);
#else
    WaitFlag<HardEvent::M_MTE1>(eventId0);
    WaitFlag<HardEvent::M_MTE1>(eventId1);
#endif
}

template <typename dst_T, typename src0_T, typename src1_T>
__aicore__ inline void GemmExecNm(const LocalTensor<dst_T>& L0c, const LocalTensor<src0_T>& src0Local,
    const LocalTensor<src1_T>& src1Local, GemmTiling tilling, const int32_t initValue)
{
    uint32_t needL0Asize = tilling.roundM * tilling.dtypeSize * tilling.c0Size * tilling.kTileBlock * 2;
    uint32_t needL0Bsize = tilling.roundN * tilling.dtypeSize * tilling.c0Size * tilling.kTileBlock * 2;
    if (needL0Asize > TOTAL_L0A_SIZE || needL0Bsize > TOTAL_L0B_SIZE) {
        GemmExecNmNopingpong(L0c, src0Local, src1Local, tilling, initValue);
        return;
    }
    GemmExecNmPingPong(L0c, src0Local, src1Local, tilling, initValue);
}

template <typename dst_T, typename src0_T, typename src1_T>
__aicore__ inline void GemmExecMnNopingpong(const LocalTensor<dst_T>& L0c, const LocalTensor<src0_T>& src0Local,
    const LocalTensor<src1_T>& src1Local, GemmTiling tilling, const int32_t initValue)
{
    LocalTensor<src1_T> L0b;
    LocalTensor<src0_T> L0a;
    GetSingleThreadBuffer(L0a, L0b);
    event_t eventIdMToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
    SetFlag<HardEvent::M_MTE1>(eventIdMToMte1);
    for (size_t indexK = 0; indexK < tilling.kIterNum; indexK++) {
        uint32_t kBlocks = tilling.kTileBlock;
        if (indexK == tilling.kIterNum - 1) {
            kBlocks = tilling.kTailBlock;
        }
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1);
        for (size_t indexM = 0; indexM < tilling.mIterNum; indexM++) {
            // load data from l1 to l0a
            LoadL0A(kBlocks, tilling.mTileBlock, tilling, indexK, indexM, src0Local, L0a);
            for (size_t indexN = 0; indexN < tilling.nIterNum; indexN++) {
                // load data from l1 to l0b
                LoadL0B(kBlocks, tilling.nTileBlock, tilling, indexK, indexN, src1Local, L0b);
                event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
                SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                PipeBarrier<PIPE_M>();
                MmadFunc(L0a, L0b, L0c, initValue, tilling, indexK);
            }
        }
        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1);
    }
    WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1);
}

template <typename dst_T, typename src0_T, typename src1_T>
__aicore__ inline void GemmExecMnPingPong(const LocalTensor<dst_T>& L0c, const LocalTensor<src0_T>& src0Local,
    const LocalTensor<src1_T>& src1Local, GemmTiling tilling, const int32_t initValue)
{
    uint32_t ping = 1;
    LocalTensor<src0_T> L0aPing;
    LocalTensor<src0_T> L0aPong;
    LocalTensor<src1_T> L0bPing;
    LocalTensor<src1_T> L0bPong;
    GetPingPongBuffer(L0aPing, L0aPong, L0bPing, L0bPong);

    event_t eventId0 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
    event_t eventId1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
    SetFlag<HardEvent::M_MTE1>(eventId0);
    SetFlag<HardEvent::M_MTE1>(eventId1);

    for (size_t i = 0; i < tilling.kIterNum; i++) {
        uint32_t kBlocks = tilling.kTileBlock;
        if (i == tilling.kIterNum - 1) {
            kBlocks = tilling.kTailBlock;
        }
        if (ping == 1) {
            WaitFlag<HardEvent::M_MTE1>(eventId0);
            for (size_t indexM = 0; indexM < tilling.mIterNum; indexM++) {
                // load data from l1 to l0a
                LoadL0A(kBlocks, tilling.mTileBlock, tilling, i, indexM, src0Local, L0aPing);
                for (size_t indexN = 0; indexN < tilling.nIterNum; indexN++) {
                    // load data from l1 to l0b
                    LoadL0B(kBlocks, tilling.nTileBlock, tilling, i, indexN, src1Local, L0bPing);

                    event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
                    SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                    WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                    PipeBarrier<PIPE_M>();
                    MmadFunc(L0aPing, L0bPing, L0c, initValue, tilling, i);
                }
            }
            SetFlag<HardEvent::M_MTE1>(eventId0);
        } else {
            WaitFlag<HardEvent::M_MTE1>(eventId1);
            for (size_t indexM = 0; indexM < tilling.mIterNum; indexM++) {
                // load data from l1 to l0a
                LoadL0A(kBlocks, tilling.mTileBlock, tilling, i, indexM, src0Local, L0aPong);
                for (size_t indexN = 0; indexN < tilling.nIterNum; indexN++) {
                    // load data from l1 to l0b
                    LoadL0B(kBlocks, tilling.nTileBlock, tilling, i, indexN, src1Local, L0bPong);
                    event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
                    SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                    WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                    PipeBarrier<PIPE_M>();
                    MmadFunc(L0aPong, L0bPong, L0c, initValue, tilling, i);
                }
            }
            SetFlag<HardEvent::M_MTE1>(eventId1);
        }
        ping = 1 - ping;
    }

    WaitFlag<HardEvent::M_MTE1>(eventId0);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventId0);
    WaitFlag<HardEvent::M_MTE1>(eventId1);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventId1);
}

template <typename dst_T, typename src0_T, typename src1_T>
__aicore__ inline void GemmExecMn(const LocalTensor<dst_T>& L0c, const LocalTensor<src0_T>& src0Local,
    const LocalTensor<src1_T>& src1Local, GemmTiling tilling, const int32_t initValue)
{
    uint32_t needL0Bsize = tilling.roundN * tilling.dtypeSize * tilling.c0Size * tilling.kTileBlock * 2;
    uint32_t needL0Asize = tilling.roundM * tilling.dtypeSize * tilling.c0Size * tilling.kTileBlock * 2;
    if (needL0Asize > TOTAL_L0A_SIZE || needL0Bsize > TOTAL_L0B_SIZE) {
        GemmExecMnNopingpong(L0c, src0Local, src1Local, tilling, initValue);
        return;
    }
    GemmExecMnPingPong(L0c, src0Local, src1Local, tilling, initValue);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_GEMM_BASE_IMPL_H