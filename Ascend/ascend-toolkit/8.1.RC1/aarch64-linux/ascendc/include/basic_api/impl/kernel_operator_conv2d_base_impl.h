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
 * \file kernel_operator_conv2d_base_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_CONV2D_BASE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_CONV2D_BASE_IMPL_H
#if ASCENDC_CPU_DEBUG
#include <unordered_set>
#endif
#include "kernel_tensor.h"
#include "kernel_operator_mm_base_impl.h"
#include "kernel_operator_gemm_base_impl.h"
#include "kernel_struct_conv2d.h"
#include "kernel_struct_mm.h"

namespace AscendC {
#if ASCENDC_CPU_DEBUG

const std::unordered_set<std::string> CONV2D_SUPPORT_TYPE { "s8s8s32", "f16f16f32", "f16f16f16" };

__aicore__ inline bool CheckConv2DRange(const uint32_t* rangePair, const uint32_t num)
{
    if ((num >= rangePair[0]) && (num <= rangePair[1])) {
        return true;
    }
    return false;
}

__aicore__ inline bool CheckConv2DParamsRange(Conv2dParams& conv2dParams, Conv2dTilling& tilling)
{
    uint32_t cinRange[] = {tilling.c0Size, 4096};
    uint32_t coutRange[] = {16, 4096};

    uint32_t loopTime[] = {CONV2D_IMG_SIZE, CONV2D_KERNEL_SIZE, CONV2D_STRIDE, CONV2D_PAD, CONV2D_DILATION};
    // imgSizeRange, kernelSizeRange, strideRange, padRange, dilationRange
    uint32_t rangePair[][2] = { {1, 4096}, {1, 255}, {1, 63}, {0, 255}, {1, 255} };

    uint32_t* arrayPtr[] = {conv2dParams.imgShape, conv2dParams.kernelShape, conv2dParams.stride,
                            conv2dParams.padList, conv2dParams.dilation};

    if (conv2dParams.cout % BLOCK_CUBE != 0) {
        return false;
    }

    if ((CheckConv2DRange(cinRange, conv2dParams.cin) == false) ||
        CheckConv2DRange(coutRange, conv2dParams.cout) == false) {
        return false;
    }

    for (size_t i = 0; i < sizeof(loopTime) / sizeof(loopTime[0]); i++) {
        for (size_t j = 0; j < loopTime[i]; j++) {
            if (CheckConv2DRange(rangePair[i], arrayPtr[i][j]) == false) {
                return false;
            }
        }
    }
    return true;
}

template <typename dst_T, typename src_T>
__aicore__ inline bool CheckConv2DOverflow(const LocalTensor<dst_T>& dstLocal, const LocalTensor<src_T>& featureMap,
    const LocalTensor<src_T>& weight, Conv2dParams& conv2dParams, Conv2dTilling& tilling)
{
    // check l0c
    uint32_t roundM = tilling.roundM;
    uint32_t roundN = tilling.roundN;
    uint32_t roundK = tilling.roundK;

    uint32_t kTileBlock = tilling.kTileBlock;

    uint32_t needElementLoc = roundM * roundN * sizeof(PrimT<dst_T>);
    if (needElementLoc > TOTAL_L0C_SIZE) {
        return false;
    }

    // check l0a:
    uint32_t needElementL0a = roundM * roundK * sizeof(PrimT<src_T>);

    // check l0b:
    uint32_t needElementL0b = roundN * roundK * sizeof(PrimT<src_T>);
    if ((needElementL0b + needElementL0a) > TOTAL_L1_SIZE) {
        return false;
    }

    uint32_t minElementL0a = roundM * kTileBlock * tilling.c0Size * sizeof(PrimT<src_T>);
    uint32_t minElementL0b = roundN * kTileBlock * tilling.c0Size * sizeof(PrimT<src_T>);

    if (minElementL0a > TOTAL_L0A_SIZE) {
        return false;
    }

    if (minElementL0b > TOTAL_L0B_SIZE) {
        return false;
    }

    return true;
}

template <typename dst_T, typename src_T>
__aicore__ inline bool CheckConv2DParams(const LocalTensor<dst_T>& dstLocal, const LocalTensor<dst_T>& bias,
    const LocalTensor<src_T>& featureMap, const LocalTensor<src_T>& weight, Conv2dParams& conv2dParams,
    Conv2dTilling& tilling)
{
    if (tilling.c0Size == 0) {
        return false;
    }

    // check scope
    const Hardware src0Scope = GetPhyType((TPosition)featureMap.GetPosition());
    const Hardware src1Scope = GetPhyType((TPosition)weight.GetPosition());
    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    const Hardware biasScope = GetPhyType((TPosition)bias.GetPosition());

    if (src0Scope != Hardware::L1 || src1Scope != Hardware::L1) {
        return false;
    }

    if ((dstScope != Hardware::UB) && (dstScope != Hardware::L0C)) {
        return false;
    }

    // check dtype
    std::string dtypeStr = GetTypeStr(featureMap) + GetTypeStr(weight) + GetTypeStr(dstLocal);
    if (CONV2D_SUPPORT_TYPE.find(dtypeStr) == CONV2D_SUPPORT_TYPE.end()) {
        return false;
    }

    // check range
    bool check = CheckConv2DParamsRange(conv2dParams, tilling);
    if (check == false) {
        return false;
    }

    // check overflow
    if (!CheckConv2DOverflow(dstLocal, featureMap, weight, conv2dParams, tilling)) {
        return false;
    }

    return true;
}
#endif

template <typename T> __aicore__ inline void GetTypeforC0(Conv2dParams& conv2dParams, Conv2dTilling& tilling)
{
    if (IsSameType<PrimT<T>, int8_t>::value) {
        tilling.c0Size = 32;
        tilling.dTypeSize = 1;
    } else if (IsSameType<PrimT<T>, half>::value) {
        tilling.c0Size = 16;   // 32Byte-block
        tilling.dTypeSize = 2; // sizeof(dtype)
    } else {
        tilling.c0Size = 0;
        tilling.dTypeSize = 0;
    }
}

__aicore__ inline void CalculateConv2dTiling(Conv2dTilling& tilling)
{
    tilling.mBlockNum = DivCeil(tilling.mNum, tilling.blockSize);
    tilling.nBlockNum = DivCeil(tilling.nNum, tilling.blockSize);
    tilling.kBlockNum = DivCeil(tilling.kNum, tilling.c0Size);

    tilling.roundM = DivCeil(tilling.mNum, tilling.blockSize) * tilling.blockSize; // blockSize = 16(16X16)
    tilling.roundN = DivCeil(tilling.nNum, tilling.blockSize) * tilling.blockSize;
    tilling.roundK = DivCeil(tilling.kNum, tilling.c0Size) * tilling.c0Size; // c0Size = 16 || c0Size = 32

    uint32_t k0a = TOTAL_L0A_SIZE / 2 / (tilling.roundM * tilling.dTypeSize);
    uint32_t k0b = TOTAL_L0B_SIZE / 2 / (tilling.roundN * tilling.dTypeSize);
    uint32_t k0 = k0a > k0b ? k0b : k0a;
    k0 = k0 > tilling.kNum ? tilling.kNum : k0;

    tilling.kTileBlock = k0 / tilling.c0Size;
    if (tilling.kTileBlock == 0) {
        tilling.kTileBlock = 1;
    }

    tilling.mIterNum = 1;
    tilling.nIterNum = 1;
    tilling.kIterNum = DivCeil(tilling.kBlockNum, tilling.kTileBlock);

    tilling.mTileBlock = DivCeil(tilling.mBlockNum, tilling.mIterNum);
    tilling.nTileBlock = DivCeil(tilling.nBlockNum, tilling.nIterNum);

    tilling.mTileNums = tilling.mTileBlock * tilling.blockSize;

    tilling.mHasTail = (tilling.howo != tilling.mIterNum * tilling.mTileBlock * tilling.blockSize) ? true : false;
    tilling.kHasTail = (tilling.kBlockNum < tilling.kIterNum * tilling.kTileBlock) ? true : false;
    tilling.nHasTail = (tilling.nBlockNum < tilling.nIterNum * tilling.nTileBlock) ? true : false;

    tilling.mTailBlock = tilling.mBlockNum - (tilling.mIterNum - 1) * tilling.mTileBlock; // mTailBlock <= mBlockNum
    tilling.mTailNums = tilling.howo - (tilling.mIterNum - 1) * tilling.mTileBlock * tilling.blockSize;

    tilling.kTailBlock = tilling.kBlockNum - (tilling.kIterNum - 1) * tilling.kTileBlock;
    tilling.nTailBlock = tilling.nBlockNum - (tilling.nIterNum - 1) * tilling.nTileBlock;
}

template <typename T>
__aicore__ inline void LoadL0AForConv2DV1(uint32_t kBlocks, uint32_t indexK, uint32_t mBlocks, uint32_t indexM,
    Conv2dParams& conv2dParams, Conv2dTilling& tilling, const LocalTensor<T>& src0Local, const LocalTensor<T>& L0a)
{
    uint32_t cinPos = indexK * tilling.kTileBlock;
    // load by column
    for (size_t index = 0; index < tilling.mTileBlock; index++) {
        uint32_t hoWoPos = (indexM * tilling.mTileBlock + index) * tilling.blockSize;
        uint32_t hoIdx = hoWoPos / tilling.wo;
        uint32_t woIdx = hoWoPos % tilling.wo;
        uint32_t hiIdx = hoIdx * tilling.strideH;
        uint32_t wiIdx = woIdx * tilling.strideW;
        // we load the whole row in 1 load3d
        uint32_t c1Idx = cinPos / (tilling.height * tilling.width);
        uint32_t kHwIdx = cinPos % (tilling.height * tilling.width);
        uint32_t l0aIdx = index * kBlocks * tilling.blockSize * tilling.c0Size;
        uint32_t disableC1 = 0;
        uint32_t c1Offset = c1Idx * tilling.c0Size * tilling.hi * tilling.wi;

        LoadData3DParamsV1<PrimT<T>> params;

        for (size_t i = 0; i < PAD_SIZE; i++) {
            params.padList[i] = conv2dParams.padList[i];
        }

        params.l1H = tilling.hi;
        params.l1W = tilling.wi;
        params.c1Index = disableC1;
        params.fetchFilterW = kHwIdx % tilling.width;
        params.fetchFilterH = kHwIdx / tilling.width;
        params.leftTopW = wiIdx - params.padList[0];
        params.leftTopH = hiIdx - params.padList[2];
        params.strideW = tilling.strideW;
        params.strideH = tilling.strideH;
        params.filterW = tilling.width;
        params.filterH = tilling.height;
        params.dilationFilterW = tilling.dilationW;
        params.dilationFilterH = tilling.dilationH;
        params.jumpStride = 1;
        params.repeatMode = 0;
        params.repeatTime = kBlocks;
        params.cSize = 0;
        params.padValue = 0;

        LoadDataImpl(L0a[l0aIdx], src0Local[c1Offset], params);
    }
}

template <typename T>
__aicore__ inline void LoadL0AForConv2DV2(uint32_t kBlocks, uint32_t indexK, uint32_t mBlocks, uint32_t indexM,
    Conv2dParams& conv2dParams, Conv2dTilling& tilling, const LocalTensor<T>& src0Local, const LocalTensor<T>& L0a)
{
    // data l0a size only need hw_actual_size * cin_actual blocks,
    // but for performance of ping pong with tail block, apply m_tile_block * cin_actual blocks
    uint32_t kStartPt = indexK * kBlocks * tilling.c0Size;
    uint32_t mStartPt = indexM * mBlocks;
    uint32_t channelSize = conv2dParams.cin;

    LoadData3DParamsV2<PrimT<T>> params;

    for (size_t i = 0; i < PAD_SIZE; i++) {
        params.padList[i] = conv2dParams.padList[i];
    }

    params.l1H = tilling.hi;
    params.l1W = tilling.wi;
    params.channelSize = channelSize;
    params.kExtension = kBlocks * tilling.c0Size;
    params.mExtension = mBlocks;
    params.kStartPt = kStartPt;
    params.mStartPt = mStartPt;
    params.strideW = tilling.strideW;
    params.strideH = tilling.strideH;
    params.filterW = tilling.width;
    params.filterH = tilling.height;
    params.dilationFilterW = tilling.dilationW;
    params.dilationFilterH = tilling.dilationH;
    params.enTranspose = false;
    params.enSmallK = false;
    params.padValue = 0;
    params.filterSizeW = false;
    params.filterSizeH = false;
    params.fMatrixCtrl = false;

    LoadDataImpl(L0a, src0Local, params);
}

template <typename T>
__aicore__ inline void LoadL0AForConv2D(uint32_t kBlocks, uint32_t indexK, uint32_t mBlocks, uint32_t indexM,
    Conv2dParams& conv2dParams, Conv2dTilling& tilling, const LocalTensor<T>& src0Local, const LocalTensor<T>& L0a)
{
#if __CCE_AICORE__ < 220
    LoadL0AForConv2DV1(kBlocks, indexK, mBlocks, indexM, conv2dParams, tilling, src0Local, L0a);
#else
    LoadL0AForConv2DV2(kBlocks, indexK, mBlocks, indexM, conv2dParams, tilling, src0Local, L0a);
#endif
}

template <typename T>
__aicore__ inline void LoadL0BForConv2D(uint32_t kBlocks, uint32_t nBlocks, uint32_t indexK, uint32_t indexN,
    Conv2dTilling& tilling, const LocalTensor<T>& src1Local, const LocalTensor<T>& L0b)
{
    if (tilling.nIterNum == 1) {
        // load one column at once
        uint32_t wSize = tilling.blockSize * tilling.c0Size;
        uint32_t wIdx = (indexK * tilling.kTileBlock * tilling.nBlockNum + indexN * tilling.nTileBlock) * wSize;
        LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = kBlocks * nBlocks;
        params.srcStride = 1;
        LoadDataImpl(L0b, src1Local[wIdx], params);
    } else {
        // load data row by row
        for (size_t index = 0; index < kBlocks; index++) {
            uint32_t wSize = indexN * tilling.nTileBlock * tilling.blockSize * tilling.c0Size;
            uint32_t wIdx =
                (indexK * tilling.kTileBlock + index) * tilling.nBlockNum * tilling.blockSize * tilling.c0Size + wSize;
            uint32_t l0bIdx = index * nBlocks * tilling.blockSize * tilling.c0Size;
            LoadData2DParams params;
            params.startIndex = 0;
            params.repeatTimes = nBlocks;
            params.srcStride = 1;
            LoadDataImpl(L0b[l0bIdx], src1Local[wIdx], params);
        }
    }
}

template <typename dst_T, typename src_T>
__aicore__ inline void MmadFuncForConv2D(const LocalTensor<src_T>& L0a, const LocalTensor<src_T>& L0b,
    const LocalTensor<dst_T>& L0c, const LocalTensor<dst_T>& bias, Conv2dParams& conv2dParams, Conv2dTilling tilling,
    uint32_t kBlocks, uint32_t mBlocks, uint32_t nBlocks, uint32_t indexK, uint32_t indexM, uint32_t indexN)
{
    // only care data in K dim
    uint32_t bSize = tilling.blockSize * tilling.blockSize;
    uint32_t dstFlattenIdx = (indexN * tilling.mBlockNum * tilling.nTileBlock + indexM * tilling.mTileBlock) * bSize;
    uint32_t hwActualSize = mBlocks;

    // for m_extension is 1, mmad is GEMV mode, GEMV mode must L0A shape M is 1
    // but current L0A shape M is not 1, set hw_actual_size to 2, mmad work in GEMM mode,
    // set to 2 won't inspect mmad result
    if (hwActualSize == 1) {
        hwActualSize = 2;
    }

    MmadParams mmadParams;

    mmadParams.m = hwActualSize;
    mmadParams.k = kBlocks * tilling.c0Size;
    mmadParams.n = nBlocks * tilling.blockSize;
    mmadParams.isBias = 1;

    if ((indexK == 0) && (conv2dParams.initY == 0)) {
        mmadParams.isBias = 0;
    }

    if ((indexK == 0) && (conv2dParams.initY == 2)) {
        mmadParams.isBias = 0;
        uint32_t biasOffset = nBlocks * indexN * 16;
        // bias size is Cout, max Cout is 4096, so nburst is 1 is enough to data move
        uint32_t burstLenUnit = 64;
        uint32_t extent = sizeof(PrimT<dst_T>) * nBlocks * 16;
        uint32_t burstLen = extent / burstLenUnit;
        BroadCastVecToMM(L0c[dstFlattenIdx], bias[biasOffset], 1, burstLen, 0, 0);
        event_t eventIdVToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_M));
        SetFlag<HardEvent::V_M>(eventIdVToM);
        WaitFlag<HardEvent::V_M>(eventIdVToM);
    }

    MmadImpl(L0c[dstFlattenIdx], L0a, L0b, mmadParams);
}

template <typename dst_T, typename src_T>
__aicore__ inline void Conv2DExecNmNopingpong(const LocalTensor<dst_T>& L0c, const LocalTensor<dst_T>& bias,
    const LocalTensor<src_T>& src0Local, const LocalTensor<src_T>& src1Local, Conv2dParams& conv2dParams,
    Conv2dTilling& tilling)
{
    LocalTensor<src_T> L0b;
    LocalTensor<src_T> L0a;
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
            LoadL0BForConv2D(kBlocks, tilling.nTileBlock, indexK, indexN, tilling, src1Local, L0b);
            for (size_t indexM = 0; indexM < tilling.mIterNum; indexM++) {
                // load data from l1 to l0a
                LoadL0AForConv2D(kBlocks, indexK, tilling.mTileNums, indexM, conv2dParams, tilling, src0Local, L0a);
                event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
                SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                PipeBarrier<PIPE_M>();
                MmadFuncForConv2D(L0a, L0b, L0c, bias, conv2dParams, tilling, kBlocks, tilling.mTileNums,
                    tilling.nTileBlock, indexK, indexM, indexN);
            }
        }
        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1);
    }
    WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1);
}

__aicore__ inline void SetWaitFlagMte1ToM()
{
    event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
    SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
    WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
    PipeBarrier<PIPE_M>();
}
 
__aicore__ inline void PingPongRealeaseEvent(event_t eventId0, event_t eventId1)
{
    WaitFlag<HardEvent::M_MTE1>(eventId0);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventId0);
    WaitFlag<HardEvent::M_MTE1>(eventId1);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventId1);
}

template <typename dst_T, typename src_T>
__aicore__ inline void Conv2DExecNmPingPong(const LocalTensor<dst_T>& L0c, const LocalTensor<dst_T>& bias,
    const LocalTensor<src_T>& src0Local, const LocalTensor<src_T>& src1Local, Conv2dParams& conv2dParams,
    Conv2dTilling& tilling)
{
    uint32_t ping = 1;
    LocalTensor<src_T> L0aPing;
    LocalTensor<src_T> L0bPing;
    LocalTensor<src_T> L0aPong;
    LocalTensor<src_T> L0bPong;
    GetPingPongBuffer(L0aPing, L0aPong, L0bPing, L0bPong);

    event_t eventId0 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
    event_t eventId1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());

    SetFlag<HardEvent::M_MTE1>(eventId0);
    SetFlag<HardEvent::M_MTE1>(eventId1);

    for (size_t indexK = 0; indexK < tilling.kIterNum; indexK++) {
        uint32_t kBlocks = tilling.kTileBlock;
        if (indexK == tilling.kIterNum - 1) {
            kBlocks = tilling.kTailBlock;
        }
        if (ping == 1) {
            WaitFlag<HardEvent::M_MTE1>(eventId0);
            for (size_t indexN = 0; indexN < tilling.nIterNum; indexN++) {
                // load data from l1 to l0b
                LoadL0BForConv2D(kBlocks, tilling.nTileBlock, indexK, indexN, tilling, src1Local, L0bPing);
                for (size_t indexM = 0; indexM < tilling.mIterNum; indexM++) {
                    // load data from l1 to l0a
                    LoadL0AForConv2D(kBlocks, indexK, tilling.mTileNums, indexM, conv2dParams, tilling, src0Local,
                        L0aPing);
                    SetWaitFlagMte1ToM();
                    MmadFuncForConv2D(L0aPing, L0bPing, L0c, bias, conv2dParams, tilling, kBlocks, tilling.mTileNums,
                        tilling.nTileBlock, indexK, indexM, indexN);
                }
            }
            SetFlag<HardEvent::M_MTE1>(eventId0);
        } else {
            WaitFlag<HardEvent::M_MTE1>(eventId1);
            for (size_t indexN = 0; indexN < tilling.nIterNum; indexN++) {
                // load data from l1 to l0b
                LoadL0BForConv2D(kBlocks, tilling.nTileBlock, indexK, indexN, tilling, src1Local, L0bPong);
                for (size_t indexM = 0; indexM < tilling.mIterNum; indexM++) {
                    // load data from l1 to l0a
                    LoadL0AForConv2D(kBlocks, indexK, tilling.mTileNums, indexM, conv2dParams, tilling, src0Local,
                        L0aPong);
                    SetWaitFlagMte1ToM();
                    MmadFuncForConv2D(L0aPong, L0bPong, L0c, bias, conv2dParams, tilling, kBlocks, tilling.mTileNums,
                        tilling.nTileBlock, indexK, indexM, indexN);
                }
            }
            SetFlag<HardEvent::M_MTE1>(eventId1);
        }
        ping = 1 - ping;
    }

    PingPongRealeaseEvent(eventId0, eventId1);
}

template <typename dst_T, typename src_T>
__aicore__ inline void Conv2DExecNm(const LocalTensor<dst_T>& L0c, const LocalTensor<dst_T>& bias,
    const LocalTensor<src_T>& src0Local, const LocalTensor<src_T>& src1Local, Conv2dParams& conv2dParams,
    Conv2dTilling& tilling)
{
    uint32_t needL0Asize = tilling.roundM * tilling.dTypeSize * tilling.c0Size * tilling.kTileBlock * 2;
    uint32_t needL0Bsize = tilling.roundN * tilling.dTypeSize * tilling.c0Size * tilling.kTileBlock * 2;
    if (needL0Asize > TOTAL_L0A_SIZE || needL0Bsize > TOTAL_L0B_SIZE) {
        Conv2DExecNmNopingpong(L0c, bias, src0Local, src1Local, conv2dParams, tilling);
        return;
    }
    Conv2DExecNmPingPong(L0c, bias, src0Local, src1Local, conv2dParams, tilling);
}

template <typename dst_T, typename src_T>
__aicore__ inline void Conv2DExecMnNopingpong(const LocalTensor<dst_T>& L0c, const LocalTensor<dst_T>& bias,
    const LocalTensor<src_T>& src0Local, const LocalTensor<src_T>& src1Local, Conv2dParams& conv2dParams,
    Conv2dTilling& tilling)
{
    LocalTensor<src_T> L0a;
    LocalTensor<src_T> L0b;
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
            LoadL0AForConv2D(kBlocks, indexK, tilling.mTileNums, indexM, conv2dParams, tilling, src0Local, L0a);
            for (size_t indexN = 0; indexN < tilling.nIterNum; indexN++) {
                // load data from l1 to l0b
                LoadL0BForConv2D(kBlocks, tilling.nTileBlock, indexK, indexN, tilling, src1Local, L0b);
                event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
                SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
                PipeBarrier<PIPE_M>();
                MmadFuncForConv2D(L0a, L0b, L0c, bias, conv2dParams, tilling, kBlocks, tilling.mTileNums,
                    tilling.nTileBlock, indexK, indexM, indexN);
            }
        }
        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1);
    }
    WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1);
}

template <typename dst_T, typename src_T>
__aicore__ inline void Conv2DExecMnPingPong(const LocalTensor<dst_T>& L0c, const LocalTensor<dst_T>& bias,
    const LocalTensor<src_T>& src0Local, const LocalTensor<src_T>& src1Local, Conv2dParams& conv2dParams,
    Conv2dTilling& tilling)
{
    uint32_t ping = 1;
    LocalTensor<src_T> L0aPing;
    LocalTensor<src_T> L0aPong;
    LocalTensor<src_T> L0bPing;
    LocalTensor<src_T> L0bPong;
    GetPingPongBuffer(L0aPing, L0aPong, L0bPing, L0bPong);

    event_t eventId0 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
    event_t eventId1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
    SetFlag<HardEvent::M_MTE1>(eventId0);
    SetFlag<HardEvent::M_MTE1>(eventId1);

    for (size_t indexK = 0; indexK < tilling.kIterNum; indexK++) {
        uint32_t kBlocks = tilling.kTileBlock;
        if (indexK == tilling.kIterNum - 1) {
            kBlocks = tilling.kTailBlock;
        }
        if (ping == 1) {
            WaitFlag<HardEvent::M_MTE1>(eventId0);
            for (size_t indexM = 0; indexM < tilling.mIterNum; indexM++) {
                // load data from l1 to l0a
                LoadL0AForConv2D(kBlocks, indexK, tilling.mTileNums, indexM, conv2dParams, tilling, src0Local, L0aPing);
                for (size_t indexN = 0; indexN < tilling.nIterNum; indexN++) {
                    // load data from l1 to l0b
                    LoadL0BForConv2D(kBlocks, tilling.nTileBlock, indexK, indexN, tilling, src1Local, L0bPing);
                    SetWaitFlagMte1ToM();
                    MmadFuncForConv2D(L0aPing, L0bPing, L0c, bias, conv2dParams, tilling, kBlocks, tilling.mTileNums,
                        tilling.nTileBlock, indexK, indexM, indexN);
                }
            }
            SetFlag<HardEvent::M_MTE1>(eventId0);
        } else {
            WaitFlag<HardEvent::M_MTE1>(eventId1);
            for (size_t indexM = 0; indexM < tilling.mIterNum; indexM++) {
                // load data from l1 to l0a
                LoadL0AForConv2D(kBlocks, indexK, tilling.mTileNums, indexM, conv2dParams, tilling, src0Local, L0aPong);
                for (size_t indexN = 0; indexN < tilling.nIterNum; indexN++) {
                    // load data from l1 to l0b
                    LoadL0BForConv2D(kBlocks, tilling.nTileBlock, indexK, indexN, tilling, src1Local, L0bPong);
                    SetWaitFlagMte1ToM();
                    MmadFuncForConv2D(L0aPong, L0bPong, L0c, bias, conv2dParams, tilling, kBlocks, tilling.mTileNums,
                        tilling.nTileBlock, indexK, indexM, indexN);
                }
            }
            SetFlag<HardEvent::M_MTE1>(eventId1);
        }
        ping = 1 - ping;
    }

    PingPongRealeaseEvent(eventId0, eventId1);
}

template <typename dst_T, typename src_T>
__aicore__ inline void Conv2DExecMn(const LocalTensor<dst_T>& L0c, const LocalTensor<dst_T>& bias,
    const LocalTensor<src_T>& src0Local, const LocalTensor<src_T>& src1Local, Conv2dParams& conv2dParams,
    Conv2dTilling& tilling)
{
    uint32_t needL0Bsize = tilling.roundN * tilling.dTypeSize * tilling.c0Size * tilling.kTileBlock * 2;
    uint32_t needL0Asize = tilling.roundM * tilling.dTypeSize * tilling.c0Size * tilling.kTileBlock * 2;
    if (needL0Asize > TOTAL_L0A_SIZE || needL0Bsize > TOTAL_L0B_SIZE) {
        Conv2DExecMnNopingpong(L0c, bias, src0Local, src1Local, conv2dParams, tilling);
        return;
    }
    Conv2DExecMnPingPong(L0c, bias, src0Local, src1Local, conv2dParams, tilling);
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_CONV2D_BASE_IMPL_H
