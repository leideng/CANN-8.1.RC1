/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file tensor_utils.h
 * \brief
 */
#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#pragma once
#include "kernel_operator.h"

using namespace AscendC;

namespace FlatQuantNS {
constexpr int NUM_TWO = 2;
constexpr int NUM_THREE = 2;
constexpr int NUM_FOUR = 4;
constexpr int NUM_FIVE = 5;
constexpr int NUM_EIGHT = 8;
constexpr int NUM_TEN = 10;
constexpr int NUM_ONE_FIVE = 15;
constexpr int NUM_ONE_SIX = 16;
constexpr int NUM_SIX_FOUR = 64;
constexpr int NUM_ONE_TWO_EIGHT = 128;
constexpr int NUM_ONE_FOUR_FOUR = 144;
constexpr int NUM_ONE_SIX_ZERO = 160;
constexpr int NUM_ONE_NINE_TWO = 192;
constexpr int NUM_TWO_ZERO_EIGHT = 208;
constexpr int NUM_FIVE_ONE_TWO = 512;
constexpr int NUM_ONE_ZERO_TWO_FOUR = 1024;
constexpr int NUM_THREE_TWO = 32;
constexpr int64_t SINGLE_CALC_COUNT = NUM_THREE_TWO * NUM_ONE_ZERO_TWO_FOUR / NUM_EIGHT;
constexpr float NUM_FLOAT_SEVEN = 7.0f;

constexpr int N_PRELOAD = NUM_TWO;   // pre-process n batches (n*K_PER_VEC) before running cube
constexpr int K_PER_VEC = NUM_FOUR;  // batch number. Each loop processes K_PER_VEC*M*N

struct FlatQuantShapeInfo {
    int32_t K;
    int32_t M;
    int32_t N;        // basic shape
    int32_t K1;
    int32_t K2;         // loop start and loop end
    int32_t Mceil;
    int32_t Nceil;   // ceil shape
    int32_t procP1;         // pre-process p1 or p2
    int32_t pstart;
    int32_t prows;  // start row number of p1 or p2 matrix
    float clipRatio;
};

#define gmfloat __gm__ float *
#define gmhalf __gm__ half *
#define gmu8 __gm__ uint8_t *
#define gmbf __gm__ bfloat16_t *
#define aifunc __aicore__ inline

/* ------------- Events ------------- */

template <pipe_t p1, pipe_t p2>
class DEvent {
public:
    aifunc DEvent()
    {}
    aifunc DEvent(int e_id1, int e_id2)
    {
        id1 = (event_t)e_id1;
        id2 = (event_t)e_id2;
    }
    aifunc DEvent(event_t e_id1, event_t e_id2)
    {
        id1 = e_id1;
        id2 = e_id2;
    }
    aifunc void wait()
    {
        if (wait_cnt % NUM_TWO == 0) {
            wait_flag(p1, p2, id1);
        } else {
            wait_flag(p1, p2, id2);
        }
        wait_cnt++;
    }
    aifunc void set()
    {
        if (set_cnt % NUM_TWO == 0) {
            set_flag(p1, p2, id1);
        } else {
            set_flag(p1, p2, id2);
        }
        set_cnt++;
    }

    aifunc void wait(int &idx)
    {
        if (idx % NUM_TWO == 0) {
            wait_flag(p1, p2, id1);
        } else {
            wait_flag(p1, p2, id2);
        }
    }
    aifunc void set(int &idx)
    {
        if (idx % NUM_TWO == 0) {
            set_flag(p1, p2, id1);
        } else {
            set_flag(p1, p2, id2);
        }
    }
    aifunc void wait(int &&idx)
    {
        if (idx % NUM_TWO == 0) {
            wait_flag(p1, p2, id1);
        } else {
            wait_flag(p1, p2, id2);
        }
    }
    aifunc void set(int &&idx)
    {
        if (idx % NUM_TWO == 0) {
            set_flag(p1, p2, id1);
        } else {
            set_flag(p1, p2, id2);
        }
    }

    aifunc void setall()
    {
        set();
        set();
    }
    aifunc void setall_force()
    {
        set(0);
        set(1);
    }
    aifunc void release()
    {
        for (int i = wait_cnt; i < set_cnt; ++i) {
            wait();
        }
    }
    aifunc void release_force()
    {
        wait(0);
        wait(1);
    }

private:
    event_t id1 = (event_t)0, id2 = (event_t)1;
    int wait_cnt = 0;
    int set_cnt = 0;
};

template <pipe_t p1, pipe_t p2>
class TEvent {
public:
    aifunc TEvent()
    {}
    aifunc TEvent(int e_id1, int e_id2, int e_id3)
    {
        id1 = (event_t)e_id1;
        id2 = (event_t)e_id2;
        id3 = (event_t)e_id3;
    }
    aifunc TEvent(event_t e_id1, event_t e_id2, event_t e_id3)
    {
        id1 = e_id1;
        id2 = e_id2;
        id3 = e_id3;
    }
    aifunc void wait()
    {
        if (wait_cnt % NUM_THREE == 0) {
            wait_flag(p1, p2, id1);
        }
        if (wait_cnt % NUM_THREE == 1) {
            wait_flag(p1, p2, id2);
        }
        if (wait_cnt % NUM_THREE == NUM_TWO) {
            wait_flag(p1, p2, id3);
        }
        wait_cnt++;
    }
    aifunc void set()
    {
        if (set_cnt % NUM_THREE == 0) {
            set_flag(p1, p2, id1);
        }
        if (set_cnt % NUM_THREE == 1) {
            set_flag(p1, p2, id2);
        }
        if (set_cnt % NUM_THREE == NUM_TWO) {
            set_flag(p1, p2, id3);
        }
        set_cnt++;
    }
    aifunc void setall()
    {
        set();
        set();
        set();
    }
    aifunc void release()
    {
        for (int i = wait_cnt; i < set_cnt; ++i) {
            wait();
        }
    }

private:
    event_t id1 = (event_t)0, id2 = (event_t)1, id3 = (event_t)NUM_TWO;
    int wait_cnt = 0;
    int set_cnt = 0;
};

/* ------------- Events ------------- */

template <typename DType>
__aicore__ inline void copy_gm_to_ubuf(LocalTensor<DType> dst, GlobalTensor<DType> src, uint8_t sid,
    uint16_t blockCount, uint16_t blockLen, uint16_t srcStride, uint16_t dstStride)
{
    DataCopy(dst, src, DataCopyParams(blockCount, blockLen, srcStride, dstStride));
}

template <typename DType>
__aicore__ inline void copy_ubuf_to_ubuf(LocalTensor<DType> dst, LocalTensor<DType> src, uint8_t sid,
    uint16_t blockCount, uint16_t blockLen, uint16_t srcStride, uint16_t dstStride)
{
    DataCopy(dst, src, DataCopyParams(blockCount, blockLen, srcStride, dstStride));
}

template <typename DType>
__aicore__ inline void copy_ubuf_to_gm(GlobalTensor<DType> dst, LocalTensor<DType> src, uint8_t sid,
    uint16_t blockCount, uint16_t blockLen, uint16_t srcStride, uint16_t dstStride)
{
    DataCopy(dst, src, DataCopyParams(blockCount, blockLen, srcStride, dstStride));
}

template <typename DType>
__aicore__ inline void copy_ubuf_to_gm(GlobalTensor<DType> dst, LocalTensor<DType> src, uint8_t sid,
    uint16_t blockCount, uint16_t blockLen, uint16_t srcStride, uint16_t dstStride, uint16_t byteMode)
{
    if (byteMode == 1) {
        if (blockLen % 32 == 0) {
            DataCopy(dst, src, blockLen / sizeof(DType));
        } else {
            DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
            DataCopyPad(dst, src, copyParams);
        }
        return;
    }
    copy_ubuf_to_gm(dst, src, sid, blockCount, blockLen, srcStride, dstStride);
}

template <typename DType>
__aicore__ inline void vabs(LocalTensor<DType> dst, LocalTensor<DType> src, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
{
    Abs<DType, false>(dst,
        src,
        (uint64_t)0,
        repeat,
        UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}

template <typename DType>
__aicore__ inline void vbrcb(
    LocalTensor<DType> dst, LocalTensor<DType> src, int16_t dstBlockStride, uint16_t dstRepeatStride, uint8_t repeat)
{
    Brcb<DType>(dst, src, repeat, BrcbRepeatParams(dstBlockStride, dstRepeatStride));
}

template <typename DType>
__aicore__ inline void vcmax(LocalTensor<DType> dst, LocalTensor<DType> src, int32_t repeat, int32_t dstRepeatStride,
    int32_t srcBlockStride, int32_t srcRepeatStride, ReduceOrder order)
{
    WholeReduceMax<DType, false>(dst, src, (int32_t)0, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride, order);
}

template <typename DType>
__aicore__ inline void vmul(LocalTensor<DType> dst, LocalTensor<DType> src0, LocalTensor<DType> src1, uint8_t repeat,
    uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
    uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    Mul<DType, false>(dst,
        src0,
        src1,
        AscendC::MASK_PLACEHOLDER,
        repeat,
        BinaryRepeatParams(
            dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride));
}

template <typename DType>
__aicore__ inline void vdiv(LocalTensor<DType> dst, LocalTensor<DType> src0, LocalTensor<DType> src1, uint8_t repeat,
    uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
    uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    Div<DType, false>(dst,
        src0,
        src1,
        (uint64_t)0,
        repeat,
        BinaryRepeatParams(
            dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride));
}

template <typename DType>
__aicore__ inline void copy_gm_to_cbuf(LocalTensor<DType> dst, GlobalTensor<DType> src, uint8_t sid, uint16_t nBurst,
    uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride, pad_t padMode)
{
    DataCopy(dst, src, DataCopyParams(nBurst, lenBurst, srcStride, dstStride));
}

template <typename DType>
__aicore__ inline void load_cbuf_to_ca(LocalTensor<DType> dst, LocalTensor<DType> src, uint16_t baseIdx,
    uint8_t repeat, uint16_t srcStride, uint8_t sid, bool transpose)
{
    LoadData(dst, src, LoadData2DParams(baseIdx, repeat, srcStride, sid, 0, transpose, 0));
}

template <typename DType>
__aicore__ inline void load_cbuf_to_cb(LocalTensor<DType> dst, LocalTensor<DType> src, uint16_t baseIdx,
    uint8_t repeat, uint16_t srcStride, uint8_t sid, bool transpose)
{
    LoadData(dst, src, LoadData2DParams(baseIdx, repeat, srcStride, sid, 0, transpose, 0));
}

template <typename CType, typename DType>
__aicore__ inline void copy_matrix_cc_to_cbuf(LocalTensor<DType> dst, LocalTensor<CType> src, uint8_t sid,
    uint16_t NSize, uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride, uint8_t UnitFlagMode,
    QuantMode_t QuantPRE, uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN)
{
    DataCopyCO12DstParams dataCopyParams(
        NSize, MSize, dstStride_dst_D, srcStride, QuantPRE, ReLUPRE, channelSplit, NZ2ND_EN);
    dataCopyParams.unitFlag = UnitFlagMode;
    DataCopy(dst, src, dataCopyParams);
}

template <typename CType, typename DType>
__aicore__ inline void copy_matrix_cc_to_gm(GlobalTensor<DType> dst, LocalTensor<CType> src, uint8_t sid,
    uint16_t NSize, uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride, uint8_t UnitFlagMode,
    QuantMode_t QuantPRE, uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN)
{
    DataCopyCO12DstParams dataCopyParams(
        NSize, MSize, dstStride_dst_D, srcStride, QuantPRE, ReLUPRE, channelSplit, NZ2ND_EN);
    dataCopyParams.unitFlag = UnitFlagMode;
    DataCopy(dst, src, dataCopyParams);
}

template <typename CType, typename DType>
__aicore__ inline void mad(LocalTensor<CType> c, LocalTensor<DType> a, LocalTensor<DType> b, uint16_t m, uint16_t k,
    uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal)
{
    MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.cmatrixInitVal = true;
    mmadParams.unitFlag = unitFlag;
    Mmad(c, a, b, mmadParams);
}
}  // namespace FlatQuantNS

#endif  // TENSOR_UTILS_H