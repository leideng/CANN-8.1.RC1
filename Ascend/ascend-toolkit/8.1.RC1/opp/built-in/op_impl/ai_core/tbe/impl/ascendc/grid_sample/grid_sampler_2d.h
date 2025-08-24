/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file grid_sampler_2d.h
 * \brief
 */
#ifndef GIRD_SAMPLER_2D
#define GIRD_SAMPLER_2D

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace GridSample {

using namespace AscendC;

template <typename T>
class GridSampler2D {
public:
    __aicore__ inline GridSampler2D(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace,
                                const GridSampleTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const GridSampleTilingData* tilingData);
    __aicore__ inline void PerLoopCompute(int32_t nIdx, int32_t hwIdx, int32_t calHWElems);
    __aicore__ inline void ComputeWeightSub(LocalTensor<float> w1Ub, LocalTensor<float> w2Ub, LocalTensor<float> x1Ub,
                                            LocalTensor<float> x2Ub, LocalTensor<float> y1Ub, LocalTensor<float> y2Ub);
    __aicore__ inline void ClipCoordinates(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
                                           LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb,
                                           LocalTensor<int32_t> coorUb, LocalTensor<uint8_t> weightMaskUb);
    __aicore__ inline void CoordinatesFrameRange(LocalTensor<int32_t> iIntUb, int32_t upBound);
    __aicore__ inline void CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
                                                       LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb,
                                                       LocalTensor<uint8_t> maskTmpXUb,
                                                       LocalTensor<uint8_t> maskTmpYUb);
    __aicore__ inline void CoordinatesSelectScalar(LocalTensor<float> iFpUb, LocalTensor<float> oFpUb,
                                                   LocalTensor<uint8_t> maskUb, const float scalarVal, const uint32_t calNum);
    __aicore__ inline void CoordinatesSelectTensor(LocalTensor<float> src0, LocalTensor<float> src1,
                                                   LocalTensor<float> coorUb, LocalTensor<uint8_t> maskUb);
    __aicore__ inline void Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb);
    __aicore__ inline void BorderClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb);
    __aicore__ inline void ReflectClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb);
    __aicore__ inline void ReflectCoordinatesGeneral(LocalTensor<float> iFpUb, LocalTensor<float> coorSubUb,
                                                     LocalTensor<float> extraFpUb, LocalTensor<float> fmodFpUb,
                                                     LocalTensor<uint8_t> maskUb, LocalTensor<float> tmpFpUb,
                                                     LocalTensor<int32_t> tmpIntUb, const int64_t twiceLow,
                                                     const int64_t twiceHigh);
    __aicore__ inline void MTE2ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
                                       int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb,
                                       LocalTensor<float> xLocal);
    __aicore__ inline void MTE2ForNHWC(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
                                       int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb,
                                       LocalTensor<float> xLocal);
    __aicore__ inline void OutTranspose(int32_t channelAlign, LocalTensor<float> xLocal, LocalTensor<float> outValueUb);
    __aicore__ inline void MTE3ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
                                       int32_t hwIdx, int32_t loopOffset, int32_t loopElems,
                                       int64_t outBaseOffset, LocalTensor<float> weightUb,
                                       LocalTensor<float> outValueUb, bool isAutomicAdd);
    __aicore__ inline void PointBilinear(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
                                         LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb,
                                         LocalTensor<uint8_t> weightMaskUb, LocalTensor<float> outValueUb,
                                         bool isAutomicAdd);

   private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> gridQueue_;

    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> inputXYFPBuf_;
    TBuf<QuePosition::VECCALC> inputXIntBuf_;
    TBuf<QuePosition::VECCALC> inputYIntBuf_;
    TBuf<QuePosition::VECCALC> inputXFpBuf_;
    TBuf<QuePosition::VECCALC> inputYFpBuf_;
    TBuf<QuePosition::VECCALC> weightBuf_;
    TBuf<QuePosition::VECCALC> weightTmpBuf_;
    TBuf<QuePosition::VECCALC> weightTmp1Buf_;
    TBuf<QuePosition::VECCALC> weightTmp2Buf_;
    TBuf<QuePosition::VECCALC> weightTmp3Buf_;
    TBuf<QuePosition::VECCALC> coorBuf_;
    TBuf<QuePosition::VECCALC> coorTmpBuf_;
    TBuf<QuePosition::VECCALC> intTmpBuf_;
    TBuf<QuePosition::VECCALC> outValueBuf_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> weightMaskBuf_;
    TBuf<QuePosition::VECCALC> modBuf_;
    TBuf<QuePosition::VECCALC> extraBuf_;
    TBuf<QuePosition::VECCALC> outTmpBuf_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmGrid_;
    GlobalTensor<T> gmWorkspace_;
    GlobalTensor<T> gmY_;

    const int64_t TRANSE_REP_STRIDE = 128;
    const int64_t B32_MASK = 64;
    const int64_t CHANNEL_BLOCK = 64;
    const int32_t TRANSE_MUL_WEGHT_LOOPS = 2;

    const int64_t X_UB_SIZE_4_GENERAL = 32768;
    const int64_t GRID_UB_SIZE_4_GENERAL = 4096;
    const int64_t Y_UB_SIZE_4_GENERAL = 2048;
    const int64_t OUT_VAL_NUM = 4096;
    const int64_t X_UB_OFFSET = 512;
    const int64_t CAL_H_W_BLOCK = 512;
    const int64_t MASK_UB_SIZE = CAL_H_W_BLOCK / 8;

    int64_t blockIDX = 0;

    // tiling params
    int64_t coreNum_ = 0;
    int64_t inputN_ = 0;
    int64_t inputC_ = 0;
    int64_t inputH_ = 0;
    int64_t inputW_ = 0;
    int64_t outputH_ = 0;
    int64_t outputW_ = 0;
    int64_t interpolationMode_ = 0;
    int64_t paddingMode_ = 0;
    int64_t alignCorners_ = 0;
    int64_t channelLast_ = 0;
    int64_t needCoreNum_ = 0;

    int64_t gridHW_ = 0;
    int64_t lastLoopHW_ = 0;
    int64_t preNUbLoop_ = 0;
    int64_t totalUbLoop_ = 0;
    int64_t preCoreLoop_ = 0;
    int64_t lastCoreLoop_ = 0;
    int64_t channelLoop_ = 0;
    int64_t perLoopChannel_ = 0;
    int64_t lastLoopChannel_ = 0;

    // const define
    constexpr static int64_t REFLECT_RATIO = 2;
    constexpr static int64_t PADDING_MODE_ZEROS = 0;
    constexpr static int64_t PADDING_MODE_BORDER = 1;
    constexpr static int64_t PADDING_MODE_REFLECTION = 2;
    constexpr static int64_t LAYOUT_NHWC = 1;

    constexpr static uint64_t B32_VECTOR_MASK = 64;
    constexpr static uint64_t B32_BLOCK_STRIDE = 1;
    constexpr static uint64_t B32_REPEAT_STRIDE = 8;
};

template <typename T>
__aicore__ inline void GridSampler2D<T>::ParseTilingData(const GridSampleTilingData* tilingData)
{
    coreNum_ = tilingData->coreNumVar;
    inputN_ = tilingData->inN;
    inputC_ = tilingData->inC;
    inputH_ = tilingData->inH;
    inputW_ = tilingData->inW;
    outputH_ = tilingData->outH;
    outputW_ = tilingData->outW;
    interpolationMode_ = tilingData->interpolationMode;
    paddingMode_ = tilingData->paddingMode;
    alignCorners_ = tilingData->alignCorners;
    channelLast_ = tilingData->channelLast;
    needCoreNum_ = tilingData->needCoreNum;
    gridHW_ = outputH_ * outputW_;
    preNUbLoop_ = (gridHW_ + CAL_H_W_BLOCK - 1) / CAL_H_W_BLOCK;
    lastLoopHW_ = gridHW_ - CAL_H_W_BLOCK * (preNUbLoop_ - 1);
    totalUbLoop_ = preNUbLoop_ * inputN_;
    preCoreLoop_ = (totalUbLoop_ + needCoreNum_ - 1) / needCoreNum_;
    needCoreNum_ = (totalUbLoop_ + preCoreLoop_ - 1) / preCoreLoop_;
    lastCoreLoop_ = totalUbLoop_ - preCoreLoop_ * (needCoreNum_ - 1);

    channelLoop_ = (inputC_ + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;
    perLoopChannel_ = CHANNEL_BLOCK;
    lastLoopChannel_ = inputC_ - perLoopChannel_ * (channelLoop_ - 1);
}


template <typename T>
__aicore__ inline void GridSampler2D<T>::Init(GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace,
                                              const GridSampleTilingData* tilingData)
{
    blockIDX = GetBlockIdx();
    // 初始化tiling
    ParseTilingData(tilingData);

    gmX_.SetGlobalBuffer((__gm__ T *)x);
    gmGrid_.SetGlobalBuffer((__gm__ T *)gird);
    gmWorkspace_.SetGlobalBuffer((__gm__ T *)workspace);
    gmY_.SetGlobalBuffer((__gm__ T *)y);

    // buffer申请初始化
    pipe.InitBuffer(gridQueue_, 1, GRID_UB_SIZE_4_GENERAL);         // 4KB

    pipe.InitBuffer(xBuf_, X_UB_SIZE_4_GENERAL);                    // 32KB
    pipe.InitBuffer(inputXYFPBuf_, GRID_UB_SIZE_4_GENERAL);         // 4KB
    pipe.InitBuffer(inputXIntBuf_, GRID_UB_SIZE_4_GENERAL);         // 4KB
    pipe.InitBuffer(inputYIntBuf_, GRID_UB_SIZE_4_GENERAL);         // 4KB
    pipe.InitBuffer(inputXFpBuf_, GRID_UB_SIZE_4_GENERAL);          // 4KB
    pipe.InitBuffer(inputYFpBuf_, GRID_UB_SIZE_4_GENERAL);          // 4KB
    pipe.InitBuffer(weightBuf_, Y_UB_SIZE_4_GENERAL * 4);           // 8KB
    pipe.InitBuffer(weightTmpBuf_, Y_UB_SIZE_4_GENERAL * 4);        // 8KB
    pipe.InitBuffer(intTmpBuf_, Y_UB_SIZE_4_GENERAL);               // 2KB
    pipe.InitBuffer(coorBuf_, Y_UB_SIZE_4_GENERAL);                 // 2KB
    pipe.InitBuffer(coorTmpBuf_, Y_UB_SIZE_4_GENERAL);              // 2KB
    pipe.InitBuffer(outValueBuf_, X_UB_SIZE_4_GENERAL);             // 32KB
    pipe.InitBuffer(maskBuf_, 960);                                 // 960B
    pipe.InitBuffer(weightMaskBuf_, 320);                           // 320B
    pipe.InitBuffer(modBuf_, Y_UB_SIZE_4_GENERAL);                  // 2KB
    pipe.InitBuffer(extraBuf_, Y_UB_SIZE_4_GENERAL);                // 2KB
    pipe.InitBuffer(outTmpBuf_, GRID_UB_SIZE_4_GENERAL);            // 4KB
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::ComputeWeightSub(LocalTensor<float> w1Ub, LocalTensor<float> w2Ub,
                                                          LocalTensor<float> x1Ub, LocalTensor<float> x2Ub,
                                                          LocalTensor<float> y1Ub, LocalTensor<float> y2Ub)
{
    Sub(w1Ub, x1Ub, x2Ub, CAL_H_W_BLOCK);
    Sub(w2Ub, y1Ub, y2Ub, CAL_H_W_BLOCK);
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::ClipCoordinates(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
                                                         LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb,
                                                         LocalTensor<int32_t> coorUb, LocalTensor<uint8_t> wMaskUb)
{
    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> inputXIntTmpUb = coorUb;
    LocalTensor<int32_t> inputYIntTmpUb = tmpIntUb;
    pipe_barrier(PIPE_V);
    Adds(inputXIntTmpUb, iXIntUb, 0, CAL_H_W_BLOCK);
    Adds(inputYIntTmpUb, iYIntUb, 0, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    Cast(iXFpUb, inputXIntTmpUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    Cast(iYFpUb, inputYIntTmpUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 3);
    LocalTensor<uint8_t> maskXUb = wMaskUb;
    LocalTensor<uint8_t> maskYUb = maskUb;
    LocalTensor<uint8_t> maskTmpXUb = maskUb[MASK_UB_SIZE];
    LocalTensor<uint8_t> maskTmpYUb = maskUb[MASK_UB_SIZE * 2];    // 2: iY temp mask
    CoordinatesGetMaskWithRange(iXFpUb, iYFpUb, maskXUb, maskYUb, maskTmpXUb, maskTmpYUb);
    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    And(maskXUbTmp, maskYUbTmp, maskXUbTmp, maskNum);
    wMaskUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    pipe_barrier(PIPE_V);

    CoordinatesFrameRange(inputXIntTmpUb, (int32_t)(inputW_ - 1));
    CoordinatesFrameRange(inputYIntTmpUb, (int32_t)(inputH_ - 1));

    pipe_barrier(PIPE_V);

    Muls(inputYIntTmpUb, inputYIntTmpUb, (int32_t)inputW_, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Add(coorUb, coorUb, inputYIntTmpUb, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
{
    if (paddingMode_ == PADDING_MODE_BORDER) {
        BorderClip(iXFpUb, iYFpUb);
    } else if (paddingMode_ == PADDING_MODE_REFLECTION) {
        ReflectClip(iXFpUb, iYFpUb);
    }
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::CoordinatesFrameRange(LocalTensor<int32_t> iIntUb, int32_t upBound)
{
    Mins(iIntUb, iIntUb, upBound, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Maxs(iIntUb, iIntUb, 0, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::CoordinatesGetMaskWithRange(
    LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb,
    LocalTensor<uint8_t> maskTmpXUb, LocalTensor<uint8_t> maskTmpYUb)
{
    CompareScalar(maskTmpXUb, iXFpUb, 0.0f, CMPMODE::GE, CAL_H_W_BLOCK);
    CompareScalar(maskXUb, iXFpUb, static_cast<float>(inputW_ - 1), CMPMODE::LE, CAL_H_W_BLOCK);
    CompareScalar(maskTmpYUb, iYFpUb, 0.0f, CMPMODE::GE, CAL_H_W_BLOCK);
    CompareScalar(maskYUb, iYFpUb, static_cast<float>(inputH_ - 1), CMPMODE::LE, CAL_H_W_BLOCK);

    pipe_barrier(PIPE_V);

    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半
    auto maskTmpXUbTmp = maskTmpXUb.ReinterpretCast<uint16_t>();
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    auto maskTmpYUbTmp = maskTmpYUb.ReinterpretCast<uint16_t>();
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    And(maskXUbTmp, maskTmpXUbTmp, maskXUbTmp, maskNum);
    And(maskYUbTmp, maskTmpYUbTmp, maskYUbTmp, maskNum);
    pipe_barrier(PIPE_V);
    maskXUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    maskYUb = maskYUbTmp.ReinterpretCast<uint8_t>();
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::CoordinatesSelectScalar(LocalTensor<float> iFpUb,
                                                                 LocalTensor<float> oFpUb,
                                                                 LocalTensor<uint8_t> maskUb,
                                                                 const float scalarVal, const uint32_t calNum)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = 0;
    repParams.src1RepStride = 0;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = (calNum + B32_VECTOR_MASK - 1) / B32_VECTOR_MASK;
    Select(oFpUb, maskUb, iFpUb, scalarVal, SELMODE::VSEL_TENSOR_SCALAR_MODE, B32_VECTOR_MASK, repeat, repParams);
    pipe_barrier(PIPE_V);
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::CoordinatesSelectTensor(LocalTensor<float> src0,
                                                                 LocalTensor<float> src1,
                                                                 LocalTensor<float> coorUb,
                                                                 LocalTensor<uint8_t> maskUb)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = B32_BLOCK_STRIDE;
    repParams.src1RepStride = B32_REPEAT_STRIDE;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = (CAL_H_W_BLOCK + B32_VECTOR_MASK - 1) / B32_VECTOR_MASK;
    Select(coorUb, maskUb, src0, src1, SELMODE::VSEL_TENSOR_TENSOR_MODE, B32_VECTOR_MASK, repeat, repParams);
    pipe_barrier(PIPE_V);
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::BorderClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb) {

    Mins(iXFpUb, iXFpUb, (float)(inputW_ - 1), CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Maxs(iXFpUb, iXFpUb, (float)0, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    Mins(iYFpUb, iYFpUb, (float)(inputH_ - 1), CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Maxs(iYFpUb, iYFpUb, (float)0, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    LocalTensor<uint8_t> maskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<T> tmpUb = inputXYFPBuf_.Get<T>();
    Muls(tmpUb, iXFpUb, (float)(0.0), CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    CoordinatesSelectScalar(iXFpUb, iXFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Muls(tmpUb, iYFpUb, (float)(0.0), CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    CoordinatesSelectScalar(iYFpUb, iYFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::ReflectClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb) {
    LocalTensor<float> coorSubUb = coorTmpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> extraFpUb = extraBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> fmodFpUb = modBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 3);
    LocalTensor<float> tmpFpUb = outTmpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);

    // coorUb = Y * inputW_ + X
    int64_t twiceLow = (alignCorners_ == 1) ? 0 : -1;
    int64_t twiceLowY = REFLECT_RATIO * (inputH_ - 1);
    int64_t twiceLowX = REFLECT_RATIO * (inputW_ - 1);
    if (alignCorners_ == 0) {
        twiceLow = -1;
        twiceLowY = REFLECT_RATIO * inputH_ - 1;
        twiceLowX = REFLECT_RATIO * inputW_ - 1;
    }
    ReflectCoordinatesGeneral(iYFpUb, iYFpUb, extraFpUb, fmodFpUb, maskUb, tmpFpUb, tmpIntUb, twiceLow, twiceLowY);
    pipe_barrier(PIPE_V);
    ReflectCoordinatesGeneral(iXFpUb, iXFpUb, extraFpUb, fmodFpUb, maskUb, tmpFpUb, tmpIntUb, twiceLow, twiceLowX);
    pipe_barrier(PIPE_V);

    LocalTensor<T> tmpUb = inputXYFPBuf_.Get<T>();
    Muls(tmpUb, iXFpUb, (float)(0.0), CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    CoordinatesSelectScalar(iXFpUb, iXFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Muls(tmpUb, iYFpUb, (float)(0.0), CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    CoordinatesSelectScalar(iYFpUb, iYFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    Mins(iXFpUb, iXFpUb, (float)(inputW_ - 1), CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Maxs(iXFpUb, iXFpUb, (float)0, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    Mins(iYFpUb, iYFpUb, (float)(inputH_ - 1), CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Maxs(iYFpUb, iYFpUb, (float)0, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
}


template <typename T>
__aicore__ inline void GridSampler2D<T>::ReflectCoordinatesGeneral(
    LocalTensor<float> iFpUb, LocalTensor<float> coorSubUb, LocalTensor<float> extraFpUb, LocalTensor<float> fmodFpUb,
    LocalTensor<uint8_t> maskUb, LocalTensor<float> tmpFpUb, LocalTensor<int32_t> tmpIntUb, const int64_t twiceLow,
    const int64_t twiceHigh)
{
    if (twiceLow == twiceHigh) {
        Duplicate(coorSubUb, (float)0.0, CAL_H_W_BLOCK);
        return;
    }

    float minS = static_cast<float>(twiceLow) / 2;
    float negMinS = static_cast<float>(-1.0) * minS;
    float spanS = static_cast<float>(twiceHigh - twiceLow) / 2;

    // new relative position
    Adds(coorSubUb, iFpUb, negMinS, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Abs(coorSubUb, coorSubUb, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    // extra
    Muls(extraFpUb, coorSubUb, static_cast<float>(1.0f / spanS), CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Cast(tmpIntUb, extraFpUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Cast(extraFpUb, tmpIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Muls(extraFpUb, extraFpUb, spanS, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Sub(extraFpUb, coorSubUb, extraFpUb, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    // flip
    Muls(coorSubUb, coorSubUb, static_cast<float>(1.0f / spanS), CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Cast(tmpIntUb, coorSubUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Cast(coorSubUb, tmpIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    // coordinate
    /*
     S1: get two results for both possibilities, out1: extra + min, out2: muls(extra, -1.0) + span + min
     S2: get mod val, mods: flips % 2
     S3: get mask tensor, masks: CompareScalar(mods, 0.0)
     S4: select val from out1 and out2 by mask tensor, out: Select(out1, out2, mask)
    */
    LocalTensor<float> out1 = tmpFpUb;
    LocalTensor<float> out2 = extraFpUb;
    LocalTensor<float> mods = fmodFpUb;

    Adds(out1, extraFpUb, minS, CAL_H_W_BLOCK);
    Muls(out2, extraFpUb, -1.0f, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Adds(out2, out2, spanS, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Adds(out2, out2, minS, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    Muls(mods, coorSubUb, static_cast<float>(1 / 2.0), CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Cast(tmpIntUb, mods, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Cast(mods, tmpIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Muls(mods, mods, 2.0f, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Sub(mods, coorSubUb, mods, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    CompareScalar(maskUb, mods, static_cast<float>(0.0), CMPMODE::EQ, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    CoordinatesSelectTensor(out1, out2, coorSubUb, maskUb);
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::MTE2ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
                                                     int32_t loopOffset, int32_t loopElems,
                                                     LocalTensor<int32_t> coorUb, LocalTensor<float> xLocal)
{
    for (int32_t i = 0; i < loopElems; i++) {
        int64_t coordVal = coorUb.GetValue(loopOffset + i);
        int64_t baseLocation = nIdx * inputC_ * inputH_ * inputW_ + coordVal + cIdx * CHANNEL_BLOCK * inputH_ * inputW_;
        for (int cIter = 0; cIter < channelAlign; cIter++) {
            int32_t xLocalOffset = i * channelAlign + cIter;
            if (cIter >= calCElems) {
                xLocal.SetValue(xLocalOffset, 0.0f);
                continue;
            }

            int64_t coordinate = baseLocation + cIter * inputH_ * inputW_;
            xLocal.SetValue(xLocalOffset, gmX_.GetValue(coordinate));
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::MTE2ForNHWC(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
                                                     int32_t loopOffset, int32_t loopElems,
                                                     LocalTensor<int32_t> coorUb, LocalTensor<float> xLocal)
{
    int64_t base = nIdx * inputH_ * inputW_ * inputC_ + cIdx * CHANNEL_BLOCK;
    auto timeStep = loopElems / 8;
    auto timeStepRes = loopElems - loopElems / 8 * 8;

    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = calCElems * sizeof(T);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    for (int32_t i = 0; i < timeStep; i++) {
        int64_t coordVal_0 = coorUb.GetValue(loopOffset + i * 8) * inputC_;
        int64_t coordVal_1 = coorUb.GetValue(loopOffset + i * 8 + 1) * inputC_;
        int64_t coordVal_2 = coorUb.GetValue(loopOffset + i * 8 + 2) * inputC_;
        int64_t coordVal_3 = coorUb.GetValue(loopOffset + i * 8 + 3) * inputC_;
        int64_t coordVal_4 = coorUb.GetValue(loopOffset + i * 8 + 4) * inputC_;
        int64_t coordVal_5 = coorUb.GetValue(loopOffset + i * 8 + 5) * inputC_;
        int64_t coordVal_6 = coorUb.GetValue(loopOffset + i * 8 + 6) * inputC_;
        int64_t coordVal_7 = coorUb.GetValue(loopOffset + i * 8 + 7) * inputC_;
        int64_t location_0 = base + coordVal_0;
        int64_t location_1 = base + coordVal_1;
        int64_t location_2 = base + coordVal_2;
        int64_t location_3 = base + coordVal_3;
        int64_t location_4 = base + coordVal_4;
        int64_t location_5 = base + coordVal_5;
        int64_t location_6 = base + coordVal_6;
        int64_t location_7 = base + coordVal_7;

        DataCopyPad(xLocal[(i * 8) * channelAlign], gmX_[location_0], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 1) * channelAlign], gmX_[location_1], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 2) * channelAlign], gmX_[location_2], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 3) * channelAlign], gmX_[location_3], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 4) * channelAlign], gmX_[location_4], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 5) * channelAlign], gmX_[location_5], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 6) * channelAlign], gmX_[location_6], params, padParams);
        DataCopyPad(xLocal[(i * 8 + 7) * channelAlign], gmX_[location_7], params, padParams);
    }
    for(auto i = loopElems / 8 * 8; i < loopElems; i++) {
        int64_t coordVal_0 = coorUb.GetValue(loopOffset + i) * inputC_;
        int64_t location_0 = base + coordVal_0;
        DataCopyPad(xLocal[i * channelAlign], gmX_[location_0], params, padParams);
    }
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::OutTranspose(int32_t channelAlign, LocalTensor<float> xLocal, LocalTensor<float> outValueUb)
{
    LocalTensor<float> dstList[16];
    LocalTensor<float> srcList[16];

    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    if (channelAlign == 8) {
        transDataParams.repeatTimes = 8;
        transDataParams.dstRepStride = 2;
        transDataParams.srcRepStride = 16;

        for (int32_t i = 0; i < 16; i++) {
            srcList[i] = xLocal[i * 8];
        }

        for (int32_t i = 0; i < 8; i++) {
            dstList[i * 2] = outValueUb[i * TRANSE_REP_STRIDE];
            dstList[i * 2 + 1] = outValueUb[i * TRANSE_REP_STRIDE + 8];
        }

        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        TransDataTo5HD<float>(dstList, srcList, transDataParams);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    } else if (channelAlign <= 64) {
        transDataParams.repeatTimes = channelAlign / 8;
        transDataParams.dstRepStride = TRANSE_REP_STRIDE;
        transDataParams.srcRepStride = 1;
        for (int32_t j = 0; j < 8; j++) {
            for (int32_t i = 0; i < 16; i++) {
                srcList[i] = xLocal[i * channelAlign + j * 16 * channelAlign];
            }

            for (int32_t i = 0; i < 8; i++) {
                dstList[i * 2] = outValueUb[i * TRANSE_REP_STRIDE + j * 16];
                dstList[i * 2 + 1] = outValueUb[i * TRANSE_REP_STRIDE + 8 + j * 16];
            }

            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            TransDataTo5HD<float>(dstList, srcList, transDataParams);
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::MTE3ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
                                                     int32_t hwIdx, int32_t loopOffset, int32_t loopElems,
                                                     int64_t outBaseOffset, LocalTensor<float> weightUb,
                                                     LocalTensor<float> outValueUb, bool isAutomicAdd)
{
    int64_t gmYBaseOffset = outBaseOffset + loopOffset + cIdx * CHANNEL_BLOCK * gridHW_;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    if (calCElems == 1) {
        Mul(outValueUb, outValueUb, weightUb[loopOffset], TRANSE_REP_STRIDE);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        if (isAutomicAdd){
            SetAtomicAdd<float>();
            DataCopyPad(gmY_[gmYBaseOffset], outValueUb, {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            SetAtomicNone();
        } else {
            DataCopyPad(gmY_[gmYBaseOffset], outValueUb, {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
        }
    } else {
        for (int32_t i = 0; i < TRANSE_MUL_WEGHT_LOOPS; i++) {
            int32_t outOffset = i * B32_MASK;
            int32_t weightOffset = loopOffset + i * B32_MASK;
            Mul(outValueUb[outOffset], outValueUb[outOffset], weightUb[weightOffset], B32_MASK, calCElems, {1, 1, 1, 16, 16, 0});
        }

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        if (isAutomicAdd){
            SetAtomicAdd<float>();
            for (int32_t i = 0; i < calCElems; i++) {
                int64_t gmYOffset = gmYBaseOffset + i * gridHW_;
                DataCopyPad(gmY_[gmYOffset], outValueUb[i * TRANSE_REP_STRIDE], {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            }
            SetAtomicNone();
        } else {
            for (int32_t i = 0; i < calCElems; i++) {
                int64_t gmYOffset = gmYBaseOffset + i * gridHW_;
                DataCopyPad(gmY_[gmYOffset], outValueUb[i * TRANSE_REP_STRIDE], {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            }
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::PointBilinear(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
                                                       LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb,
                                                       LocalTensor<uint8_t> weightMaskUb, LocalTensor<float> outValueUb,
                                                       bool isAutomicAdd)
{
    if (paddingMode_ == PADDING_MODE_ZEROS) {
        CoordinatesSelectScalar(weightUb, weightUb, weightMaskUb, 0.0f, CAL_H_W_BLOCK);
    }

    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    auto weightMaskUbTmp = weightMaskUb.ReinterpretCast<uint64_t>();
    auto maskUbTmp = maskUb.ReinterpretCast<uint64_t>();

    int32_t trans_loop = (calHWElems + TRANSE_REP_STRIDE -1) / TRANSE_REP_STRIDE;
    int32_t loop_elems = TRANSE_REP_STRIDE;
    int32_t loop_offset = 0;
    int64_t outBaseOffset = nIdx * gridHW_ * inputC_ + hwIdx * CAL_H_W_BLOCK;
    int32_t ubOffset = 0;
    int32_t maskOffset = 0;
    pipe_barrier(PIPE_ALL);
    for (int32_t loop_idx = 0; loop_idx < trans_loop; loop_idx++) {
        if (loop_idx == trans_loop -1) {
            loop_elems = calHWElems - TRANSE_REP_STRIDE * (trans_loop - 1);
        }
        loop_offset = loop_idx * TRANSE_REP_STRIDE;
        maskOffset = loop_idx * 2;
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        maskUbTmp.SetValue(0, weightMaskUbTmp.GetValue(maskOffset));
        maskUbTmp.SetValue(1, weightMaskUbTmp.GetValue(maskOffset + 1));

        LocalTensor<float> xLocal = xBuf_.AllocTensor<float>();
        for (int32_t cIdx = 0; cIdx < channelLoop_; cIdx++) {
            int32_t calCElems = perLoopChannel_;
            if (cIdx == channelLoop_ -1)
            {
                calCElems = lastLoopChannel_;
            }
            int32_t channelAlign = (calCElems + 8 - 1) / 8 * 8;
            if (channelLast_ == LAYOUT_NHWC) {
                MTE2ForNHWC(nIdx, cIdx, calCElems, channelAlign, loop_offset, loop_elems, coordinatesUb, xLocal);
            } else {
                MTE2ForNCHW(nIdx, cIdx, calCElems, channelAlign, loop_offset, loop_elems, coordinatesUb, xLocal);
            }
            event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMte2V);
            WaitFlag<HardEvent::MTE2_V>(eventMte2V);
            OutTranspose(channelAlign, xLocal, outValueUb);
            pipe_barrier(PIPE_V);
            for (size_t i = 0; i < calCElems; i++)
            {
                ubOffset = i * TRANSE_REP_STRIDE;
                Select(outValueUb[ubOffset], maskUbTmp, outValueUb[ubOffset], 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, TRANSE_REP_STRIDE);
            }
            pipe_barrier(PIPE_V);
            MTE3ForNCHW(nIdx, cIdx, calCElems, channelAlign, hwIdx, loop_offset, loop_elems, outBaseOffset, weightUb, outValueUb, isAutomicAdd);
            event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMte3V);
            WaitFlag<HardEvent::MTE3_V>(eventMte3V);
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::PerLoopCompute(int32_t nIdx, int32_t hwIdx, int32_t calHWElems)
{
    int64_t gridGmOffset = nIdx * gridHW_ * 2 + hwIdx * CAL_H_W_BLOCK * 2;

    LocalTensor<T> gridLocal = gridQueue_.AllocTensor<T>();
    DataCopyExtParams paramsGrid;
    paramsGrid.blockCount = 1;
    paramsGrid.blockLen = calHWElems * 2 * sizeof(T);
    paramsGrid.srcStride = 0;
    paramsGrid.dstStride = 0;
    DataCopyPadExtParams<float> padParamsGrid{false, 0, 0, 0};
    DataCopyPad(gridLocal, gmGrid_[gridGmOffset], paramsGrid, padParamsGrid);
    gridQueue_.EnQue(gridLocal);
    gridQueue_.DeQue();

    LocalTensor<T> inputXYUb = inputXYFPBuf_.Get<T>();
    Adds(inputXYUb, gridLocal, (float)1.0, CAL_H_W_BLOCK * 2);

    uint32_t mask = CAL_H_W_BLOCK * 2;
    uint64_t rsvdCnt = 0;
    uint8_t xPattern = 1;
    uint8_t yPattern = 2;

    uint8_t src0RepeatStride = 8;
    uint8_t src1RepeatStride = 8;
    pipe_barrier(PIPE_V);
    LocalTensor<float> inputXFpLocal = gridLocal;
    LocalTensor<float> inputYFpLocal = gridLocal[CAL_H_W_BLOCK];
    GatherMask(inputXFpLocal, inputXYUb, xPattern, true, mask,
                {1, 1, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    GatherMask(inputYFpLocal, inputXYUb, yPattern, true, mask,
                {1, 1, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    pipe_barrier(PIPE_V);

    if (alignCorners_ == 1) {
        Muls(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * (inputW_ - (float)1.0)), CAL_H_W_BLOCK);
        Muls(inputYFpLocal, inputYFpLocal, (float)((float)0.5 * (inputH_ - (float)1.0)), CAL_H_W_BLOCK);
    } else {
        Muls(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * inputW_), CAL_H_W_BLOCK);
        Muls(inputYFpLocal, inputYFpLocal, (float)((float)0.5 * inputH_), CAL_H_W_BLOCK);
        pipe_barrier(PIPE_V);
        Adds(inputXFpLocal, inputXFpLocal, (float)(-0.5), CAL_H_W_BLOCK);
        Adds(inputYFpLocal, inputYFpLocal, (float)(-0.5), CAL_H_W_BLOCK);
    }
    pipe_barrier(PIPE_V);

    Clip(inputXFpLocal, inputYFpLocal);

    LocalTensor<int32_t> inputXWIntLocal = inputXIntBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> inputXEIntLocal = inputXIntBuf_.GetWithOffset<int32_t>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<int32_t> inputYWIntLocal = inputYIntBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> inputYEIntLocal = inputYIntBuf_.GetWithOffset<int32_t>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);

    LocalTensor<float> inputXWFpLocal = inputXFpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> inputXEFpLocal = inputXFpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> inputYWFpLocal = inputYFpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> inputYEFpLocal = inputYFpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);

    Cast(inputXWIntLocal, inputXFpLocal, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    Cast(inputYWIntLocal, inputYFpLocal, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);
    Cast(inputXWFpLocal, inputXWIntLocal, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    Cast(inputYWFpLocal, inputYWIntLocal, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    Adds(inputXEIntLocal, inputXWIntLocal, 1, CAL_H_W_BLOCK);
    Adds(inputYEIntLocal, inputYWIntLocal, 1, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    Adds(inputXEFpLocal, inputXWFpLocal, (float)1.0, CAL_H_W_BLOCK);
    Adds(inputYEFpLocal, inputYWFpLocal, (float)1.0, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    LocalTensor<float> nwWeightLocal = weightBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> neWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> swWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 2 * 4);
    LocalTensor<float> seWeightLocal = weightBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 3 * 4);

    LocalTensor<float> weightTmpLocal = weightTmpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> weightTmp1Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 4);
    LocalTensor<float> weightTmp2Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 2 * 4);
    LocalTensor<float> weightTmp3Local = weightTmpBuf_.GetWithOffset<float>(CAL_H_W_BLOCK, CAL_H_W_BLOCK * 3 * 4);
    ComputeWeightSub(nwWeightLocal, weightTmpLocal, inputXEFpLocal, inputXFpLocal, inputYEFpLocal, inputYFpLocal);
    ComputeWeightSub(neWeightLocal, weightTmp1Local, inputXFpLocal, inputXWFpLocal, inputYEFpLocal, inputYFpLocal);
    ComputeWeightSub(swWeightLocal, weightTmp2Local, inputXEFpLocal, inputXFpLocal, inputYFpLocal, inputYWFpLocal);
    ComputeWeightSub(seWeightLocal, weightTmp3Local, inputXFpLocal, inputXWFpLocal, inputYFpLocal, inputYWFpLocal);
    pipe_barrier(PIPE_V);
    Mul(nwWeightLocal, nwWeightLocal, weightTmpLocal, CAL_H_W_BLOCK);
    Mul(neWeightLocal, neWeightLocal, weightTmp1Local, CAL_H_W_BLOCK);
    Mul(swWeightLocal, swWeightLocal, weightTmp2Local, CAL_H_W_BLOCK);
    Mul(seWeightLocal, seWeightLocal, weightTmp3Local, CAL_H_W_BLOCK);
    pipe_barrier(PIPE_V);

    LocalTensor<int32_t> coordinatesLocal = coorBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<float> outValueLocal = outValueBuf_.Get<float>();
    LocalTensor<uint8_t> weightMaskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    ClipCoordinates(inputXWFpLocal, inputYWFpLocal, inputXWIntLocal, inputYWIntLocal, coordinatesLocal, weightMaskUb);
    PointBilinear(nIdx, hwIdx, calHWElems, coordinatesLocal, nwWeightLocal, weightMaskUb, outValueLocal, false);
    ClipCoordinates(inputXEFpLocal, inputYWFpLocal, inputXEIntLocal, inputYWIntLocal, coordinatesLocal, weightMaskUb);
    PointBilinear(nIdx, hwIdx, calHWElems, coordinatesLocal, neWeightLocal, weightMaskUb, outValueLocal, true);
    ClipCoordinates(inputXWFpLocal, inputYEFpLocal, inputXWIntLocal, inputYEIntLocal, coordinatesLocal, weightMaskUb);
    PointBilinear(nIdx, hwIdx, calHWElems, coordinatesLocal, swWeightLocal, weightMaskUb, outValueLocal, true);
    ClipCoordinates(inputXEFpLocal, inputYEFpLocal, inputXEIntLocal, inputYEIntLocal, coordinatesLocal, weightMaskUb);
    PointBilinear(nIdx, hwIdx, calHWElems, coordinatesLocal, seWeightLocal, weightMaskUb, outValueLocal, true);

    gridQueue_.FreeTensor(gridLocal);
}

template <typename T>
__aicore__ inline void GridSampler2D<T>::Process()
{
    if (blockIDX >= needCoreNum_) {
        return;
    }

    int32_t nIdx = 0;
    int32_t hwIdx = 0;
    int32_t preLoopNum = blockIDX * preCoreLoop_;
    int32_t calHWElems = 0;

    int64_t loopSize = preCoreLoop_;
    if (blockIDX == needCoreNum_ -1) {
        loopSize = lastCoreLoop_;
    }

    for (int32_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {
        int32_t
        nIdx = (preLoopNum + loopIdx) / preNUbLoop_;
        hwIdx = (preLoopNum + loopIdx) % preNUbLoop_;
        calHWElems = CAL_H_W_BLOCK;
        if (hwIdx == preNUbLoop_ -1) {
            calHWElems = lastLoopHW_;
        }
        PerLoopCompute(nIdx, hwIdx, calHWElems);
    }
}

}  // namespace GridSample
#endif  // GIRD_SAMPLER_2D