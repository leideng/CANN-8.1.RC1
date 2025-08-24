/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ascend_antiquant_c220_impl.h
 * \brief
 */
#ifndef IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_C220_IMPL_H
#define IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_C220_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "ascend_antiquant_common.h"

namespace AscendC {

template <typename SrcType, typename OutType>
__aicore__ inline void CheckApiDtypeValid()
{
    constexpr bool inputValid = (IsSameType<SrcType, int8_t>::value) || (IsSameType<SrcType, int4b_t>::value);
    constexpr bool outputValid = (IsSameType<OutType, half>::value) || (IsSameType<OutType, bfloat16_t>::value);
    ASCENDC_ASSERT((inputValid && outputValid), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in AscendAntiQuant, "
        "current api support dtype combination is src: int8_t / int4b_t, dst: half / bfloat16_t.");});
}

template <typename SrcType, bool withOffset = true>
__aicore__ inline void AntiQuantInnerLoop(const LocalTensor<bfloat16_t> &dst, const LocalTensor<SrcType> &src,
    const LocalTensor<bfloat16_t> &offset, const LocalTensor<bfloat16_t> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const UnaryRepeatParams &unaryParamsCastSrc,
    const UnaryRepeatParams &unaryParamsToFP32, const UnaryRepeatParams &unaryParamsFP32ToDst,
    const BinaryRepeatParams &binaryParams, const uint32_t calCount)
{
    uint32_t srcFp16Pos = calCount * sizeof(bfloat16_t);
    uint32_t offsetFp32Pos = calCount * sizeof(float);
    auto fp16TmpBuffer = sharedTmpBuffer[srcFp16Pos].ReinterpretCast<half>();
    auto offsetBuffer = sharedTmpBuffer[offsetFp32Pos].ReinterpretCast<float>();
    auto resultBuffer = sharedTmpBuffer.ReinterpretCast<float>();

    SetVectorMask<float, MaskMode::COUNTER>(0, calCount);
    Cast<half, SrcType, false>(fp16TmpBuffer, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsCastSrc);
    PipeBarrier<PIPE_V>();
    Cast<float, half, false>(resultBuffer, fp16TmpBuffer, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsToFP32);
    PipeBarrier<PIPE_V>();
    if constexpr (withOffset) {
        Cast<float, bfloat16_t, false>(offsetBuffer, offset, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            unaryParamsToFP32);
        PipeBarrier<PIPE_V>();
        Add<float, false>(resultBuffer, resultBuffer, offsetBuffer, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    }
    Cast<float, bfloat16_t, false>(offsetBuffer, scale, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsToFP32);
    PipeBarrier<PIPE_V>();
    Mul<float, false>(resultBuffer, resultBuffer, offsetBuffer, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    Cast<bfloat16_t, float, false>(dst, resultBuffer, RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1, unaryParamsFP32ToDst);
    PipeBarrier<PIPE_V>();
}

template <typename SrcType, bool withOffset = true>
__aicore__ inline void AntiQuantInnerLoop(const LocalTensor<bfloat16_t> &dst, const LocalTensor<SrcType> &src,
    const bfloat16_t offset, const bfloat16_t scale, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const UnaryRepeatParams &unaryParamsCastSrc, const UnaryRepeatParams &unaryParamsToFP32,
    const UnaryRepeatParams &unaryParamsFP32ToDst, const UnaryRepeatParams &unaryParamsScalar, const uint32_t calCount)
{
    uint32_t srcFp16Pos = calCount * sizeof(bfloat16_t);
    auto fp16TmpBuffer = sharedTmpBuffer[srcFp16Pos].ReinterpretCast<half>();
    auto resultBuffer = sharedTmpBuffer.ReinterpretCast<float>();

    SetVectorMask<float, MaskMode::COUNTER>(0, calCount);
    Cast<half, SrcType, false>(fp16TmpBuffer, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsCastSrc);
    PipeBarrier<PIPE_V>();
    Cast<float, half, false>(resultBuffer, fp16TmpBuffer, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsToFP32);
    PipeBarrier<PIPE_V>();
    if constexpr (withOffset) {
        Adds<float, false>(resultBuffer, resultBuffer, ToFloat(offset), MASK_PLACEHOLDER, 1, unaryParamsScalar);
        PipeBarrier<PIPE_V>();
    }
    Muls<float, false>(resultBuffer, resultBuffer, ToFloat(scale), MASK_PLACEHOLDER, 1, unaryParamsScalar);
    PipeBarrier<PIPE_V>();
    Cast<bfloat16_t, float, false>(dst, resultBuffer, RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1, unaryParamsFP32ToDst);
    PipeBarrier<PIPE_V>();
}

template <typename SrcType>
__aicore__ inline void AscendAntiQuantNoTransposePerformance(const LocalTensor<bfloat16_t> &dst,
    const LocalTensor<SrcType> &src, const LocalTensor<bfloat16_t> &offset, const LocalTensor<bfloat16_t> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K, const uint32_t N)
{
    uint32_t posOffsetScale = N * sizeof(float) * ANTIQUANT_TWO;
    uint32_t posCast = posOffsetScale + ANTIQUANT_SINGLE_N_SIZE_BF16 * K * sizeof(half);
    auto fp16TmpBuffer = sharedTmpBuffer[posCast].ReinterpretCast<half>();
    auto resultBuffer = sharedTmpBuffer[posOffsetScale].ReinterpretCast<float>();

    UnaryRepeatParams s42f16unaryParams;
    s42f16unaryParams.srcRepStride = N / ANTIQUANT_TWO / ONE_BLK_SIZE;
    s42f16unaryParams.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    UnaryRepeatParams s82f16unaryParams;
    s82f16unaryParams.srcRepStride = N * sizeof(int8_t) / ONE_BLK_SIZE;
    s82f16unaryParams.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    UnaryRepeatParams f162f32unaryParams;
    f162f32unaryParams.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    BinaryRepeatParams binaryParams;
    binaryParams.src1RepStride = 0;
    UnaryRepeatParams f322f16Params;
    f322f16Params.dstRepStride = N * sizeof(half) / ONE_BLK_SIZE;

    for (uint32_t i = 0; i < N / ANTIQUANT_SINGLE_N_SIZE_BF16; i++) {
        SetMaskNorm();
        SetVectorMask<half, MaskMode::NORMAL>(ANTIQUANT_SINGLE_N_SIZE_BF16);
        if constexpr (IsSameType<SrcType, int4b_t>::value) {
            // 1.cast 64K to fp16, use norm mode
            Cast<half, int4b_t, false>(fp16TmpBuffer, src[ANTIQUANT_SINGLE_N_SIZE_BF16 * i], RoundMode::CAST_NONE,
                MASK_PLACEHOLDER, K, s42f16unaryParams);
        } else {
            // 1.cast 64K to fp16, use norm mode
            Cast<half, int8_t, false>(fp16TmpBuffer, src[ANTIQUANT_SINGLE_N_SIZE_BF16 * i], RoundMode::CAST_NONE,
                MASK_PLACEHOLDER, K, s82f16unaryParams);
        }
        PipeBarrier<PIPE_V>();
        // cast 64K to fp32, use count mode
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, ANTIQUANT_SINGLE_N_SIZE_BF16 * K);
        Cast<float, half, false>(resultBuffer, fp16TmpBuffer, RoundMode::CAST_NONE,
            MASK_PLACEHOLDER, K, f162f32unaryParams);
        PipeBarrier<PIPE_V>();
        // 2.add offset
        auto offsetBuffer = sharedTmpBuffer[ANTIQUANT_SINGLE_N_SIZE_BF16 * i * sizeof(float)].ReinterpretCast<float>();
        Add<float, false>(resultBuffer, resultBuffer, offsetBuffer, MASK_PLACEHOLDER, K, binaryParams);
        PipeBarrier<PIPE_V>();
        // 3.mul scale
        auto scaleBuffer = sharedTmpBuffer[N * sizeof(float) +
            ANTIQUANT_SINGLE_N_SIZE_BF16 * i * sizeof(float)].ReinterpretCast<float>();
        Mul<float, false>(resultBuffer, resultBuffer, scaleBuffer, MASK_PLACEHOLDER, K, binaryParams);
        PipeBarrier<PIPE_V>();
        // 4.cast back to bf16
        Cast<bfloat16_t, float, false>(dst[ANTIQUANT_SINGLE_N_SIZE_BF16 * i], resultBuffer,
            RoundMode::CAST_RINT, MASK_PLACEHOLDER, K, f322f16Params);
        PipeBarrier<PIPE_V>();
    }
    SetMaskNorm();
    ResetMask();
}

template <typename SrcType>
__aicore__ inline void AscendAntiQuantNoTransposePerformanceTail(const LocalTensor<bfloat16_t> &dst,
    const LocalTensor<SrcType> &src, const LocalTensor<bfloat16_t> &offset, const LocalTensor<bfloat16_t> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K, const uint32_t N, const uint32_t mask)
{
    uint32_t index = N / ANTIQUANT_SINGLE_N_SIZE_BF16 * ANTIQUANT_SINGLE_N_SIZE_BF16;
    uint32_t posOffset = N * sizeof(float);
    uint32_t posOffsetScale = posOffset * ANTIQUANT_TWO;
    uint32_t posCast = posOffsetScale + ANTIQUANT_SINGLE_N_SIZE_BF16 * K * sizeof(half);
    auto fp16TmpBuffer = sharedTmpBuffer[posCast].ReinterpretCast<half>();
    auto resultBuffer = sharedTmpBuffer[posOffsetScale].ReinterpretCast<float>();
    auto offsetBuffer = sharedTmpBuffer[index * sizeof(float)].ReinterpretCast<float>();
    auto scaleBuffer = sharedTmpBuffer[posOffset + index * sizeof(float)].ReinterpretCast<float>();

    UnaryRepeatParams s42f16unaryParams;
    s42f16unaryParams.srcRepStride = N / ANTIQUANT_TWO / ONE_BLK_SIZE;
    UnaryRepeatParams s82f16unaryParams;
    s82f16unaryParams.srcRepStride = N * sizeof(int8_t) / ONE_BLK_SIZE;
    s82f16unaryParams.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    UnaryRepeatParams f162f32unaryParams;
    f162f32unaryParams.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    BinaryRepeatParams binaryParams;
    binaryParams.src1RepStride = 0;
    UnaryRepeatParams f322f16Params;
    f322f16Params.dstRepStride = N * sizeof(bfloat16_t) / ONE_BLK_SIZE;

    // 1.cast 64K to fp16, usr norm mode
    SetMaskNorm();
    SetVectorMask<half, MaskMode::NORMAL>(mask);
    if constexpr (IsSameType<SrcType, int4b_t>::value) {
        Cast<half, int4b_t, false>(fp16TmpBuffer, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, K, s42f16unaryParams);
    } else {
        Cast<half, int8_t, false>(fp16TmpBuffer, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, K, s82f16unaryParams);
    }
    PipeBarrier<PIPE_V>();

    // cast 64K to fp32, use count mode
    Cast<float, half, false>(resultBuffer, fp16TmpBuffer, RoundMode::CAST_NONE, MASK_PLACEHOLDER, K,
        f162f32unaryParams);
    PipeBarrier<PIPE_V>();
    // 2.add offset
    Add<float, false>(resultBuffer, resultBuffer, offsetBuffer, MASK_PLACEHOLDER, K, binaryParams);
    PipeBarrier<PIPE_V>();
    // 3.mul scale
    Mul<float, false>(resultBuffer, resultBuffer, scaleBuffer, MASK_PLACEHOLDER, K, binaryParams);
    PipeBarrier<PIPE_V>();
    // 4.cast back to bf16
    Cast<bfloat16_t, float, false>(dst, resultBuffer, RoundMode::CAST_RINT, MASK_PLACEHOLDER, K, f322f16Params);
    PipeBarrier<PIPE_V>();
    ResetMask();
}

template <typename SrcType>
__aicore__ inline void PreCast(const LocalTensor<bfloat16_t> &dst, const LocalTensor<SrcType> &src,
    const LocalTensor<bfloat16_t> &offset, const LocalTensor<bfloat16_t> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K)
{
    uint32_t posOffset = offset.GetSize() * sizeof(float);
    uint32_t repeatEle = ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t);
    uint32_t repeatTimes =
        offset.GetSize() % repeatEle == 0 ? offset.GetSize() / repeatEle : offset.GetSize() / repeatEle + 1;
    auto offsetBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    auto scaleBuffer = sharedTmpBuffer[posOffset].ReinterpretCast<float>();

    UnaryRepeatParams unaryParams;
    unaryParams.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;

    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(0, offset.GetSize());
    Cast<float, bfloat16_t, false>(offsetBuffer, offset, RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
        unaryParams);
    PipeBarrier<PIPE_V>();
    Cast<float, bfloat16_t, false>(scaleBuffer, scale, RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
        unaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename DstType>
__aicore__ inline bool AntiQuantCheckPerformanceMode(const LocalTensor<DstType> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K)
{
    if constexpr (IsSameType<DstType, bfloat16_t>::value) {
        uint32_t maxTmpBufferSize =
            scale.GetSize() * ANTIQUANT_TWO * sizeof(float) + ANTIQUANT_SINGLE_N_SIZE_BF16 * K * sizeof(float);
        return sharedTmpBuffer.GetSize() >= maxTmpBufferSize;
    }
    return true;
}

// scale * (src + offset)   src: N * K, scale: N, offset: N  NOffset: offset used for tmpTensorOffset, tmpTensorScale
// For now, calCount must equal to N * K then can use brcb
template <typename SrcType, typename DstType, bool isOffset>
__aicore__ inline void CalculationMax(const LocalTensor<SrcType> &src, const LocalTensor<DstType> &dst,
    AntiquantParams<float> &params, const uint32_t calCount, const uint32_t N, const uint32_t K, const uint32_t NOffset)
{
    // store FP16 result in second half of FP32 tmpTensor to avoid input FP16 being replaced
    uint32_t srcFp16Pos = calCount / ANTIQUANT_TWO; // therefore start from (calCount / 2)th FP32 tmpTensor
    auto fp16TmpBuffer = params.tempTensorInput[srcFp16Pos].ReinterpretCast<half>();

    UnaryRepeatParams unaryParams;
    unaryParams.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    UnaryRepeatParams f322f16Params;
    f322f16Params.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    uint32_t count = K / ANTIQUANT_SINGLE_N_SIZE; // times of for loop   K = n * 64
    // src1BlkStride = 0: need same line for add and mul
    // src1RepStride = 1: 1 block for 64 num calculation
    // dst, src0RepStride = count * 8: one repeat calculate 64 num, need to jump n * 8 block
    BinaryRepeatParams binaryParams(1, 1, 0, count * DEFAULT_REPEAT_STRIDE, count * DEFAULT_REPEAT_STRIDE, 1);

    SetVectorMask<half, MaskMode::COUNTER>(0, calCount);
    // INT8 -> FP16
    Cast<half, int8_t, false>(fp16TmpBuffer, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // FP16 -> FP32
    Cast<float, half, false>(params.tempTensorInput, fp16TmpBuffer, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    SetVectorMask<float, MaskMode::COUNTER>(0, ANTIQUANT_SINGLE_N_SIZE * N); // brcb  src1 has N line, 1 line has 64 num
    for (uint32_t i = 0; i < count; i++) {
        // scale * (src + offset)
        uint32_t curOffset = i * ANTIQUANT_SINGLE_N_SIZE;
        // calculate the first group (0 ~ 64) in first loop, second group (64 ~ 128) in second loop
        if constexpr (isOffset) {
            Add<float, false>(params.tempTensorInput[curOffset], params.tempTensorInput[curOffset],
                params.tempTensorOffset[NOffset], MASK_PLACEHOLDER, N, binaryParams);
            PipeBarrier<PIPE_V>();
        }
        Mul<float, false>(params.tempTensorInput[curOffset], params.tempTensorInput[curOffset],
            params.tempTensorScale[NOffset], MASK_PLACEHOLDER, N, binaryParams);
        PipeBarrier<PIPE_V>();
    }

    // FP32 -> BF16
    SetVectorMask<float, MaskMode::COUNTER>(0, calCount);
    Cast<bfloat16_t, float, false>(dst, params.tempTensorInput, RoundMode::CAST_RINT,
        MASK_PLACEHOLDER, 1, f322f16Params);
    PipeBarrier<PIPE_V>();
}

// scale and offset are shape [N]. consistent with allocate tmpBuffer process
template <typename DstType>
__aicore__ inline void GetAntiquantTensorInfo(const LocalTensor<DstType> &scale, const LocalTensor<float> &stackBuffer,
    AntiquantParams<float> &params)
{
    uint32_t N = scale.GetSize();                                      // scale and offset are shape [N]
    params.tempTensorOffset = stackBuffer[0];                          // store 8 * N * FP32    N -> brcb -> 8 * N
    params.tempTensorScale = stackBuffer[ANTIQUANT_BRCB_BASE * N];     // store 8 * N * FP32    N -> brcb -> 8 * N
    params.tempTensorInput = stackBuffer[ANTIQUANT_BRCB_BASE * ANTIQUANT_TWO * N]; // need [N * 64 * FP32, N * K * FP32]
}

// 1. BF16 scale + offset (when withoffset = true) -> cast -> FP32  2. FP32 scale[N] + offset[N] -> brcb -> [8 * N]
// scaleEleNum means shape [N] for scale and offset
template <typename DstType, bool withOffset = true>
__aicore__ inline void CastAndBrcb(const LocalTensor<DstType> &offset, const LocalTensor<DstType> &scale,
    AntiquantParams<float> &params, const UnaryRepeatParams &unaryParams, const uint32_t scaleEleNum)
{
    uint32_t offsetEleNum = offset.GetSize();
    const uint32_t alignBase = (ONE_BLK_SIZE / sizeof(float));
    uint32_t scaleAlign = (scaleEleNum + alignBase - 1) / alignBase * alignBase;
    uint32_t tensorIndex = ANTIQUANT_BRCB_BASE * offsetEleNum - offsetEleNum;  // avoid being overwritten by brcb

    // BF16 scale + offset -> cast -> FP32
    SetVectorMask<half, MaskMode::COUNTER>(0, scaleEleNum);
    if constexpr (withOffset) {
        Cast<float, DstType, false>(params.tempTensorOffset[tensorIndex], offset, RoundMode::CAST_NONE,
            MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    }
    Cast<float, DstType, false>(params.tempTensorScale[tensorIndex], scale, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    constexpr uint16_t brcbDstBlkStride = 1;                                                // 1 num -> 8 num(1 block)
    constexpr uint16_t brcbDstRepStride = ANTIQUANT_BRCB_BASE;                              // 1 brcb: 8 num -> 64 num
    uint8_t repeatTimes = (scaleEleNum + ANTIQUANT_BRCB_BASE - 1 ) / ANTIQUANT_BRCB_BASE;   // 1 brcb needs 8 input num
    BrcbRepeatParams brcbParams(brcbDstBlkStride, brcbDstRepStride);

    SetMaskNorm();
    ResetMask();
    // brcb: 1 FP32 A -> 1 block contains 8 FP32 A, after 1 block, do the same to the next FP32 B
    if constexpr (withOffset) {
        Brcb(params.tempTensorOffset, params.tempTensorOffset[tensorIndex], repeatTimes, brcbParams);
        PipeBarrier<PIPE_V>();
    }
    Brcb(params.tempTensorScale, params.tempTensorScale[tensorIndex], repeatTimes, brcbParams);
    PipeBarrier<PIPE_V>();
}

// Cast src to FP32 src, shape is [N, 64]   calcount = 64 * N   K = 64 * n  scaleEleNum = [N]
template <typename SrcType>
__aicore__ inline void CastSrcProcess(const LocalTensor<SrcType> &src, const LocalTensor<float> &srcFP32,
    const UnaryRepeatParams &src2Fp16Params, const UnaryRepeatParams &fp162Fp32Params, const uint32_t calCount,
    const uint32_t scaleEleNum, const uint32_t n)
{
    // store FP16 result in second half of FP32 tmpTensor to avoid input FP16 being overwritten
    uint32_t srcFp16Pos = calCount / ANTIQUANT_TWO; // therefore start from (calCount / 2)th FP32 tmpTensor
    auto fp16TmpBuf = srcFP32[srcFp16Pos].ReinterpretCast<half>();
    SetMaskNorm();
    SetVectorMask<half, MaskMode::NORMAL>(0, FULL_MASK);     // In each row, first 64 num for calculation. Total N * 64
    Cast<half, SrcType, false>(fp16TmpBuf, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, scaleEleNum, src2Fp16Params);
    PipeBarrier<PIPE_V>();
    Cast<float, half, false>(srcFP32, fp16TmpBuf, RoundMode::CAST_NONE, MASK_PLACEHOLDER, scaleEleNum, fp162Fp32Params);
    PipeBarrier<PIPE_V>();
}

// FP32 scale * (FP32 src + FP32 offset)   [N * 8] * [N * 64 + N * 8]
template <bool withOffset>
__aicore__ inline void AddMulProcess(const LocalTensor<float> &srcFP32, const LocalTensor<float> &offsetFP32,
    const LocalTensor<float> &scaleFP32, const BinaryRepeatParams &binaryParams, const uint32_t N)
{
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, ANTIQUANT_SINGLE_N_SIZE * N);

    if constexpr (withOffset) {
        Add<float, false>(srcFP32, srcFP32, offsetFP32, MASK_PLACEHOLDER, N, binaryParams);
        PipeBarrier<PIPE_V>();
    }
    Mul<float, false>(srcFP32, srcFP32, scaleFP32, MASK_PLACEHOLDER, N, binaryParams);
    PipeBarrier<PIPE_V>();
}

// FP32 result -> BF16 / FP16 dst
template <typename dstT>
__aicore__ inline void CastDstProcess(const LocalTensor<float> &resFP32, const LocalTensor<dstT> &dst,
    const UnaryRepeatParams &unaryParamsFp322Dst, const uint32_t n, const uint32_t srcN)
{
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(0, FULL_MASK);
    constexpr RoundMode castMode = IsSameType<dstT, bfloat16_t>::value ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
    Cast<dstT, float, false>(dst, resFP32, castMode, MASK_PLACEHOLDER, srcN, unaryParamsFp322Dst);
    PipeBarrier<PIPE_V>();
}

// Only used when groupSize is 64 * n.   scaleEleNum: scale.GetSize   shapeN = N in shape[N, K]
template <typename SrcType, typename DstType, bool withOffset>
__aicore__ inline void CalcN64ByBrcb(const LocalTensor<DstType> &dst, const LocalTensor<SrcType> &src,
    const LocalTensor<DstType> &offset, const LocalTensor<DstType> &scale, const LocalTensor<float> &stackBuffer,
    const uint32_t scaleEleNum, const uint32_t K, const uint32_t shapeN)
{
    AntiquantParams<float> params;
    GetAntiquantTensorInfo<DstType>(scale, stackBuffer, params);

    uint32_t n = K / ANTIQUANT_SINGLE_N_SIZE;     // K = 64*n, means need n loop with each loop calculate 64*N
    uint32_t numPerLoop = ANTIQUANT_SINGLE_N_SIZE * scaleEleNum;   // scaleEle * 64

    UnaryRepeatParams unaryParamsSrc2Fp16;   // int4 / int8 -> FP16
    if constexpr (IsSameType<SrcType, int4b_t>::value) {
        unaryParamsSrc2Fp16.srcRepStride = n;                   // 1 row = K(num) / 64(num per block) = n
    } else { // int8_t
        unaryParamsSrc2Fp16.srcRepStride = ANTIQUANT_TWO * n;   // 1 row = K(num) / 32(num per block) = 2n
    }
    unaryParamsSrc2Fp16.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;    // stored continuous
    UnaryRepeatParams unaryParamsB16Fp32;    // FP16 / BF16 -> FP32
    unaryParamsB16Fp32.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;     // continuous FP16 / BF16 to continuous FP32

    BinaryRepeatParams binaryParams;         // FP32 calculation
    binaryParams.src1BlkStride = 0;          // same line for add and mul
    binaryParams.src1RepStride = 1;          // one line for 64 num calculation

    UnaryRepeatParams unaryParamsFp322Dst;   // FP32 -> half / bf16
    unaryParamsFp322Dst.dstRepStride = ANTIQUANT_SINGLE_N_SIZE * n / (ONE_BLK_SIZE / sizeof(DstType));  // 4 * n

    SetMaskCount();
    CastAndBrcb<DstType, withOffset>(offset, scale, params, unaryParamsB16Fp32, scaleEleNum); // Get FP32 offset + scale

    uint32_t curNKOffset = 0;
    for (uint32_t i = 0; i < n; i++) {   // each loop calculate [scaleEleNum, 64] elements
        curNKOffset = ANTIQUANT_SINGLE_N_SIZE * i;
        CastSrcProcess<SrcType>(src[curNKOffset], params.tempTensorInput, unaryParamsSrc2Fp16, unaryParamsB16Fp32,
            numPerLoop, scaleEleNum, n);
        AddMulProcess<withOffset>(params.tempTensorInput, params.tempTensorOffset, params.tempTensorScale, binaryParams,
            scaleEleNum);
        CastDstProcess<DstType>(params.tempTensorInput, dst[curNKOffset], unaryParamsFp322Dst, n, shapeN);
    }
}

template <bool withOffset = true>
__aicore__ inline void AntiQuantFp16Brcb(const LocalTensor<half> &scale, const LocalTensor<half> &offset,
    AntiquantParams<half> &params, const uint32_t scaleN)
{
    const uint8_t repeatTimes = scaleN / BRCB_BROADCAST_NUMBER;
    BrcbRepeatParams brcbParams(DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    SetMaskNorm();
    ResetMask();
    Brcb(params.tempTensorScale, scale, repeatTimes, brcbParams);
    PipeBarrier<PIPE_V>();
    if constexpr (withOffset) {
        Brcb(params.tempTensorOffset, offset, repeatTimes, brcbParams);
        PipeBarrier<PIPE_V>();
    }
}

template <typename SrcType, typename DstType>
__aicore__ inline void AscendAntiQuantBF16Transpose(const LocalTensor<DstType> &dst, const LocalTensor<SrcType> &src,
    const LocalTensor<DstType> &offset, const LocalTensor<DstType> &scale, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const uint32_t K, const AntiQuantShapeInfo& shapeInfo = {})
{
    uint32_t srcEleNum = src.GetSize();
    if (K > ANTIQUANT_MAX_K * ANTIQUANT_BRCB_BASE || (K % ANTIQUANT_SINGLE_N_SIZE != 0)) {
        return AntiQuantImplScalar(dst, src, offset, scale, sharedTmpBuffer, srcEleNum, K, shapeInfo);
    }

    auto stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    // when alloc tmpBuffer, use GetSize to make sure always N = 16 * a, thus brcb is aligned
    uint32_t scaleEleNum = scale.GetSize();   // int8 use old condition for compatibility
    uint32_t shapeN = src.GetSize() / K;      // int8 use old condition for compatibility
    uint32_t stackBufferSize = scaleEleNum * (ANTIQUANT_SINGLE_N_SIZE + ANTIQUANT_BRCB_BASE * ANTIQUANT_TWO);
    stackBuffer.SetSize(stackBufferSize);
    if constexpr (IsSameType<SrcType, int4b_t>::value) {  // int4 use correct shapeInfo
        scaleEleNum = (shapeInfo.scaleHeight == 0 ? scale.GetShapeInfo().shape[0] : shapeInfo.scaleHeight);
        shapeN = scaleEleNum;
    }
    CalcN64ByBrcb<SrcType, DstType, true>(dst, src, offset, scale, stackBuffer, scaleEleNum, K, shapeN);
}

template <typename SrcType, typename DstType>
__aicore__ inline void AscendAntiQuantBF16Transpose(const LocalTensor<DstType> &dst, const LocalTensor<SrcType> &src,
    const LocalTensor<DstType> &scale, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K,
    const AntiQuantShapeInfo& shapeInfo = {})
{
    uint32_t srcEleNum = src.GetSize();
    if (K > ANTIQUANT_MAX_K * ANTIQUANT_BRCB_BASE || (K % ANTIQUANT_SINGLE_N_SIZE != 0)) {
        return AntiQuantImplScalar(dst, src, scale, sharedTmpBuffer, srcEleNum, K, shapeInfo);
    }

    // when alloc tmpBuffer, use GetSize to make sure always N = 16 * a, thus brcb is aligned
    uint32_t scaleEleNum = scale.GetSize();   // int8 use old condition for compatibility
    uint32_t shapeN = src.GetSize() / K;      // int8 use old condition for compatibility
    auto stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    uint32_t stackBufferSize = scaleEleNum * (ANTIQUANT_SINGLE_N_SIZE + ANTIQUANT_BRCB_BASE * ANTIQUANT_TWO);
    stackBuffer.SetSize(stackBufferSize);
    if constexpr (IsSameType<SrcType, int4b_t>::value) {  // int4 use correct shapeInfo
        scaleEleNum = (shapeInfo.scaleHeight == 0 ? scale.GetShapeInfo().shape[0] : shapeInfo.scaleHeight);
        shapeN = scaleEleNum;
    }
    CalcN64ByBrcb<SrcType, DstType, false>(dst, src, scale, scale, stackBuffer, scaleEleNum, K, shapeN);
}

} // namespace AscendC
#endif // IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_C220_IMPL_H