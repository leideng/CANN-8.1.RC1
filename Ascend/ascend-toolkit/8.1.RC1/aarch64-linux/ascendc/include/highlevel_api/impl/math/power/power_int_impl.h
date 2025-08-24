/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file power_int_impl.h
 * \brief
 */
#ifndef IMPL_MATH_POWER_POWER_INT_IMPL_H
#define IMPL_MATH_POWER_POWER_INT_IMPL_H

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
#include "power_v220_impl.h"
#elif defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
#include "power_v200_impl.h"
#endif

namespace AscendC {
/*!
 * \ingroup PowerI
 * \brief Intrinsics of Power(Int Input).
 */
__aicore__ inline void InitFinePowerI(AscPowerIParams& param, const UnaryRepeatParams& unaryParam,
    const BinaryRepeatParams& binaryParam, const uint8_t& repeat, const uint32_t calCount)
{
    Sub<int32_t, false>(param.recordExpNode, param.oriAbsExp, param.recordExpNode, MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    CompareZeroPositive(param.mask, param.recordExpNode, unaryParam, repeat);
}

__aicore__ inline void FineProcessPowerI(const LocalTensor<int32_t>& dst, const LocalTensor<int32_t>& src0,
    AscPowerIParams& param, const UnaryRepeatParams& unaryParam,
    const BinaryRepeatParams& binaryParam, const uint8_t& repeat, const uint32_t calCount)
{
    Cast<float, int32_t, false>(param.tmpTensor1.ReinterpretCast<float>(), param.recordExpNode,
        RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    // ReduceSum is ended with a scalar operate on dst[0]
    ReduceSumCount(param.tmpScalar.ReinterpretCast<float>(), param.tmpTensor1.ReinterpretCast<float>(),
        param.tmpTensor3.ReinterpretCast<float>(), calCount);
    param.expIterateSum = param.tmpScalar.ReinterpretCast<float>().GetValue(0);
    PipeBarrier<PIPE_V>();
    if (param.expIterateSum != 0) {
        Mul<int32_t, false>(param.tmpTensor1, dst, src0, MASK_PLACEHOLDER, 1, binaryParam);
        Adds<int32_t, false>(param.tmpTensor2, param.recordExpNode, -1, MASK_PLACEHOLDER, 1, unaryParam);
        PipeBarrier<PIPE_V>();
        VselPowerTensorTensor(dst.ReinterpretCast<float>(), param.mask, dst.ReinterpretCast<float>(),
            param.tmpTensor1.ReinterpretCast<float>(), param.tmpScalar.ReinterpretCast<float>(),
            SELMODE::VSEL_TENSOR_TENSOR_MODE, 1, binaryParam, calCount);
        VselPowerTensorTensor(param.recordExpNode.ReinterpretCast<float>(), param.mask,
            param.recordExpNode.ReinterpretCast<float>(), param.tmpTensor2.ReinterpretCast<float>(),
            param.tmpScalar.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, 1, binaryParam, calCount);
        PipeBarrier<PIPE_V>();
        CompareZeroPositive(param.mask, param.recordExpNode, unaryParam, repeat);
    }
}

__aicore__ inline void BulkProcessPowerI(const LocalTensor<int32_t>& dst, AscPowerIParams& param,
    const UnaryRepeatParams& unaryParam,
    const BinaryRepeatParams& binaryParam, const uint8_t& repeat, const uint32_t calCount)
{
    Cast<float, int32_t, false>(param.tmpTensor1.ReinterpretCast<float>(), param.expUBIterate,
        RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    // ReduceSum is ended with a scalar operate on dst[0]
    ReduceSumCount(param.tmpScalar.ReinterpretCast<float>(), param.tmpTensor1.ReinterpretCast<float>(),
        param.tmpTensor3.ReinterpretCast<float>(), calCount);
    param.expIterateSum = param.tmpScalar.ReinterpretCast<float>().GetValue(0);
    PipeBarrier<PIPE_V>();
    if (param.expIterateSum != 0) {
        Mul<int32_t, false>(param.tmpTensor1, dst, dst, MASK_PLACEHOLDER, 1, binaryParam);
        int32_t scalarValue = 2;
        Muls<int32_t, false>(param.tmpTensor2, param.recordExpNode, scalarValue, MASK_PLACEHOLDER, 1, unaryParam);
        PipeBarrier<PIPE_V>();
        VselPowerTensorTensor(dst.ReinterpretCast<float>(), param.mask, dst.ReinterpretCast<float>(),
            param.tmpTensor1.ReinterpretCast<float>(), param.tmpScalar.ReinterpretCast<float>(),
            SELMODE::VSEL_TENSOR_TENSOR_MODE, 1, binaryParam, calCount);
        VselPowerTensorTensor(param.recordExpNode.ReinterpretCast<float>(), param.mask,
            param.recordExpNode.ReinterpretCast<float>(), param.tmpTensor2.ReinterpretCast<float>(),
            param.tmpScalar.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, 1, binaryParam, calCount);
        ShiftRightOneBit(param.expUBIterate, param.tmpTensor3, unaryParam, binaryParam, calCount);
        PipeBarrier<PIPE_V>();
        CompareIntZero(param.mask, param.expUBIterate, param.tmpTensor3, unaryParam, repeat);
    }
}

__aicore__ inline void InitBulkPowerI(AscPowerIParams& param, const LocalTensor<int32_t>& src0,
    const LocalTensor<int32_t>& src1, const LocalTensor<int32_t>& dst, const UnaryRepeatParams& unaryParam,
    const BinaryRepeatParams& binaryParam, const uint8_t& repeat, const uint32_t calCount)
{
    Cast<float, int32_t, false>(
        param.tmpTensor1.ReinterpretCast<float>(), src1, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    CompareScalar<float, uint8_t, false>(param.negMask, param.tmpTensor1.ReinterpretCast<float>(),
        static_cast<float>(0), CMPMODE::LT, MASK_PLACEHOLDER, repeat, unaryParam);
    PipeBarrier<PIPE_V>();
    Muls<int32_t, false>(param.tmpTensor1, src1, -1, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    VselPowerTensorTensor(param.expUBIterate.ReinterpretCast<float>(), param.negMask,
        param.tmpTensor1.ReinterpretCast<float>(), src1.ReinterpretCast<float>(),
        param.tmpScalar.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, 1, binaryParam, calCount);
    PipeBarrier<PIPE_V>();
    Muls<int32_t, false>(param.oriAbsExp, param.expUBIterate, 1, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    CompareZeroPositive(param.mask, param.oriAbsExp, unaryParam, repeat);
    Duplicate<int32_t, false>(param.recordExpNode, 0, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t, false>(param.tmpTensor2, 1, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    VselPowerTensorTensor(param.recordExpNode.ReinterpretCast<float>(), param.mask,
        param.recordExpNode.ReinterpretCast<float>(), param.tmpTensor2.ReinterpretCast<float>(),
        param.tmpScalar.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, 1, binaryParam, calCount);
    VselPowerTensorTensor(dst.ReinterpretCast<float>(), param.mask, param.tmpTensor2.ReinterpretCast<float>(),
        src0.ReinterpretCast<float>(), param.tmpScalar.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE,
        1, binaryParam, calCount);
    ShiftRightOneBit(param.expUBIterate, param.tmpTensor3, unaryParam, binaryParam, calCount);
    PipeBarrier<PIPE_V>();
    CompareZeroPositive(param.mask, param.expUBIterate, unaryParam, repeat);
}

__aicore__ inline void HandleNegativeExpPowerI(const LocalTensor<int32_t>& dst, const LocalTensor<int32_t>& src0,
    AscPowerIParams& param, const UnaryRepeatParams& unaryParam,
    const BinaryRepeatParams& binaryParam, const uint8_t& repeat, const uint32_t calCount)
{
    CompareIntZero(param.mask, dst, param.tmpTensor3, unaryParam, repeat);
    LocalTensor<float> resF32 = param.oriAbsExp.ReinterpretCast<float>();
    LocalTensor<float> oneTensor = param.expUBIterate.ReinterpretCast<float>();
    Cast<float, int32_t, false>(resF32, dst, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParam);
    Duplicate<float, false>(oneTensor, 1.0f, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    Div<float, false>(resF32, oneTensor, resF32, MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    Cast<int32_t, float, false>(param.tmpTensor1, resF32, RoundMode::CAST_TRUNC, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    VselPowerTensorTensor(dst.ReinterpretCast<float>(), param.negMask, param.tmpTensor1.ReinterpretCast<float>(),
        dst.ReinterpretCast<float>(), param.tmpScalar.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE,
        1, binaryParam, calCount);
    Duplicate<int32_t, false>(param.tmpTensor2, 0, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    VselPowerTensorTensor(dst.ReinterpretCast<float>(), param.mask, param.tmpTensor2.ReinterpretCast<float>(),
        dst.ReinterpretCast<float>(), param.tmpScalar.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE,
        1, binaryParam, calCount);
}

__aicore__ inline void CommonPowerI(const LocalTensor<int32_t>& dstTensor, const LocalTensor<int32_t>& srcTensor0,
    const LocalTensor<int32_t>& srcTensor1, AscPowerIParams& param, const uint32_t calCount)
{
    const UnaryRepeatParams unaryParam;
    const BinaryRepeatParams binaryParam;
    const uint8_t repeat = DivCeil(calCount * sizeof(int32_t), ONE_REPEAT_BYTE_SIZE);

    PipeBarrier<PIPE_V>();
    InitBulkPowerI(param, srcTensor0, srcTensor1, dstTensor, unaryParam, binaryParam, repeat, calCount);
    PipeBarrier<PIPE_V>();
    param.expIterateSum = 1;
    do {
        BulkProcessPowerI(dstTensor, param, unaryParam, binaryParam, repeat, calCount);
        PipeBarrier<PIPE_V>();
    } while (param.expIterateSum != 0);
    InitFinePowerI(param, unaryParam, binaryParam, repeat, calCount);
    PipeBarrier<PIPE_V>();
    do {
        FineProcessPowerI(dstTensor, srcTensor0, param, unaryParam, binaryParam, repeat, calCount);
        PipeBarrier<PIPE_V>();
    } while (param.expIterateSum != 0);
    HandleNegativeExpPowerI(dstTensor, srcTensor0, param, unaryParam, binaryParam, repeat, calCount);
    PipeBarrier<PIPE_V>();
}
} // namespace AscendC

#endif // IMPL_MATH_POWER_POWER_INT_IMPL_H