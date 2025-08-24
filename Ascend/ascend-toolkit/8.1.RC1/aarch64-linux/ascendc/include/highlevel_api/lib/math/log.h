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
 * \file log.h
 * \brief
 */
#ifndef LIB_MATH_LOG_H
#define LIB_MATH_LOG_H
#include "kernel_tensor.h"
#include "../../impl/math/log/log_common_impl.h"

#if __CCE_AICORE__ >= 200

namespace AscendC {

#pragma begin_pipe(V)
/*
 * @brief dst = log(src)
 * @ingroup Log
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    uint32_t calCount)
{
    // Only for AI Vector Core.
#if __CCE_AICORE__ == 220
    if ASCEND_IS_AIC {
        return;
    }
#endif
    LogImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
}

/*
 * @brief dst = log(src), calcalate
 * @ingroup Log
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
{
    Log<T, isReuseSource>(dstTensor, srcTensor, srcTensor.GetSize());
}

/*
 * @brief dst = log2(src)
 * @ingroup Log
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    // Only for AI Vector Core.
#if __CCE_AICORE__ == 220
    if ASCEND_IS_AIC {
        return;
    }
#endif
    Log2Impl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

/*
 * @brief dst = log2(src)
 * @ingroup Log
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    Log2<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, srcTensor.GetSize());
}

/*
 * @brief dst = log2(src)
 * @ingroup Log
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    uint32_t calCount)
{
    LocalTensor<uint8_t> stackTensor;
    // Only half requires tmpbuf.
    if constexpr (std::is_same<T, half>::value) {
        bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackTensor);
        ASCENDC_ASSERT((ans),
                { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    }

    Log2<T, isReuseSource>(dstTensor, srcTensor, stackTensor, calCount);
}

/*
 * @brief dst = log2(src)
 * @ingroup Log
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
{
    Log2<T, isReuseSource>(dstTensor, srcTensor, srcTensor.GetSize());
}

/*
 * @brief dst = log10(src)
 * @ingroup Log
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log10(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    uint32_t calCount)
{
    // Only for AI Vector Core.
#if __CCE_AICORE__ == 220
    if ASCEND_IS_AIC {
        return;
    }
#endif
    Log10Impl<T, isReuseSource>(dstTensor, srcTensor, calCount);
}

/*
 * @brief dst = log10(src)
 * @ingroup Log
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log10(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
{
    Log10<T, isReuseSource>(dstTensor, srcTensor, srcTensor.GetSize());
}

#pragma end_pipe
}  // namespace AscendC

#endif
#endif  // LIB_MATH_LOG_H
