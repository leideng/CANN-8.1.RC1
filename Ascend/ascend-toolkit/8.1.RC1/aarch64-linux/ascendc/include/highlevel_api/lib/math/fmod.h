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
 * \file fmod.h
 * \brief
 */

#ifndef LIB_MATH_FMOD_H
#define LIB_MATH_FMOD_H
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220

#include "kernel_tensor.h"
#include "../../impl/math/fmod/fmod_common_impl.h"
#include "kernel_pop_stack_buffer.h"

namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \ingroup Fmod
 * \brief compute Fmod elementwisely
 * \tparam T: half/float
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Tensor: input LocalTensor
 * \param [in] src1Tensor: input LocalTensor, dst = src0 % src1
 * \param [in] sharedTmpBuffer: extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at fmod_tiling.h.
 *             Generally, the more space you allocate, the better performance you will achieve, and the performance
 *             reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it is not guaranteed
 *             that the shared space will be cleared after usage, the data could be anything.
 * \param [in] calCount: the number of elements to be processed.
 * \note src/dst Tensor must be 32B aligned, and it doesn't allow src/dst/sharedTmpBuffer tensor address overlap.
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Fmod(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    FmodImpl<T>(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
}

/*!
 * \ingroup Fmod
 * \brief compute Fmod elementwisely
 * \tparam T: half/float
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Tensor: input LocalTensor
 * \param [in] src1Tensor: input LocalTensor, dst = src0 % src1
 * \param [in] calCount: the number of elements to be processed.
 * \note src/dst Tensor must be 32B aligned, and it doesn't allow src/dst/sharedTmpBuffer tensor address overlap.
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Fmod(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, const uint32_t calCount)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    FmodImpl<T>(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
}

/*!
 * \ingroup Fmod
 * \brief compute Fmod elementwisely for whole source tensor
 * \tparam T: half/float
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Tensor: input LocalTensor
 * \param [in] src1Tensor: input LocalTensor, dst = src0 % src1
 * \param [in] sharedTmpBuffer: extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at fmod_tiling.h.
 *             Generally, the more space you allocate, the better performance you will achieve, and the performance
 *             reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it is not guaranteed
 *             that the shared space will be cleared after usage, the data could be anything.
 * \note src/dst Tensor must be 32B aligned, and it doesn't allow src/dst/sharedTmpBuffer tensor address overlap.
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Fmod(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    FmodImpl<T>(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, src0Tensor.GetSize());
}

/*!
 * \ingroup Fmod
 * \brief compute Fmod elementwisely for whole source tensor
 * \tparam T: half/float
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Tensor: input LocalTensor
 * \param [in] src1Tensor: input LocalTensor, dst = src0 % src1
 * \note src/dst Tensor must be 32B aligned, and it doesn't allow src/dst/sharedTmpBuffer tensor address overlap.
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Fmod(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    FmodImpl<T>(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, src0Tensor.GetSize());
}

#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_MATH_FMOD_H
