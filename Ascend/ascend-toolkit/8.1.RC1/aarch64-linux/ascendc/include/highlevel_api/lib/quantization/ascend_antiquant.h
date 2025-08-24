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
 * \file ascend_antiquant.h
 * \brief
 */
#ifndef LIB_QUANTIZATION_ASCEND_ANTIQUANT_H
#define LIB_QUANTIZATION_ASCEND_ANTIQUANT_H
#include "kernel_tensor.h"
#include "../../impl/quantization/antiquant/ascend_antiquant_impl.h"
namespace AscendC {
#pragma begin_pipe(V)
/* !
 * \brief compute dst = scale * (src + offset)
 * \param [out] dst, output LocalTensor
 * \param [in] src, input LocalTensor
 * \param [in] offset, input LocalTensor
 * \param [in] scale, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] K, src height/width when isTranspos is false/true
 * \param [in] shapeInfo, input offset/scale shape
 * \param [in] isTranspose, enable transpose of input
 */
template <typename InputDataType, typename OutputDataType, bool isTranspose>
__aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType> &dst, const LocalTensor<InputDataType> &src,
    const LocalTensor<OutputDataType> &offset, const LocalTensor<OutputDataType> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K, const AntiQuantShapeInfo& shapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendAntiQuantImpl<InputDataType, OutputDataType, isTranspose>(dst, src, offset, scale, sharedTmpBuffer, K,
        shapeInfo);
}

/* !
 * \ingroup AscendAntiQuant
 * \param [out] dst, output LocalTensor
 * \param [in] src, input LocalTensor
 * \param [in] scale, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] K, src height/width when isTranspos is false/true
 * \param [in] shapeInfo, input offset/scale shape
 * \param [in] isTranspose, enable transpose of input
 */
template <typename InputDataType, typename OutputDataType, bool isTranspose>
__aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType> &dst, const LocalTensor<InputDataType> &src,
    const LocalTensor<OutputDataType> &scale, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K,
    const AntiQuantShapeInfo& shapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendAntiQuantImpl<InputDataType, OutputDataType, isTranspose>(dst, src, scale, sharedTmpBuffer, K, shapeInfo);
}

/* !
 * \ingroup AscendAntiQuant
 * \param [out] dst, output LocalTensor
 * \param [in] src, input LocalTensor
 * \param [in] offset, input LocalTensor
 * \param [in] scale, input LocalTensor
 * \param [in] K, src height/width when isTranspos is false/true
 * \param [in] shapeInfo, input offset/scale shape
 * \param [in] isTranspose, enable transpose of input
 */
template <typename InputDataType, typename OutputDataType, bool isTranspose>
__aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType> &dst, const LocalTensor<InputDataType> &src,
    const LocalTensor<OutputDataType> &offset, const LocalTensor<OutputDataType> &scale, const uint32_t K,
    const AntiQuantShapeInfo& shapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendAntiQuantImpl<InputDataType, OutputDataType, isTranspose>(dst, src, offset, scale, K, shapeInfo);
}

/* !
 * \ingroup AscendAntiQuant
 * \param [out] dst, output LocalTensor
 * \param [in] src, input LocalTensor
 * \param [in] offset, input Scalar
 * \param [in] scale, input Scalar
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] K, src height/width when isTranspos is false/true
 * \param [in] shapeInfo, input offset/scale shape
 * \param [in] isTranspose, enable transpose of input
 */
template <typename InputDataType, typename OutputDataType, bool isTranspose>
__aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType> &dst, const LocalTensor<InputDataType> &src,
    const OutputDataType offset, const OutputDataType scale, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const uint32_t K, const AntiQuantShapeInfo& shapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendAntiQuantImpl<InputDataType, OutputDataType, isTranspose>(dst, src, offset, scale, sharedTmpBuffer, K,
        shapeInfo);
}

/* !
 * \ingroup AscendAntiQuant
 * \param [out] dst, output LocalTensor
 * \param [in] src, input LocalTensor
 * \param [in] scale, input Scalar
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] K, src height/width when isTranspos is false/true
 * \param [in] shapeInfo, input offset/scale shape
 * \param [in] isTranspose, enable transpose of input
 */
template <typename InputDataType, typename OutputDataType, bool isTranspose>
__aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType> &dst, const LocalTensor<InputDataType> &src,
    const OutputDataType scale, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K,
    const AntiQuantShapeInfo& shapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendAntiQuantImpl<InputDataType, OutputDataType, isTranspose>(dst, src, scale, sharedTmpBuffer, K, shapeInfo);
}

/* !
 * \ingroup AscendAntiQuant
 * \param [out] dst, output LocalTensor
 * \param [in] src, input LocalTensor
 * \param [in] offset, input Scalar
 * \param [in] scale, input Scalar
 * \param [in] K, src height/width when isTranspos is false/true
 * \param [in] shapeInfo, input offset/scale shape
 * \param [in] isTranspose, enable transpose of input
 */
template <typename InputDataType, typename OutputDataType, bool isTranspose>
__aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType> &dst, const LocalTensor<InputDataType> &src,
    const OutputDataType offset, const OutputDataType scale, const uint32_t K, const AntiQuantShapeInfo& shapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendAntiQuantImpl<InputDataType, OutputDataType, isTranspose>(dst, src, offset, scale, K, shapeInfo);
}
#pragma end_pipe
} // namespace AscendC
#endif // LIB_QUANTIZATION_ASCEND_ANTIQUANT_H