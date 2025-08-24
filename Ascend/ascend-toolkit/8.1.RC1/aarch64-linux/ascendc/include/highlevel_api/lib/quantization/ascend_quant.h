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
 * \file ascend_quant.h
 * \brief AscendQuant is an Quantization function which refers to
 *        techniques for performing computations and storing tensors at lower bitwidths than floating point precision.
 * Mathematical formulas: AscendQuant(x, scale, offset) = scale * x + offset
 * For per tensor interface:
 * quantization parameters(scale and offset) are scalars which are used in whole input(x).
 * For per channel interface:
 * input x shape: [m, n]
 * quantization parameter scale or offset shape: [n]
 * quantization parameter contains tensor which is recycled every n elements.
 */
#ifndef LIB_ASCEND_QUANT_ASCEND_QUANT_H
#define LIB_ASCEND_QUANT_ASCEND_QUANT_H
#include "lib/quantization/ascend_quant_utils.h"
#if __CCE_AICORE__ == 100 || __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220

#include "kernel_tensor.h"
#include "../../impl/quantization/quant/ascend_quant_common_impl.h"
namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per tensor interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scale: quantization parameter, scalar
 * \param [in] offset: quantization parameter, scalar
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scale, offset, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per tensor interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] scale: quantization parameter, scalar
 * \param [in] offset: quantization parameter, scalar
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const float scale, const float offset, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, scale, offset, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per tensor interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scale: quantization parameter, scalar
 * \param [in] offset: quantization parameter, scalar
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset)
{
    AscendQuant<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scale, offset, srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per tensor interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] scale: quantization parameter, scalar
 * \param [in] offset: quantization parameter, scalar
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<T>& srcTensor, const float scale, const float offset)
{
    AscendQuant<T, isReuseSource, config>(dstTensor, srcTensor, scale, offset, srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offset: quantization parameter, scalar
 * \param [in] scaleCount: amount of per channel scales to be calculated
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const T offset, const uint32_t scaleCount, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset,
        scaleCount, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offset: quantization parameter, scalar
 * \param [in] scaleCount: amount of per channel scales to be calculated
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& scaleTensor, const T offset, const uint32_t scaleCount, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, scaleTensor, offset, scaleCount, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offset: quantization parameter, scalar
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const T offset)
{
    AscendQuant<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset,
        scaleTensor.GetSize(), srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offset: quantization parameter, scalar
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const T offset)
{
    AscendQuant<T, isReuseSource, config>(dstTensor, srcTensor, scaleTensor, offset,
        scaleTensor.GetSize(), srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offsetTensor: LocalTensor with per channel offsets, shape: [n]
 * \param [in] scaleCount: amount of per channel scales to be calculated
 * \param [in] offsetCount: amount of per channel offsets to be calculated
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const LocalTensor<T>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount,
    const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor,
        scaleCount, offsetCount, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offsetTensor: LocalTensor with per channel offsets, shape: [n]
 * \param [in] scaleCount: amount of per channel scales to be calculated
 * \param [in] offsetCount: amount of per channel offsets to be calculated
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor, const uint32_t scaleCount,
    const uint32_t offsetCount, const uint32_t calCount)
{
    AscendQuantImpl<T, isReuseSource, config>(dstTensor, srcTensor, scaleTensor, offsetTensor,
        scaleCount, offsetCount, calCount);
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_quant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offsetTensor: LocalTensor with per channel offsets, shape: [n]
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const LocalTensor<T>& offsetTensor)
{
    AscendQuant<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor,
        scaleTensor.GetSize(), offsetTensor.GetSize(), srcTensor.GetSize());
}

/*!
 * \ingroup AscendQuant
 * \brief AscendQuant per channel interface
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor, shape: [m, n]
 * \param [in] srcTensor: input LocalTensor, shape: [m, n]
 * \param [in] scaleTensor: LocalTensor with per channel scales, shape: [n]
 * \param [in] offsetTensor: LocalTensor with per channel offsets, shape: [n]
 */
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor)
{
    AscendQuant<T, isReuseSource, config>(dstTensor, srcTensor, scaleTensor, offsetTensor,
        scaleTensor.GetSize(), offsetTensor.GetSize(), srcTensor.GetSize());
}

#pragma end_pipe
}  // namespace AscendC
#endif
#endif  // LIB_ASCEND_QUANT_ASCEND_QUANT_H
