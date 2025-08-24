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
 * \file ascend_dequant.h
 * \brief
 */
#ifndef LIB_QUANTIZATION_ASCEND_DEQUANT_H
#define LIB_QUANTIZATION_ASCEND_DEQUANT_H
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220 || __CCE_AICORE__ == 200)
#include "kernel_tensor.h"
#include "../../impl/quantization/dequant/ascend_dequant_common_impl.h"

namespace AscendC {
#pragma begin_pipe(V)

/*!
 * \ingroup AscendDequant
 * \brief For DequantParams {m, n, calcount}, m means src tensor has m rows, each row has n num and the first calcount
 *        num will mul with corresponding num in deqScale.
 *        Ex: src(x, y) means yth num in the xth row in srcTensor
 *            Then dst(x, y) = src(x, y) * deqScale(y) if y is in range [0, calCount)
 * \tparam dstT: dstTensor data type
 * \tparam scaleT: deqScale tensor data type.
 * \tparam mode: deqScale calculate mode when dequantParams is in format {1, m*n, n}
 *               If mode = DEQUANT_WITH_SINGLE_ROW, then {1, m*n, n} will be converted to {m, n, n} and then process.
 *               If mode = DEQUANT_WITH_MULTI_ROW, then {1, m*n, n} will be transferred for following process.
 * \param [out] dstTensor: Output localTensor.
 * \param [in] srcTensor: Input src localTensor
 * \param [in] deqScale: Input deqScale localTensor
 * \param [in] sharedTmpBuffer： extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_dequant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] params: DequantParams with m, n, calcount to describe the calculation process like above.
 */
template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
__aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor,
    const LocalTensor<scaleT>& deqScale, const LocalTensor<uint8_t>& sharedTmpBuffer, DequantParams params)
{
    AscendDequantImpl<dstT, scaleT, true, mode>(dstTensor, srcTensor, deqScale, sharedTmpBuffer, params,
        params.m * params.n);
}

/*!
 * \ingroup AscendDequant
 * \brief For DequantParams {m, n, calcount}, m means src tensor has m rows, each row has n num and the first calcount
 *        num will mul with corresponding num in deqScale.
 *        Ex: src(x, y) means yth num in the xth row in srcTensor
 *            Then dst(x, y) = src(x, y) * deqScale(y) if y is in range [0, calCount)
 * \tparam dstT: dstTensor data type
 * \tparam scaleT: deqScale tensor data type.
 * \tparam mode: deqScale calculate mode when dequantParams is in format {1, m*n, n}
 *               If mode = DEQUANT_WITH_SINGLE_ROW, then {1, m*n, n} will be converted to {m, n, n} and then process.
 *               If mode = DEQUANT_WITH_MULTI_ROW, then {1, m*n, n} will be transferred for following process.
 * \param [out] dstTensor: Output localTensor.
 * \param [in] srcTensor: Input src localTensor
 * \param [in] deqScale: Input deqScale localTensor
 * \param [in] params: DequantParams with m, n, calcount to describe the calculation process like above.
 */
template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
__aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor,
    const LocalTensor<scaleT>& deqScale, DequantParams params)
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendDequantImpl<dstT, scaleT, mode>(dstTensor, srcTensor, deqScale, params);
}


/*!
 * \ingroup AscendDequant
 * \brief For AscendDequant function with calCount, assume that deqScale has n num, then for each n num in srcTensor,
 *        corresponding dst result = src(i) * deqScale(i) for index i (0 <= i < deqScale.GetSize())
 *        Note: must srcTensor.GetSize() % deqScale.GetSize() = 0
 * \tparam dstT: dstTensor data type
 * \tparam scaleT: deqScale tensor data type.
 * \tparam mode: deqScale calculate mode when dequantParams is in format {1, m*n, n}
 *               If mode = DEQUANT_WITH_SINGLE_ROW, then {1, m*n, n} will be converted to {m, n, n} and then process.
 *               If mode = DEQUANT_WITH_MULTI_ROW, then {1, m*n, n} will be transferred for following process.
 * \param [out] dstTensor: Output localTensor.
 * \param [in] srcTensor: Input src localTensor
 * \param [in] deqScale: Input deqScale localTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_dequant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] calCount: The number of elements in srcTensor to be processed.
 */
template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
__aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor,
    const LocalTensor<scaleT>& deqScale, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    AscendDequantCalcountImpl<dstT, scaleT, mode>(dstTensor, srcTensor, deqScale, sharedTmpBuffer, calCount);
}

/*!
 * \ingroup AscendDequant
 * \brief For AscendDequant function with calCount, assume that deqScale has n num, then for each n num in srcTensor,
 *        corresponding dst result = src(i) * deqScale(i) for index i (0 <= i < deqScale.GetSize())
 *        Note: must srcTensor.GetSize() % deqScale.GetSize() = 0
 * \tparam dstT: dstTensor data type
 * \tparam scaleT: deqScale tensor data type.
 * \tparam mode: deqScale calculate mode when dequantParams is in format {1, m*n, n}
 *               If mode = DEQUANT_WITH_SINGLE_ROW, then {1, m*n, n} will be converted to {m, n, n} and then process.
 *               If mode = DEQUANT_WITH_MULTI_ROW, then {1, m*n, n} will be transferred for following process.
 * \param [out] dstTensor: Output localTensor.
 * \param [in] srcTensor: Input src localTensor
 * \param [in] deqScale: Input deqScale localTensor
 * \param [in] calCount: The number of elements in srcTensor to be processed.
 */
template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
__aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor,
    const LocalTensor<scaleT>& deqScale, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendDequantCalcountImpl<dstT, scaleT, mode>(dstTensor, srcTensor, deqScale, calCount);
}

/*!
 * \ingroup AscendDequant
 * \brief For AscendDequant function without calCount and dequantParams, assume that deqScale has n num, then for each
 *        n num in srcTensor, corresponding dst result = src(i) * deqScale(i) for index i (0 <= i < deqScale.GetSize())
 *        It is equivalent to calcount set as srcTensor.GetSize()
 *        Note: must srcTensor.GetSize() % deqScale.GetSize() = 0
 * \tparam dstT: dstTensor data type
 * \tparam scaleT: deqScale tensor data type.
 * \tparam mode: deqScale calculate mode when dequantParams is in format {1, m*n, n}
 *               If mode = DEQUANT_WITH_SINGLE_ROW, then {1, m*n, n} will be converted to {m, n, n} and then process.
 *               If mode = DEQUANT_WITH_MULTI_ROW, then {1, m*n, n} will be transferred for following process.
 * \param [out] dstTensor: Output localTensor.
 * \param [in] srcTensor: Input src localTensor
 * \param [in] deqScale: Input deqScale localTensor
 * \param [in] sharedTmpBuffer： extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_dequant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 */
template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
__aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor,
    const LocalTensor<scaleT>& deqScale, const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    AscendDequantNoCalcountImpl<dstT, scaleT, mode>(dstTensor, srcTensor, deqScale, sharedTmpBuffer);
}

/*!
 * \ingroup AscendDequant
 * \brief For AscendDequant function without calCount and dequantParams, assume that deqScale has n num, then for each
 *        n num in srcTensor, corresponding dst result = src(i) * deqScale(i) for index i (0 <= i < deqScale.GetSize())
 *        It is equivalent to calcount set as srcTensor.GetSize()
 *        Note: must srcTensor.GetSize() % deqScale.GetSize() = 0
 * \tparam dstT: dstTensor data type
 * \tparam scaleT: deqScale tensor data type.
 * \tparam mode: deqScale calculate mode when dequantParams is in format {1, m*n, n}
 *               If mode = DEQUANT_WITH_SINGLE_ROW, then {1, m*n, n} will be converted to {m, n, n} and then process.
 *               If mode = DEQUANT_WITH_MULTI_ROW, then {1, m*n, n} will be transferred for following process.
 * \param [out] dstTensor: Output localTensor.
 * \param [in] srcTensor: Input src localTensor
 * \param [in] deqScale: Input deqScale localTensor
 */
template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
__aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor,
    const LocalTensor<scaleT>& deqScale)
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendDequantNoCalcountImpl<dstT, scaleT, mode>(dstTensor, srcTensor, deqScale);
}

/*!
 * \ingroup AscendDequant
 * \brief For DequantParams {m, n, calcount}, m means src tensor has m rows, each row has n num and the first calcount
 *        num will mul with scalar deqScale.
 *        Ex: src(x, y) means yth num in the xth row in srcTensor
 *            Then dst(x, y) = src(x, y) * deqScale if y is in range [0, calCount)
 * \tparam dstT: dstTensor data type
 * \tparam scaleT: deqScale tensor data type.
 * \tparam mode: deqScale calculate mode when dequantParams is in format {1, m*n, n}
 *               If mode = DEQUANT_WITH_SINGLE_ROW, then {1, m*n, n} will be converted to {m, n, n} and then process.
 *               If mode = DEQUANT_WITH_MULTI_ROW, then {1, m*n, n} will be transferred for following process.
 * \param [out] dstTensor: Output localTensor.
 * \param [in] srcTensor: Input src localTensor.
 * \param [in] deqScale: Input deqScale scalar.
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             ascend_dequant_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] params: DequantParams with m, n, calcount to describe the calculation process like above.
 */
template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
__aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor,
    const scaleT deqScale, const LocalTensor<uint8_t>& sharedTmpBuffer, DequantParams params)
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendDequantScalarImpl<dstT, scaleT, true, mode>(dstTensor, srcTensor, deqScale, sharedTmpBuffer, params);
}

/*!
 * \ingroup AscendDequant
 * \brief For DequantParams {m, n, calcount}, m means src tensor has m rows, each row has n num and the first calcount
 *        num will mul with scalar deqScale.
 *        Ex: src(x, y) means yth num in the xth row in srcTensor
 *            Then dst(x, y) = src(x, y) * deqScale if y is in range [0, calCount)
 * \tparam dstT: dstTensor data type
 * \tparam scaleT: deqScale tensor data type.
 * \tparam mode: deqScale calculate mode when dequantParams is in format {1, m*n, n}
 *               If mode = DEQUANT_WITH_SINGLE_ROW, then {1, m*n, n} will be converted to {m, n, n} and then process.
 *               If mode = DEQUANT_WITH_MULTI_ROW, then {1, m*n, n} will be transferred for following process.
 * \param [out] dstTensor: Output localTensor.
 * \param [in] srcTensor: Input src localTensor.
 * \param [in] deqScale: Input deqScale scalar.
 * \param [in] params: DequantParams with m, n, calcount to describe the calculation process like above.
 */
template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
__aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor,
    const scaleT deqScale, DequantParams params)
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendDequantScalarImpl<dstT, scaleT, mode>(dstTensor, srcTensor, deqScale, params);
}

#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_QUANTIZATION_ASCEND_DEQUANT_H