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
 * \file arithprogression.h
 * \brief
 */
#ifndef LIB_ARITHPROGRESSION_H
#define LIB_ARITHPROGRESSION_H

#include "../../impl/index/arithprogression/arithprogression_common_impl.h"

namespace AscendC {
/* !
 * \brief This function realizes the arithmetic sequence function. The formula is as follows:
 * dst[i+1] = dst[i] + diffValue
 *
 * \note support data type: half, float, int16_t, int32_t
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] firstValue, first value of arithmetic sequence
 * \param [in] diffValue, diff value of arithmetic sequence
 * \param [in] count, length of the sequence
 */
template <typename T>
__aicore__ inline __in_pipe__(S) __out_pipe__(V, S) void ArithProgression(const LocalTensor<T> &dstLocal,
    const T firstValue, const T diffValue, const int32_t count)
{
    ArithProgressionImpl(dstLocal, firstValue, diffValue, count);
}
} // namespace AscendC

#endif // LIB_ARITHPROGRESSION_H
