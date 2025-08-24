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
 * \file kernel_operator_vec_transpose_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_TRANSPOSE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_TRANSPOSE_IMPL_H
#include "kernel_struct_transpose.h"

namespace AscendC {
template <typename T>
__aicore__ inline void TransDataTo5HDImpl(__ubuf__ T* dstList[16], __ubuf__ T* srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    ASSERT(false && "TransDataTo5HD is not supported on current device!");
}

template <typename T>
__aicore__ inline void TransDataTo5HDImpl(uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE],
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE], const TransDataTo5HDParams& transDataTo5HDParams)
{
    ASSERT(false && "TransDataTo5HD is not supported on current device!");
}

template <typename T>
__aicore__ inline void TransDataTo5HDVldVaRegImpl(
    __ubuf__ uint64_t* dst, __ubuf__ uint64_t* src, const TransDataTo5HDParams& transDataTo5HDParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported TransDataTo5HD on current device"); });
}

template <typename T>
__aicore__ inline void Transpose4DImpl(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const TransposeParamsExt &transposeParams)
{
    ASCENDC_ASSERT(false,
                   { KERNEL_LOG(KERNEL_ERROR, "unsupported transpose between NCHW and NHWC on current device"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_TRANSPOSE_IMPL_H