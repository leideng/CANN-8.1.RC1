/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file inner_kernel_operator_scalar_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_SCALAR_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_SCALAR_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_scalar.h"

namespace AscendC {
template <int countValue>
__aicore__ inline int64_t ScalarGetCountOfValue(uint64_t valueIn)
{
    return ScalarGetCountOfValueImpl<countValue>(valueIn);
}

__aicore__ inline int64_t ScalarCountLeadingZero(uint64_t valueIn)
{
    return ScalarCountLeadingZeroImpl(valueIn);
}

__aicore__ inline int64_t CountBitsCntSameAsSignBit(int64_t valueIn)
{
    return CountBitsCntSameAsSignBitImpl(valueIn);
}

template <int countValue>
__aicore__ inline int64_t ScalarGetSFFValue(uint64_t valueIn)
{
    return ScalarGetSFFValueImpl<countValue>(valueIn);
}

template <typename srcT, typename dstT, RoundMode roundMode>
__aicore__ inline dstT ScalarCast(srcT valueIn)
{
    return ScalarCastImpl<srcT, dstT, roundMode>(valueIn);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_INNER_OPERATOR_SCALAR_INTERFACE_H