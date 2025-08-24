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
 * \file kernel_operator_vec_brcb_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BRCB_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BRCB_IMPL_H
#include "kernel_struct_brcb.h"

namespace AscendC {
/* **************************************************************************************************
 * Brcb                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void BrcbImpl(__ubuf__ T* dst, __ubuf__ T* src0, const uint8_t repeatTimes,
    const BrcbRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        ResetMask();
        if constexpr(sizeof(T) == B16_BYTE_SIZE) {
            vbrcb((__ubuf__ uint16_t*)dst, (__ubuf__ uint16_t*)src0, repeatParams.dstBlkStride,
                repeatParams.dstRepStride, repeatTimes);
        } else if constexpr(sizeof(T) == B32_BYTE_SIZE) {
            vbrcb((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src0, repeatParams.dstBlkStride,
                repeatParams.dstRepStride, repeatTimes);
        } else {
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Brcb, current api support dtype "
                "combination is src and dst both: half / bfloat16_t / int16_t / uint16_t / float / int32_t / "
                "uint32_t.");});
        }
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BRCB_IMPL_H