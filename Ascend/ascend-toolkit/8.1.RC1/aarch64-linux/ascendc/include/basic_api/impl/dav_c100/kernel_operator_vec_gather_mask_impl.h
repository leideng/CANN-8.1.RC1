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
 * \file kernel_operator_vec_gather_mask_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_IMPL_H
#include "kernel_struct_gather.h"

namespace AscendC {
template <typename T, typename U, GatherMaskMode mode = defaultGatherMaskMode>
__aicore__ inline void GatherMaskCal(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ U* src1, const bool reduceMode,
    const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "GatherMask");
}

template <typename T, GatherMaskMode mode = defaultGatherMaskMode>
__aicore__ inline void GatherMaskCal(__ubuf__ T* dst, __ubuf__ T* src0, const uint8_t src1Pattern,
    const bool reduceMode, const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "GatherMask");
}

__aicore__ inline int64_t GetGatherMaskRemainCountImpl()
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "GetGatherMaskRemainCount");
    return 0;
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_IMPL_H
