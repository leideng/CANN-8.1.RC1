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
 * \file kernel_struct_proposal.h
 * \brief
 */
#ifndef ASCENDC_MODULE_STRUCT_PROPOSAL_H
#define ASCENDC_MODULE_STRUCT_PROPOSAL_H
#include "utils/kernel_utils_constants.h"

namespace AscendC {
struct MrgSort4Info {
    __aicore__ MrgSort4Info() {}
    
    __aicore__ MrgSort4Info(const uint16_t elementLengthsIn[MRG_SORT_ELEMENT_LEN], const bool ifExhaustedSuspensionIn,
        const uint16_t validBitIn, const uint16_t repeatTimesIn)
        : ifExhaustedSuspension(ifExhaustedSuspensionIn),
          validBit(validBitIn),
          repeatTimes(repeatTimesIn)
    {
        for (int32_t i = 0; i < MRG_SORT_ELEMENT_LEN; ++i) {
            elementLengths[i] = elementLengthsIn[i];
        }
    }

    uint16_t elementLengths[MRG_SORT_ELEMENT_LEN] = { 0 };
    bool ifExhaustedSuspension = false;
    uint16_t validBit = 0;
    uint8_t repeatTimes = 1;
};

template <typename T> struct MrgSortSrcList {
    __aicore__ MrgSortSrcList() {}

    __aicore__ MrgSortSrcList(const LocalTensor<T>& src1In, const LocalTensor<T>& src2In, const LocalTensor<T>& src3In,
        const LocalTensor<T>& src4In)
    {
        src1 = src1In[0];
        src2 = src2In[0];
        src3 = src3In[0];
        src4 = src4In[0];
    }

    LocalTensor<T> src1;
    LocalTensor<T> src2;
    LocalTensor<T> src3;
    LocalTensor<T> src4;
};
} // namespace AscendC
#endif // ASCENDC_MODULE_STRUCT_PROPOSAL_H
