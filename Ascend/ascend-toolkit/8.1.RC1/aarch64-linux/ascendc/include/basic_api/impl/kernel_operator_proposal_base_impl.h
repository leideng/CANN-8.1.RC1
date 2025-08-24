/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file kernel_operator_proposal_base_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_PROPOSAL_BASE_H
#define ASCENDC_MODULE_OPERATOR_PROPOSAL_BASE_H

namespace AscendC {
constexpr uint32_t SORT_LEN = 4;
constexpr uint32_t SORT_NUM_TWO = 2;
constexpr uint32_t SORT_NUM_THREE = 3;

__aicore__ inline void ComSortInnerLoopTail(uint32_t& offset0Tail, uint32_t& offset1Tail, uint32_t& offset2Tail,
    uint32_t& offset3Tail, uint16_t& validBitTail, uint16_t (&elementCountListTail)[SORT_LEN],
    const uint32_t baseOffset, const uint32_t elementCountTail, int32_t mergeTmpTailQueNum)
{
    if (mergeTmpTailQueNum == SORT_NUM_TWO) {
        offset1Tail = offset0Tail + baseOffset;
        elementCountListTail[1] = elementCountTail;
        offset2Tail = 0;
        elementCountListTail[2] = 0;
        offset3Tail = 0;
        elementCountListTail[3] = 0;
        validBitTail = 0b0011;
    } else if (mergeTmpTailQueNum == SORT_NUM_THREE) {
        offset1Tail = offset0Tail + baseOffset;
        offset2Tail = offset0Tail + SORT_NUM_TWO * baseOffset;
        elementCountListTail[2] = elementCountTail;
        offset3Tail = 0;
        elementCountListTail[3] = 0;
        validBitTail = 0b0111;
    } else {
        offset1Tail = offset0Tail + baseOffset;
        offset2Tail = offset0Tail + SORT_NUM_TWO * baseOffset;
        offset3Tail = offset0Tail + SORT_NUM_THREE * baseOffset;
        elementCountListTail[3] = elementCountTail;
        validBitTail = 0b1111;
    }
}

}  // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_PROPOSAL_BASE_H