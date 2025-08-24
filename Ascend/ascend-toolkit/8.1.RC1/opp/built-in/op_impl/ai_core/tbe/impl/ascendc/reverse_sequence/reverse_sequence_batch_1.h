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
 * \file reverse_sequence_batch_1.h
 * \brief
 */
#ifndef REVERSE_SEQUENCE_BATCH_1_H
#define REVERSE_SEQUENCE_BATCH_1_H

#include "reverse_sequence_base.h"

namespace ReverseSequence {
using namespace AscendC;

template <typename T, bool isCSmall = true>
class ReverseSequenceBatch1 : public ReverseSequenceBase<T> {
public:
    __aicore__ inline ReverseSequenceBatch1(const ReverseSequenceTilingData* tilingDataPtr)
        : ReverseSequenceBase<T>(tilingDataPtr){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR seq_lengths, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void ReverseSeq(const int64_t batchIndex, const int64_t seqGroupIndex,
                                      const int64_t reverseCount);
};

template <typename T, bool isCSmall>
__aicore__ inline void ReverseSequenceBatch1<T, isCSmall>::Init(GM_ADDR x, GM_ADDR seq_lengths, GM_ADDR y,
                                                                GM_ADDR workspace) {
    this->BaseInit(seq_lengths, workspace);
    int64_t curCoreGmStart = this->batchBaseIndex_ * this->cSize_ * this->xDtypeSize_;
    this->xGM_.SetGlobalBuffer(x + curCoreGmStart);
    this->yGM_.SetGlobalBuffer(y + curCoreGmStart);
}

template <typename T, bool isCSmall>
__aicore__ inline void ReverseSequenceBatch1<T, isCSmall>::ReverseSeq(const int64_t batchIndex,
                                                                      const int64_t seqGroupIndex,
                                                                      const int64_t reverseCount) {
    int64_t baseOffset = seqGroupIndex * this->seqDimValue_ * this->batchSize_ * this->cSize_;
    int64_t gmInOffset = baseOffset + batchIndex * this->cSize_;
    int64_t gmOutOffset = (reverseCount - 1) * this->batchSize_ * this->cSize_ + gmInOffset;
    for (int64_t i = 0; i < reverseCount; i++) {
        if constexpr (isCSmall) {
            this->CopyInX(gmInOffset, this->cSize_);
            this->CopyOutX(gmOutOffset, this->cSize_);
        } else {
            this->SingleReverseCBig(gmInOffset, gmOutOffset);
        }
        gmInOffset += (this->batchSize_ * this->cSize_);
        gmOutOffset -= (this->batchSize_ * this->cSize_);
    }
    for (int64_t i = reverseCount; i < this->seqDimValue_; i++) {
        if constexpr (isCSmall) {
            this->CopyInX(gmInOffset, this->cSize_);
            this->CopyOutX(gmInOffset, this->cSize_);
        } else {
            this->SingleCopyCBig(gmInOffset);
        }
        gmInOffset += (this->batchSize_ * this->cSize_);
    }
}

}  // namespace ReverseSequence

#endif  // REVERSE_SEQUENCE_BATCH_1_H