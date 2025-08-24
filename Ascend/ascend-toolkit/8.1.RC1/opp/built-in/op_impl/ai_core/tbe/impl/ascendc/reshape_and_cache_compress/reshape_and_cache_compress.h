/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file reshape_and_cache_compress.h
 * \brief
 */

#ifndef ASCEND_RESHAPE_AND_CACHE_COMPRESS_H
#define ASCEND_RESHAPE_AND_CACHE_COMPRESS_H

#include "../reshape_and_cache/reshape_and_cache_base.h"
namespace ReshapeAndCache {
template <typename T>
class ReshapeAndCacheCompress : public ReshapeAndCacheBase {
public:
    __aicore__ inline ReshapeAndCacheCompress() {}

    __aicore__ inline void Process(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn,
        GM_ADDR slotMapping, GM_ADDR winsIn, GM_ADDR seqLenIn, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<T>(valueInputGt_, valueIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<int32_t>(winsInputGt_, winsIn);
        InitGlobalTensor<int32_t>(seqLenInputGt_, seqLenIn);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        InitGlobalTensor<T>(valueOutputGt_, valueCacheOut);

        tokenSize_ = 1 * headSizeK_;
        numBlocks_ = static_cast<uint32_t>(tokenSize_) * sizeof(T) / BLOCK_SIZE;

        InitTBuf<int32_t>(tokenBuf_, static_cast<uint32_t>(tokenSize_));
        tokenLocal_ = tokenBuf_.Get<T>();

        InitTBuf<int32_t>(winsBuf_, numHeads_ * numBatchs_);
        winsLocal_ = winsBuf_.Get<int32_t>();

        InitTBuf<int32_t>(seqLenBuf_, numBatchs_);
        seqLenLocal_ = seqLenBuf_.Get<int32_t>();

        DataCopy(winsLocal_, winsInputGt_, RoundUp(numHeads_ * numBatchs_ * sizeof(int32_t), BLOCK_SIZE));
        AscendC::PipeBarrier<PIPE_MTE2>();
        DataCopy(seqLenLocal_, seqLenInputGt_, RoundUp(numBatchs_ * sizeof(int32_t), BLOCK_SIZE));
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int i = 1; i < numBatchs_; i++) { // 获取累加的seqlen
            seqLenLocal_.SetValue(i, seqLenLocal_.GetValue(i) + seqLenLocal_.GetValue(i-1));
        }

        AllocateTask();
        AscendC::DataCopyParams copyParamsIn = {0, 0, 0, 0};
        AscendC::DataCopyParams copyParamsOut = {0, 0, 0, 0};
        uint64_t start;
        uint64_t cacheStart;
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            auto batchId = (i + startTaskId_) / numHeads_;
            auto headId = (i + startTaskId_) % numHeads_;
            uint32_t headWin = static_cast<uint32_t>(winsLocal_.GetValue(i + startTaskId_));
            int32_t consumSeqLen = seqLenLocal_.GetValue(batchId);
            auto offsetPerLine = (numHeads_ - 1) * numBlocks_;

            // 至多需要这么多次 去分块搬运一个head
            uint32_t totalCopyLoop = ((headWin * tokenSize_ * sizeof(T)) / MAX_UB_SIZE) + 1;
            uint32_t tokensPerLoop = headWin / totalCopyLoop;
            uint32_t tailTokens = headWin - totalCopyLoop * tokensPerLoop;
            for (int j = 0; j < totalCopyLoop; j++) {
                copyParamsIn = {static_cast<uint16_t>(tokensPerLoop), static_cast<uint16_t>(numBlocks_),
                    static_cast<uint16_t>(offsetPerLine), 0};
                copyParamsOut = {static_cast<uint16_t>(tokensPerLoop), static_cast<uint16_t>(numBlocks_), 0, 0};
                start = tokenSize_ * (numHeads_ * (consumSeqLen - headWin + tokensPerLoop * j) + headId);
                cacheStart = tokenSize_ * (slotInputGt_.GetValue(i + startTaskId_) + tokensPerLoop * j);
                CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, start, cacheStart, copyParamsIn, copyParamsOut);
                CopyKvCache(valueInputGt_, tokenLocal_, valueOutputGt_, start, cacheStart, copyParamsIn, copyParamsOut);
            }
            if (tailTokens != 0) {
                copyParamsIn = {static_cast<uint16_t>(tailTokens), static_cast<uint16_t>(numBlocks_),
                    static_cast<uint16_t>(offsetPerLine), 0};
                copyParamsOut = {static_cast<uint16_t>(tailTokens), static_cast<uint16_t>(numBlocks_), 0, 0};
                start = tokenSize_ * (numHeads_ * (consumSeqLen - tailTokens) + headId);
                cacheStart = tokenSize_ * (slotInputGt_.GetValue(i + startTaskId_) + tokensPerLoop * totalCopyLoop);
                CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, start, cacheStart, copyParamsIn, copyParamsOut);
                CopyKvCache(valueInputGt_, tokenLocal_, valueOutputGt_, start, cacheStart, copyParamsIn, copyParamsOut);
            }
        }
    }

private:
    uint64_t tokenSize_ = 0;
    uint32_t numBlocks_ = 0;

    AscendC::LocalTensor<T> tokenLocal_;
    AscendC::LocalTensor<int32_t> winsLocal_; // 临时存放 wins
    AscendC::LocalTensor<int32_t> seqLenLocal_; // 临时存放 seqLen

    AscendC::GlobalTensor<T> keyInputGt_;
    AscendC::GlobalTensor<T> valueInputGt_;
    AscendC::GlobalTensor<int32_t> slotInputGt_;
    AscendC::GlobalTensor<int32_t> winsInputGt_;
    AscendC::GlobalTensor<int32_t> seqLenInputGt_;

    AscendC::GlobalTensor<T> keyOutputGt_;
    AscendC::GlobalTensor<T> valueOutputGt_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> winsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> seqLenBuf_;
};
}
#endif