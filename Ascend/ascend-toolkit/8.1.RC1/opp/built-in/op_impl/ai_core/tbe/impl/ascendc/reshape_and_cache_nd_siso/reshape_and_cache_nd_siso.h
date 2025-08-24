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
 * \file reshape_and_cache_nd_siso.h
 * \brief
 */
#ifndef ASCEND_RESHAPE_AND_CACHE_ND_SISO_H
#define ASCEND_RESHAPE_AND_CACHE_ND_SISO_H

#include "../reshape_and_cache/reshape_and_cache_base.h"
namespace ReshapeAndCache {
constexpr int32_t MAX_UB_USED = MAX_UB_SIZE / BUFFER_NUM;

template <typename T>
class ReshapeAndCacheNdSiso : public ReshapeAndCacheBase {
public:
    __aicore__ inline ReshapeAndCacheNdSiso() {}

    __aicore__ inline void ProcessKey(GM_ADDR keyIn, GM_ADDR keyCacheIn, GM_ADDR slotMapping, GM_ADDR keyCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);

        tokenSizeK_ = numHeads_ * headSizeK_;
        PrepareCopy<T>(tokenSizeK_, loopK_, tailK_, MAX_UB_USED, queBindK_);
        AllocateTask();

        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            CopyToCache<T>(i, tokenSizeK_, loopK_, tailK_, keyInputGt_,
                keyOutputGt_, slotInputGt_, MAX_UB_USED, queBindK_);
        }
    }

    __aicore__ inline void ProcessKeyIncrement(GM_ADDR keyIn, GM_ADDR keyCacheIn,
        GM_ADDR slotMapping, GM_ADDR keyCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);

        tokenSizeK_ = numHeads_ * headSizeK_;
        numBlocksK_ = tokenSizeK_ * sizeof(T) / BLOCK_SIZE;
        InitTBuf<T>(tokenBuf_, tokenSizeK_);
        tokenLocal_ = tokenBuf_.Get<T>();

        AllocateTask();
        AscendC::DataCopyParams copyParams = {1, static_cast<uint16_t>(numBlocksK_), 0, 0};
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            uint32_t start = (i + startTaskId_) * tokenSizeK_;
            int64_t slotValue = (int64_t)(slotInputGt_.GetValue(i + startTaskId_));
            if (slotValue < 0) continue;
            uint64_t cacheStart = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(tokenSizeK_);
            CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, start, cacheStart, copyParams, copyParams);
        }
    }

private:
    uint32_t tokenSizeK_ = 0;
    uint32_t numBlocksK_ = 0;
    uint32_t loopK_ = 0;
    uint32_t tailK_ = 0;

    AscendC::LocalTensor<T> tokenLocal_;
    AscendC::GlobalTensor<T> keyInputGt_;
    AscendC::GlobalTensor<T> keyOutputGt_;
    AscendC::GlobalTensor<int32_t> slotInputGt_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tokenBuf_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> queBindK_;
};
}
#endif