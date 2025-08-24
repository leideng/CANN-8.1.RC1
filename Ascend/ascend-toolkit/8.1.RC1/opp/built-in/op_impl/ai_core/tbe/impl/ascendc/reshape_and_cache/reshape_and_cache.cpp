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
 * \file reshape_and_cache.cpp
 * \brief
 */
#include "reshape_and_cache_base.h"

using namespace ReshapeAndCache;

namespace {
constexpr int32_t MAX_UB_USED = MAX_UB_SIZE / (BUFFER_NUM * 2);
}

template <typename T>
class ReshapeAndCacheNd : public ReshapeAndCacheBase {
public:
    __aicore__ inline ReshapeAndCacheNd() {}

    __aicore__ inline void ProcessKVUnequal(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn,
        GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<T>(valueInputGt_, valueIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        InitGlobalTensor<T>(valueOutputGt_, valueCacheOut);

        tokenSizeK_ = numHeads_ * headSizeK_;
        tokenSizeV_ = numHeads_ * headSizeV_;
        PrepareCopy<T>(tokenSizeK_, loopK_, tailK_, MAX_UB_USED, queBindK_);
        PrepareCopy<T>(tokenSizeV_, loopV_, tailV_, MAX_UB_USED, queBindV_);
        AllocateTask();

        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            CopyToCache<T>(i, tokenSizeK_, loopK_, tailK_, keyInputGt_,
                keyOutputGt_, slotInputGt_, MAX_UB_USED, queBindK_);
            CopyToCache<T>(i, tokenSizeV_, loopV_, tailV_, valueInputGt_,
                valueOutputGt_, slotInputGt_, MAX_UB_USED, queBindV_);
        }
    }

    __aicore__ inline void ProcessKVUnequalIncrement(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn,
        GM_ADDR valueCacheIn, GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<T>(valueInputGt_, valueIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        InitGlobalTensor<T>(valueOutputGt_, valueCacheOut);

        tokenSizeK_ = numHeads_ * headSizeK_;
        tokenSizeV_ = numHeads_ * headSizeV_;
        numBlocksK_ = tokenSizeK_ * sizeof(T) / BLOCK_SIZE;
        numBlocksV_ = tokenSizeV_ * sizeof(T) / BLOCK_SIZE;
        uint32_t tokenSize = tokenSizeK_ > tokenSizeV_ ? tokenSizeK_ : tokenSizeV_;
        InitTBuf<T>(tokenBuf_, tokenSize);
        tokenLocal_ = tokenBuf_.Get<T>();

        AllocateTask();
        AscendC::DataCopyParams copyParamsK = {1, static_cast<uint16_t>(numBlocksK_), 0, 0};
        AscendC::DataCopyParams copyParamsV = {1, static_cast<uint16_t>(numBlocksV_), 0, 0};
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            uint32_t startK = (i + startTaskId_) * tokenSizeK_;
            uint32_t startV = (i + startTaskId_) * tokenSizeV_;
            int64_t slotValue = (int64_t)(slotInputGt_.GetValue(i + startTaskId_));
            if (slotValue < 0) continue;
            uint64_t cacheStartK = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(tokenSizeK_);
            uint64_t cacheStartV = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(tokenSizeV_);
            CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, startK, cacheStartK, copyParamsK, copyParamsK);
            CopyKvCache(valueInputGt_, tokenLocal_, valueOutputGt_, startV, cacheStartV, copyParamsV, copyParamsV);
        }
    }

    __aicore__ inline void ProcessKVEqual(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn,
        GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<T>(valueInputGt_, valueIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        InitGlobalTensor<T>(valueOutputGt_, valueCacheOut);

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
            CopyKvCache(valueInputGt_, tokenLocal_, valueOutputGt_, start, cacheStart, copyParams, copyParams);
        }
    }

private:
    uint32_t tokenSizeK_ = 0;
    uint32_t tokenSizeV_ = 0;
    uint32_t numBlocksK_ = 0;
    uint32_t numBlocksV_ = 0;
    uint32_t loopK_ = 0;
    uint32_t tailK_ = 0;
    uint32_t loopV_ = 0;
    uint32_t tailV_ = 0;

    AscendC::LocalTensor<T> tokenLocal_;
    AscendC::GlobalTensor<T> keyInputGt_;
    AscendC::GlobalTensor<T> keyOutputGt_;
    AscendC::GlobalTensor<T> valueInputGt_;
    AscendC::GlobalTensor<T> valueOutputGt_;
    AscendC::GlobalTensor<int32_t> slotInputGt_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tokenBuf_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> queBindK_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> queBindV_;
};

extern "C" __global__ __aicore__ void reshape_and_cache(
    GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn,
    GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(100000000)) {
        ReshapeAndCacheNd<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVEqual(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }
    else if (TILING_KEY_IS(101000000)) {
        ReshapeAndCacheNd<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequalIncrement(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }
    else if (TILING_KEY_IS(102000000)) {
        ReshapeAndCacheNd<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequal(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }
    else if (TILING_KEY_IS(200000000)) {
        ReshapeAndCacheNd<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVEqual(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }
    else if (TILING_KEY_IS(201000000)) {
        ReshapeAndCacheNd<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequalIncrement(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }
    else if (TILING_KEY_IS(202000000)) {
        ReshapeAndCacheNd<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequal(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }
}