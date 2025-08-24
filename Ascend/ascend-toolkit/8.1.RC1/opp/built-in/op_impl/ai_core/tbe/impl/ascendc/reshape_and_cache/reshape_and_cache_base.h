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
 * \file reshape_and_cache_base.h
 * \brief
 */
#ifndef ASCEND_RESHAPE_AND_CACHE_BASE_H
#define ASCEND_RESHAPE_AND_CACHE_BASE_H

#include "kernel_operator.h"

namespace ReshapeAndCache {
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t SCALAR_INPUT_PARAMETERS = 4;  // 算子输入input3~7是四个int32类型的参数
constexpr uint32_t COPY_NUM = 512; // 用于设置计算rope压缩均值前的搬运大小
constexpr uint32_t MAX_UB_SIZE = 192 * 1024; // UB size
constexpr uint32_t TASK_MULTIPLE = 2;       // Compress_rope模式下KV分核，分核任务量翻倍

class ReshapeAndCacheBase {
public:
    __aicore__ inline ReshapeAndCacheBase() {}

    __aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y = 16)
    {
        return y == 0 ? 0 : (x + y - 1) / y * y;
    }

    __aicore__ inline void Init(AscendC::TPipe *pipe, ReshapeAndCacheTilingData *tilingData)
    {
        pipe_ = pipe;
        // init tiling
        tilingData_ = tilingData;

        numTokens_ = tilingData_->numTokens;
        numHeads_ = tilingData_->numHeads;
        headSizeK_ = tilingData_->headSizeK;
        headSizeV_ = tilingData_->headSizeV;
        numBatchs_ = tilingData_->numBatchs;
    }

    template <typename T>
    __aicore__ inline void InitGlobalTensor(AscendC::GlobalTensor<T> &gm, GM_ADDR addr)
    {
        gm.SetGlobalBuffer((__gm__ T *)addr);
    }

    template <typename T, typename B>
    __aicore__ inline void InitTBuf(B &buf, uint32_t len)
    {
        pipe_->InitBuffer(buf, RoundUp(len * sizeof(T), BLOCK_SIZE));
    }

    template <typename T, typename B>
    __aicore__ inline void InitScalarTBuf(B &buf, uint32_t len)
    {
        pipe_->InitBuffer(buf, SCALAR_INPUT_PARAMETERS * RoundUp(len * sizeof(T), BLOCK_SIZE));
    }

    __aicore__ inline void AllocateTask()
    {
        coreNums_ = static_cast<uint32_t>(AscendC::GetBlockNum());
        perCoreTaskNum_ = numTokens_ / coreNums_;
        tailTaskNum_ = numTokens_ % coreNums_;
        blockId_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
        startTaskId_ = blockId_ * perCoreTaskNum_;

        if (blockId_ < tailTaskNum_) {
            perCoreTaskNum_++;
            startTaskId_ += blockId_;
        } else {
            startTaskId_ += tailTaskNum_;
        }
    }

    __aicore__ inline void AllocateTaskRope()
    {
        coreNums_ = static_cast<uint32_t>(AscendC::GetBlockNum());
        perCoreTaskNum_ = numTokens_ * TASK_MULTIPLE / coreNums_;
        tailTaskNum_ = numTokens_ * TASK_MULTIPLE % coreNums_;
        blockId_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
        startTaskId_ = blockId_ * perCoreTaskNum_;

        if (blockId_ < tailTaskNum_) {
            perCoreTaskNum_++;
            startTaskId_ += blockId_;
        } else {
            startTaskId_ += tailTaskNum_;
        }
    }

    template <typename T>
    __aicore__ inline void CopyKvCache(
        AscendC::GlobalTensor<T>& src,
        AscendC::LocalTensor<T>& ubAddr,
        AscendC::GlobalTensor<T>& dst,
        uint64_t start,
        uint64_t cacheStart,
        AscendC::DataCopyParams& copyParamsIn,
        AscendC::DataCopyParams& copyParamsOut)
    {
        DataCopy(ubAddr, src[start], copyParamsIn);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
        DataCopy(dst[cacheStart], ubAddr, copyParamsOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    template <typename T>
    __aicore__ inline void CopyKvCacheDoubleBuf(
        AscendC::GlobalTensor<T>& src,
        AscendC::LocalTensor<T>& ubAddr,
        AscendC::GlobalTensor<T>& dst,
        event_t eventID,
        uint64_t start,
        uint64_t cacheStart,
        AscendC::DataCopyParams& copyParamsIn,
        AscendC::DataCopyParams& copyParamsOut)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
        DataCopy(ubAddr, src[start], copyParamsIn);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventID);
        DataCopy(dst[cacheStart], ubAddr, copyParamsOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
    }

    template <typename T1>
    __aicore__ inline void CopyToCache(
        uint32_t index, uint32_t tokenSize,
        uint32_t loop, uint32_t tail,
        AscendC::GlobalTensor<T1>& src,
        AscendC::GlobalTensor<T1>& dst,
        AscendC::GlobalTensor<int32_t>& slotInput, const int32_t &maxUbUsed,
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM>& queBind)
    {
        int64_t slotValue = (int64_t)(slotInput.GetValue(index + startTaskId_));
        if (slotValue < 0) {
            return;
        }
        uint64_t start = static_cast<uint64_t>(tokenSize) * (index + startTaskId_);
        uint64_t cacheStart = static_cast<uint64_t>(slotValue) * tokenSize;

        AscendC::DataCopyParams copyParams = {1, static_cast<uint16_t>(maxUbUsed / BLOCK_SIZE), 0, 0};
        for (uint32_t j = 0; j < loop; j++) {
            auto bindLocal = queBind.AllocTensor<T1>();
            DataCopy(bindLocal, src[start], copyParams);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<T1>();
            DataCopy(dst[cacheStart], bindLocal, copyParams);
            queBind.FreeTensor(bindLocal);
            start += (maxUbUsed / sizeof(T1));
            cacheStart += static_cast<uint64_t>(maxUbUsed / sizeof(T1));
        }
        if (tail > 0) {
            AscendC::DataCopyExtParams copyParam = {1, tail, 0, 0, 0};
            AscendC::DataCopyPadExtParams<T1> padParams = {false, 0, 0, 0};
            auto bindLocal = queBind.AllocTensor<T1>();
            DataCopyPad(bindLocal, src[start], copyParam, padParams);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<T1>();
            DataCopyPad(dst[cacheStart], bindLocal, copyParam);
            queBind.FreeTensor(bindLocal);
        }
    }

    template <typename T>
    __aicore__ inline void PrepareCopy(
        uint32_t tokenSize, uint32_t &loop,
        uint32_t &tail, const int32_t &maxUbUsed,
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM>& queBind)
    {
        loop = (tokenSize * sizeof(T)) / maxUbUsed;
        tail = (tokenSize * sizeof(T)) % maxUbUsed;
        pipe_->InitBuffer(queBind, BUFFER_NUM, maxUbUsed);
    }

protected:
    ReshapeAndCacheTilingData *tilingData_ = nullptr;
    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t headSizeK_ = 0;
    uint32_t headSizeV_ = 0;
    uint32_t numBatchs_ = 0;

    uint32_t coreNums_ = 0;
    uint32_t perCoreTaskNum_ = 0;
    uint32_t tailTaskNum_ = 0;
    uint32_t blockId_ = 0;
    uint32_t startTaskId_ = 0;

    AscendC::TPipe* pipe_;
};
}
#endif
