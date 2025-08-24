/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file linear_index.h
 * \brief
 */
#ifndef LINEAR_INDEX_H
#define LINEAR_INDEX_H
#include "kernel_operator.h"
#define IS_CAST_INT (is_same<T, int64_t>::value)
using namespace AscendC;

template <typename Tp, Tp v>
struct integral_constant {
  static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

constexpr uint32_t BUFFER_NUM = 1;
constexpr int INT32_OFFSET = 31;

template <typename T, const uint32_t MODE>
class KernelLinearIndex {
public:
    __aicore__ inline KernelLinearIndex() {}
    __aicore__ inline void Init(const LinearIndexTilingData* __restrict tiling_data, TPipe *tmpPipe,
                                GM_ADDR indices, GM_ADDR output)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        pipe = tmpPipe;
        coreId = GetBlockIdx();
        usedCoreNum = tiling_data->usedCoreNum;
        indicesCount = tiling_data->indicesCount;
        indicesAlign = tiling_data->indicesAlign;
        eachCount = tiling_data->eachCount;
        lastCount = tiling_data->lastCount;
        eachNum = tiling_data->eachNum;
        eachLoop = tiling_data->eachLoop;
        eachTail = tiling_data->eachTail;
        lastNum = tiling_data->lastNum;
        lastLoop = tiling_data->lastLoop;
        lastTail = tiling_data->lastTail;
        target = tiling_data->target;
        selfStride = static_cast<int>(tiling_data->selfStride);
        indicesStride = static_cast<int>(tiling_data->indicesStride);
    
        currentEach = coreId == (usedCoreNum - 1) ? lastNum : eachNum;
        currentLoop = coreId == (usedCoreNum - 1) ? lastLoop : eachLoop;
        currentTail = coreId == (usedCoreNum - 1) ? lastTail : eachTail;

        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(indices), indicesCount);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(output), indicesCount);

        pipe->InitBuffer(calcIndices32Buf, indicesAlign * sizeof(int));
        pipe->InitBuffer(calcIndicesBuf, indicesAlign * sizeof(int));
        
        if constexpr (IS_CAST_INT) {
            pipe->InitBuffer(inQueueIndics, BUFFER_NUM, indicesAlign * sizeof(T));
            indicesLocal = inQueueIndics.AllocTensor<T>();
        }
        indices32Local = calcIndices32Buf.Get<int>();
        indicesTemp = calcIndicesBuf.Get<int>();

        if constexpr (MODE == 1) {
            pipe->InitBuffer(arangeBuffer, indicesAlign * sizeof(float));
            pipe->InitBuffer(arangeIntBuffer, indicesAlign * sizeof(int));
            arangeLocal = arangeBuffer.Get<float>();
            arangeIntLocal = arangeIntBuffer.Get<int>();
        }
        if constexpr (MODE == 2) {
            pipe->InitBuffer(arangeBuffer, indicesAlign * sizeof(float));
            arangeLocal = arangeBuffer.Get<float>();
        }

        padParams = {false, 0, 0, 0};
        tPadParams = {false, 0, 0, static_cast<T>(0)};
    }

    __aicore__ inline void Process()
    {
        for (size_t i = 0; i < currentLoop; ++i) {
            uint64_t indicesOffset = eachCount * coreId + i * currentEach;
            int offset = static_cast<int>(indicesOffset);
            currentNum = currentEach;
            if (i == currentLoop - 1) {
                currentNum = currentTail;
            }
            indicesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
            outExtParams = {(uint16_t)1, static_cast<uint32_t>(currentNum * sizeof(int)), 0, 0, 0};
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            if constexpr (IS_CAST_INT) {
                DataCopyPadGm2UBImpl((__ubuf__ uint32_t*)indicesLocal.GetPhyAddr(),
                                     (__gm__ uint32_t*)indicesGm[indicesOffset].GetPhyAddr(),
                                     indicesExtParams, padParams); // datacopypad int64
            } else {
                DataCopyPad(indices32Local, indicesGm[indicesOffset], indicesExtParams, tPadParams);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            if constexpr (IS_CAST_INT) {
                Cast<int, T>(indices32Local, indicesLocal, RoundMode::CAST_NONE, indicesAlign);
                PipeBarrier<PIPE_V>();
            }
            // 以下代码用于负数索引转正数
            // 左移31位，负数索引结果为-1，正数索引结果为0
            ShiftRight(indicesTemp, indices32Local, INT32_OFFSET, static_cast<int>(indicesAlign));
            PipeBarrier<PIPE_V>();
            // 乘以边界值
            Muls(indicesTemp, indicesTemp, static_cast<int>(target), static_cast<int>(indicesAlign));
            PipeBarrier<PIPE_V>();
            // 负数索引加上边界值
            Sub(indices32Local, indices32Local, indicesTemp, static_cast<int>(indicesAlign));

            // 以下代码用于合轴
            if constexpr (MODE == 1) { // 二维dim = 0 场景
                // 构造一个{indicesOffset, indicesOffset+1, indicesOffset+2, ……}的tensor
                ArithProgression(arangeLocal, static_cast<float>(offset), static_cast<float>(1), indicesAlign);
                PipeBarrier<PIPE_V>();
                Cast(arangeIntLocal, arangeLocal, RoundMode::CAST_FLOOR, indicesAlign);
                // 计算除以indicesStride的余数
                Muls(arangeLocal, arangeLocal, 1 / static_cast<float>(indicesStride), indicesAlign);
                PipeBarrier<PIPE_V>();
                Cast(indicesTemp, arangeLocal, RoundMode::CAST_FLOOR, indicesAlign);
                PipeBarrier<PIPE_V>();
                Muls(indicesTemp, indicesTemp, static_cast<int>(indicesStride), indicesAlign);
                PipeBarrier<PIPE_V>();
                Sub(indicesTemp, arangeIntLocal, indicesTemp, indicesAlign);
                // indices32Local = indices32Local * selfStride + 余数
                Muls(indices32Local, indices32Local, selfStride, indicesAlign);
                PipeBarrier<PIPE_V>();
                Add(indices32Local, indices32Local, indicesTemp, indicesAlign);
            }
            if constexpr (MODE == 2) { // 二维dim = 1 场景
                // 构造一个{indicesOffset, indicesOffset+1, indicesOffset+2, ……}的tensor
                ArithProgression(arangeLocal, static_cast<float>(offset), static_cast<float>(1), indicesAlign);
                PipeBarrier<PIPE_V>();
                // 计算除以indicesStride的商
                Muls(arangeLocal, arangeLocal, 1 / static_cast<float>(indicesStride), indicesAlign);
                PipeBarrier<PIPE_V>();
                Cast(indicesTemp, arangeLocal, RoundMode::CAST_FLOOR, indicesAlign);
                PipeBarrier<PIPE_V>();
                // indices32Local = indices32Local  + selfStride * 商
                Muls(indicesTemp, indicesTemp, selfStride, indicesAlign);
                PipeBarrier<PIPE_V>();
                Add(indices32Local, indicesTemp, indices32Local, indicesAlign);
            }
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            DataCopyPad(outputGm[indicesOffset], indices32Local, outExtParams);
        }
        if constexpr (IS_CAST_INT) {
            inQueueIndics.FreeTensor(indicesLocal);
        }
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIndics;
    TBuf<QuePosition::VECCALC> calcIndices32Buf, calcIndicesBuf, arangeBuffer, arangeIntBuffer;
    GlobalTensor<T> indicesGm;
    GlobalTensor<int> outputGm;
    LocalTensor<T> indicesLocal;
    LocalTensor<int> indices32Local, indicesTemp, arangeIntLocal;
    LocalTensor<float> arangeLocal;
    DataCopyExtParams indicesExtParams, outExtParams;
    DataCopyPadExtParams<T> tPadParams;
    DataCopyPadExtParams<uint32_t> padParams;
    uint32_t coreId;
    uint64_t usedCoreNum;
    uint64_t currentNum;
    uint64_t currentEach;
    uint64_t currentLoop;
    uint64_t currentTail;
    uint64_t indicesCount;
    uint64_t indicesAlign;
    uint64_t eachCount;
    uint64_t lastCount;
    uint64_t eachNum;
    uint64_t eachLoop;
    uint64_t eachTail;
    uint64_t lastNum;
    uint64_t lastLoop;
    uint64_t lastTail;
    uint64_t target;
    int selfStride;
    int indicesStride;
    uint32_t dataAlign = 32;
};

#endif  // LINEAR_INDEX_H
