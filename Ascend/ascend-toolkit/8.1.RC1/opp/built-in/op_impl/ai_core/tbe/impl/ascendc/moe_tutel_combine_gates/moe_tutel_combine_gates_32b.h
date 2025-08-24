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
 * \file moe_tutel_combine_gates_32b.h
 * \brief
 */
#ifndef MOE_TUTEL_COMBINE_GATES_32B_
#define MOE_TUTEL_COMBINE_GATES_32B_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class MoeTutelCombineGatesFloat {
public:
    __aicore__ inline MoeTutelCombineGatesFloat() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y_grad, GM_ADDR indices, GM_ADDR locations, GM_ADDR gates_grad,
                                MoeTutelCombineGatesTilingData* tilingData)
    {
        samples = tilingData->samples;
        hidden = tilingData->hidden;
        capacity = tilingData->capacity;
        batchSize = tilingData->batchSize;
        taskNumPerBlock = tilingData->taskNumPerBlock;

        taskNum = samples * batchSize;
        curBlockIdx = GetBlockIdx();
        startOffset = curBlockIdx * taskNumPerBlock;
        endOffset = (curBlockIdx + 1) * taskNumPerBlock;
        if (endOffset > taskNum) {
            endOffset = taskNum;
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), samples * hidden);
        yGradGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y_grad), batchSize * capacity * hidden);

        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices), samples * batchSize);
        locationsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(locations), samples * batchSize);
        gatesGradGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gates_grad), samples * batchSize);

        pipe.InitBuffer(inQueueX, 1, hidden * sizeof(T));
        pipe.InitBuffer(inQueueY, 1, hidden * sizeof(T));
        pipe.InitBuffer(inQueueIndices, 1, innerOffset * sizeof(int32_t));
        pipe.InitBuffer(inQueueLocations, 1, innerOffset * sizeof(int32_t));

        pipe.InitBuffer(mulQueue, 1, hidden * sizeof(T));
        pipe.InitBuffer(sumQueue, 1, innerOffset * sizeof(T));
        pipe.InitBuffer(workQueue, 1, hidden * sizeof(T));

        pipe.InitBuffer(outQueueGates, 1, innerOffset * sizeof(T));
        pipe.InitBuffer(tmpQueue, 1, innerOffset * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        Compute();
    }

private:
    __aicore__ inline void Compute() {
        LocalTensor<int32_t> indicesLocal = inQueueIndices.AllocTensor<int32_t>();
        LocalTensor<int32_t> locationsLocal = inQueueLocations.AllocTensor<int32_t>();
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();

        LocalTensor<T> mulLocal = mulQueue.AllocTensor<T>();
        LocalTensor<T> sumLocal = sumQueue.AllocTensor<T>();

        LocalTensor<T> workLocal = workQueue.AllocTensor<T>();
        LocalTensor<T> tmpLocal = tmpQueue.AllocTensor<T>();

        LocalTensor<T> gatesLocal = outQueueGates.AllocTensor<T>();

        for (int32_t taskOffset = startOffset; taskOffset < endOffset; taskOffset += innerOffset) {
            DataCopy(indicesLocal, indicesGm[taskOffset], innerOffset);
            DataCopy(locationsLocal, locationsGm[taskOffset], innerOffset);
            innerTask = innerOffset;
            if (taskOffset + innerOffset > taskNum) {
                innerTask = taskNum - taskOffset;
            }
            moveCounts = 0;
            Duplicate<float>(sumLocal, 0.0, innerOffset);
            for (int32_t i = 0; i < innerTask; i++) {
                idx0 = indicesLocal.GetValue(i);
                idx1 = locationsLocal.GetValue(i);
                pipe_barrier(PIPE_ALL);
                if (idx1 < capacity && idx0 >= 0) {
                    DataCopy(xLocal, xGm[((taskOffset + i) % samples) * hidden], hidden);
                    offset = idx0 * capacity + idx1;
                    DataCopy(yLocal, yGradGm[offset * hidden], hidden);
                    pipe_barrier(PIPE_ALL);
                    Mul(mulLocal, xLocal, yLocal, hidden);
                    ReduceSum<T>(tmpLocal, mulLocal, workLocal, hidden);
                    pipe_barrier(PIPE_ALL);
                    sumLocal.SetValue(i, tmpLocal.GetValue(0));
                    moveCounts += 1;
                }
            }
            if (moveCounts > 0) {
                DataCopy(gatesLocal, sumLocal, innerOffset);
                DataCopy(gatesGradGm[taskOffset], gatesLocal, innerOffset);
            }
        }
        inQueueIndices.FreeTensor(indicesLocal);
        inQueueLocations.FreeTensor(locationsLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        
        mulQueue.FreeTensor(mulLocal);
        sumQueue.FreeTensor(sumLocal);

        workQueue.FreeTensor(workLocal);
        tmpQueue.FreeTensor(tmpLocal);

        outQueueGates.FreeTensor(gatesLocal);
    }

private:
    TPipe pipe;
    int32_t blockNum = {0};
    int32_t curBlockIdx = {0};
    int32_t startOffset = {0};
    int32_t endOffset = {0};
    int32_t innerOffset = {16};
    int32_t batchSize;
    int32_t samples;
    int32_t capacity;
    int32_t hidden;
    int32_t taskNumPerBlock;
    int32_t idx0;
    int32_t idx1;
    int32_t offset;
    int32_t taskNum;
    int32_t innerTask;
    int32_t moveCounts;

    GlobalTensor<T> xGm, yGradGm, gatesGradGm;
    GlobalTensor<int32_t> indicesGm, locationsGm;

    TQue<QuePosition::VECIN, 1> inQueueX, inQueueY, inQueueIndices, inQueueLocations;
    TQue<QuePosition::VECOUT, 1> outQueueGates, mulQueue, sumQueue, workQueue, tmpQueue;
};
#endif  // MOE_TUTEL_COMBINE_GATES_32B_
