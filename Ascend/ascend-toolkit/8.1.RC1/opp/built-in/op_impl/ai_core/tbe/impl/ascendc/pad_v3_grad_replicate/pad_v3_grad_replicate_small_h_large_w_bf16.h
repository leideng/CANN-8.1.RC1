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
 * \file pad_v3_grad_replicate_small_h_large_w_bf16.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_SMALL_H_LARGE_W_BF16_H_
#define _PAD_V3_GRAD_REPLICATE_SMALL_H_LARGE_W_BF16_H_

#include "pad_v3_grad_replicate_base.h"

template <typename T>
class PadV3GradReplicateSmallHLargeWBf16 {
public:
    __aicore__ inline PadV3GradReplicateSmallHLargeWBf16() {};
    __aicore__ inline void Init(const PadV3GradReplicateTilingData &__restrict tilingData, 
                                GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void CopyGm2UB(const int32_t cycleIdx, const int64_t copyCount, const int32_t batchIdx);
    __aicore__ inline void CopyOut2Gm(const int32_t batchIdx, const int32_t cycles, const int32_t flag);
    __aicore__ inline void CopyWs2UB(const int32_t batchIdx, const int64_t copyCount, const int32_t flag);
    __aicore__ inline void CopyOut2Workspace(const int32_t tIdx, const int64_t calCount);
    __aicore__ inline void ComputeHGrad(const int32_t calCount);
    __aicore__ inline void ImplTransposeAndCompute(const int64_t transCount, const int32_t flag);
    __aicore__ inline void CopyIn(const int32_t copyCount, const int64_t workspaceOffset);
    __aicore__ inline void Compute(const int32_t copyCount);
    __aicore__ inline void CopyOut(const int32_t copyCount, const int64_t offset);
    __aicore__ inline void Process();

private:
    TPipe *pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, 1> xInQueue;
    TQue<QuePosition::VECOUT, 1> yOutQueue;
    TBuf<TPosition::VECCALC> transposeBuf;
    TBuf<TPosition::VECCALC> floatCastResBuf;
    LocalTensor<float> floatTenosr;
    LocalTensor<float> transposeData;
    
    uint32_t batch = 0;
    uint32_t ncPerCore = 0;
    uint32_t tailNC = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t alignHeight = 0;
    uint32_t alignWidth = 0;
    uint32_t outHeight = 0;
    uint32_t outWidth = 0;
    uint32_t alignOutHeight = 0;
    uint32_t alignOutWidth = 0;
    uint32_t padTop = 0;
    uint32_t padBottom = 0;
    uint32_t padLeft = 0;
    uint32_t padRight = 0;
    uint32_t blockNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t blockIdx = 0;
    uint32_t perBlockCount = 0;
    uint64_t workspacePerCore = 0;
    int64_t batchStride = 0;
    int64_t outBatchStride = 0;
    uint32_t loopNC = 0;
    int64_t ncOffset = 0;
    event_t MTE3ToMTE2Event;

    GlobalTensor<T> mGmX;
    GlobalTensor<T> mGmY;
    GlobalTensor<T> mGmWorkspace;
};

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWBf16<T>::Init(
                                                    const PadV3GradReplicateTilingData &__restrict tilingData,
                                                    GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace) {
    batch = tilingData.batch;
    ncPerCore = tilingData.ncPerCore;
    tailNC = tilingData.tailNC;
    height = tilingData.height;
    width = tilingData.width;
    outHeight = tilingData.outHeight;
    outWidth = tilingData.outWidth;
    alignHeight = tilingData.alignHeight;
    alignWidth = tilingData.alignWidth;
    alignOutHeight = tilingData.alignOutHeight;
    alignOutWidth = tilingData.alignOutWidth;
    padTop = tilingData.padTop;
    padBottom = tilingData.padBottom;
    padLeft = tilingData.padLeft;
    padRight = tilingData.padRight;
    blockNum = tilingData.blockNum;
    ubFactorElement = tilingData.ubFactorElement;
    workspacePerCore = tilingData.workspacePerCore / sizeof(T);

    batchStride = height * width;
    outBatchStride = outHeight * outWidth;
    blockIdx = GetBlockIdx();
    perBlockCount = BLOCK_BYTES / sizeof(T);
    
    if (blockIdx < tailNC) {
        loopNC = ncPerCore + 1;
        ncOffset = blockIdx * loopNC;
    } else {
        loopNC = ncPerCore;
        ncOffset = blockIdx * ncPerCore + tailNC;
    }

    mGmX.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    mGmY.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
    mGmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
}

// init used buffer
template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWBf16<T>::InitBuffer(TPipe *inputPipe) {
    pipe = inputPipe;
    pipe->InitBuffer(xInQueue, 1, SMALL_HEIGHT_LIMIT * ubFactorElement * sizeof(T));
    pipe->InitBuffer(yOutQueue, 1, SMALL_HEIGHT_LIMIT * ubFactorElement * sizeof(T));
    pipe->InitBuffer(transposeBuf, SMALL_HEIGHT_LIMIT * ubFactorElement * sizeof(float));
    pipe->InitBuffer(floatCastResBuf, SMALL_HEIGHT_LIMIT * ubFactorElement * sizeof(float));
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWBf16<T>::CopyGm2UB(const int32_t cycleIdx, const int64_t copyCount,
                                                              const int32_t batchIdx) {
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    int32_t alignCopyCount = CeilAlign(copyCount, perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(alignCopyCount - copyCount), (T)0};
    int64_t offset = 0;
    for (size_t i = 0; i < height; i++) {
        offset = i * width + cycleIdx * ubFactorElement + batchIdx * batchStride + ncOffset * batchStride;
        DataCopyPad(xLocal[i * ubFactorElement], mGmX[offset], copyParams, padParams);
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWBf16<T>::CopyOut2Gm(const int32_t batchIdx, const int32_t cycles,
                                                                         const int32_t flag) {
    int64_t gmYOffset = 0;
    DataCopyExtParams leftCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - padLeft) * sizeof(T)), 0, 0, 0};
    DataCopyExtParams rightCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - padRight) * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    if (flag == 0) {
        for (size_t i = 0; i < cycles; i++) {
            gmYOffset = outWidth * i + batchIdx * outBatchStride + ncOffset * outBatchStride;
            DataCopyPad(mGmY[gmYOffset], yLocal[i * COPY_ROWS_AND_COLS], leftCopyParams);
        }
    } else {
        for (size_t i = 0; i < cycles; i++) {
            gmYOffset = outWidth * (i + 1) - (COPY_ROWS_AND_COLS - padRight) + batchIdx * outBatchStride +
                        ncOffset * outBatchStride;
            DataCopyPad(mGmY[gmYOffset], yLocal[i * COPY_ROWS_AND_COLS], rightCopyParams);
        }
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWBf16<T>::CopyWs2UB(const int32_t batchIdx, const int64_t copyCount,
                                                                        const int32_t flag) {
    DataCopyExtParams copyParams{1, (uint32_t)(COPY_ROWS_AND_COLS * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    if (flag == 0) {
        for (size_t i = 0; i < outHeight; i++) {
            workspaceOffset = i * width + blockIdx * workspacePerCore;
            DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmWorkspace[workspaceOffset], copyParams, padParams);
        }
    } else {
        for (size_t i = 0; i < outHeight; i++) {
            workspaceOffset = (i + 1) * width - COPY_ROWS_AND_COLS + blockIdx * workspacePerCore;
            DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmWorkspace[workspaceOffset], copyParams, padParams);
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWBf16<T>::CopyOut2Workspace(const int32_t tIdx,
                                                                                const int64_t calCount) {
    int64_t workspaceOffset;
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    for (size_t i = 0; i < outHeight; i++) {
        workspaceOffset = i * width + tIdx * ubFactorElement + blockIdx * workspacePerCore;
        DataCopyPad(mGmWorkspace[workspaceOffset], yLocal[i * ubFactorElement], copyParams);
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWBf16<T>::ImplTransposeAndCompute(const int64_t transCount,
                                                                                      const int32_t flag) {
    uint32_t loopTimes = CeilDiv(transCount, TRANSDATA_BASE_H);
    uint64_t xSrcLocalList0[TRANSDATA_BASE_H];
    uint64_t xDstLocalList0[TRANSDATA_BASE_H];
    uint64_t xSrcLocalList1[TRANSDATA_BASE_H];
    uint64_t xDstLocalList1[TRANSDATA_BASE_H];
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = 1;
    transDataParams.dstRepStride = 0;
    transDataParams.srcRepStride = 0;
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, SMALL_HEIGHT_LIMIT * ubFactorElement);
    for (size_t time = 0; time < COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM; time++) {
        for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList0[i] = (uint64_t)(floatTenosr[COPY_ROWS_AND_COLS * i + FLOAT_BLOCK_NUM * time].GetPhyAddr());
        }
        for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
            xDstLocalList0[CONST_VALUE_2 * i] =
                (uint64_t)(transposeData[i * ubFactorElement + FLOAT_BLOCK_NUM * ubFactorElement * time]
                                .GetPhyAddr());
            xDstLocalList0[CONST_VALUE_2 * i + 1] =
                (uint64_t)(transposeData[i * ubFactorElement + FLOAT_BLOCK_NUM * ubFactorElement * time +
                                            FLOAT_BLOCK_NUM]
                                .GetPhyAddr());
        }
        transDataParams.repeatTimes = loopTimes;
        transDataParams.srcRepStride = TRANSDATA_BASE_H * COPY_ROWS_AND_COLS * sizeof(float) / DATA_BLOCK_BYTES;
        transDataParams.dstRepStride = COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM;
        TransDataTo5HD<float>(xDstLocalList0, xSrcLocalList0, transDataParams);
    }
    if (flag == 0) {
        for (size_t i = 0; i < padLeft; i++) {
            Add(transposeData[padLeft * ubFactorElement], transposeData[i * ubFactorElement],
                transposeData[padLeft * ubFactorElement], ubFactorElement);
        }
        DataCopy(transposeData, transposeData[padLeft * ubFactorElement],
                    (COPY_ROWS_AND_COLS - padLeft) * ubFactorElement);

    } else {
        for (size_t i = 0; i < padRight; i++) {
            Add(transposeData[(COPY_ROWS_AND_COLS - 1 - padRight) * ubFactorElement],
                transposeData[(COPY_ROWS_AND_COLS - 1 - i) * ubFactorElement],
                transposeData[(COPY_ROWS_AND_COLS - 1 - padRight) * ubFactorElement], ubFactorElement);
        }
    }
    for (size_t time = 0; time < ubFactorElement / FLOAT_BLOCK_NUM; time++) {
        for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList1[i] =
                (uint64_t)(transposeData[ubFactorElement * i + time * FLOAT_BLOCK_NUM].GetPhyAddr());
        }
        for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
            xDstLocalList1[CONST_VALUE_2 * i] =
                (uint64_t)(floatTenosr[COPY_ROWS_AND_COLS * i + time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM]
                                .GetPhyAddr());
            xDstLocalList1[CONST_VALUE_2 * i + 1] =
                (uint64_t)(floatTenosr[COPY_ROWS_AND_COLS * i + time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM +
                                    FLOAT_BLOCK_NUM]
                                .GetPhyAddr());
        }
        transDataParams.repeatTimes = 1;
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
        TransDataTo5HD<float>(xDstLocalList1, xSrcLocalList1, transDataParams);
    }
    Cast(yLocal, floatTenosr, RoundMode::CAST_RINT, SMALL_HEIGHT_LIMIT * ubFactorElement);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWBf16<T>::ComputeHGrad(const int32_t calCount) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, SMALL_HEIGHT_LIMIT * ubFactorElement);
    // Compute grad
    for (size_t i = 0; i < padTop; i++) {
        Add(floatTenosr[padTop * ubFactorElement], floatTenosr[i * ubFactorElement],
            floatTenosr[padTop * ubFactorElement], calCount);
    }
    for (size_t i = 0; i < padBottom; i++) {
        Add(floatTenosr[(height - 1 - padBottom) * ubFactorElement], floatTenosr[(height - 1 - i) * ubFactorElement],
            floatTenosr[(height - 1 - padBottom) * ubFactorElement], calCount);
    }
    Cast(yLocal, floatTenosr[padTop * ubFactorElement], RoundMode::CAST_RINT, outHeight * ubFactorElement);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWBf16<T>::CopyIn(const int32_t copyCount,
                                                                     const int64_t workspaceOffset) {
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    DataCopyPad(xLocal, mGmWorkspace[workspaceOffset], copyParams, padParams);
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWBf16<T>::Compute(const int32_t copyCount) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    uint32_t alignCopyCount = CeilAlign(copyCount, perBlockCount);
    DataCopy(yLocal, xLocal, alignCopyCount);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWBf16<T>::CopyOut(const int32_t copyCount, const int64_t offset) {
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(mGmY[offset], yLocal, copyParams);
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWBf16<T>::Process() {
    int64_t gmYOffset;
    int64_t workspaceOffset;
    int64_t calCount = ubFactorElement;
    uint32_t copyCount = SMALL_HEIGHT_LIMIT * ubFactorElement;
    uint32_t copyTimesOneRow = CeilDiv(width, ubFactorElement);
    uint32_t copyMidDataTimes =
        CeilDiv(width - CONST_VALUE_2 * COPY_ROWS_AND_COLS, SMALL_HEIGHT_LIMIT * ubFactorElement);
    floatTenosr = floatCastResBuf.Get<float>();
    transposeData = transposeBuf.Get<float>();
    MTE3ToMTE2Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    for (size_t loop = 0; loop < loopNC; loop++) {
        calCount = ubFactorElement;
        for (size_t time = 0; time < copyTimesOneRow; time++) {
            if (time == copyTimesOneRow - 1) {
                calCount = width - (copyTimesOneRow - 1) * ubFactorElement;
            }
            CopyGm2UB(time, calCount, loop);
            ComputeHGrad(calCount);
            CopyOut2Workspace(time, calCount);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
        wait_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
        for (size_t i = 0; i < outHeight; i++) {
            copyCount = SMALL_HEIGHT_LIMIT * ubFactorElement;
            for (size_t j = 0; j < copyMidDataTimes; j++) {
                if (j == copyMidDataTimes - 1) {
                    copyCount =
                        width - CONST_VALUE_2 * COPY_ROWS_AND_COLS -
                        (copyMidDataTimes - 1) * ubFactorElement * SMALL_HEIGHT_LIMIT;
                }
                workspaceOffset = COPY_ROWS_AND_COLS + j * ubFactorElement * SMALL_HEIGHT_LIMIT + i * width +
                                  blockIdx * workspacePerCore;
                gmYOffset = COPY_ROWS_AND_COLS - padLeft + j * ubFactorElement * SMALL_HEIGHT_LIMIT + i * outWidth +
                            loop * outBatchStride + ncOffset * outBatchStride;
                CopyIn(copyCount, workspaceOffset);
                Compute(ubFactorElement * SMALL_HEIGHT_LIMIT);
                CopyOut(copyCount, gmYOffset);
            }
        }
        CopyWs2UB(loop, COPY_ROWS_AND_COLS, 0);
        ImplTransposeAndCompute(ubFactorElement, 0);
        CopyOut2Gm(loop, outHeight, 0);
        CopyWs2UB(loop, COPY_ROWS_AND_COLS, 1);
        ImplTransposeAndCompute(ubFactorElement, 1);
        CopyOut2Gm(loop, outHeight, 1);
    }
}
#endif  // _PAD_V3_GRAD_REPLICATE_SMALL_H_LARGE_W_BF16_H_