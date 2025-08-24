/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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
 * \file swi_glu_quant.h
 * \brief
 */
#ifndef SWI_GLU_QUANT_H
#define SWI_GLU_QUANT_H

#include "swi_glu_quant_base.h"

namespace SwiGluQuantOpt {
using namespace AscendC;

template <typename inType, typename outType>
class SwiGluQuant : public SwiGluQuantBase {
public:
    __aicore__ inline SwiGluQuant(TPipe *pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(GM_ADDR input_gm, GM_ADDR smooth_scales, GM_ADDR offsets, GM_ADDR group_index,
        GM_ADDR y_gm, GM_ADDR scale_gm, GM_ADDR workspace, const SwiGluQuantTilingData *__restrict tilingData)
    {
        ParseTilingData(tilingData);
        InitParams();
        InitBaseBuffer();
        InitAndSetBuffer(input_gm, smooth_scales, offsets, group_index, y_gm, scale_gm);
    }

    __aicore__ inline void Process()
    {
        GroupCopyIn();
        groupLocal = groupQueue.DeQue<int32_t>();

        DuplicateConst();
        ProcessCoreMultiUbMulti();

        groupQueue.FreeTensor(groupLocal);
    }

private:
    __aicore__ inline void InitParams()
    {   
        mergedColLen = SPLIT_NUM * tilingData_.colLen;
        colLen = tilingData_.colLen;
        basicColLen = tilingData_.basicColLen;

        coreIdx = static_cast<uint32_t>(GetBlockIdx());
        headCoreNum = tilingData_.headCoreNum;

        if (coreIdx < headCoreNum) {
            rowLenPerCore = tilingData_.rowLenPerHeadCore;
            basicRowLen = tilingData_.basicRowLenHeadCore;
            rowLoop = CeilDiv(rowLenPerCore, basicRowLen);
            baseRow = coreIdx * rowLenPerCore;
        }
        else if (coreIdx >= headCoreNum && coreIdx < tilingData_.realCoreNum) {
            rowLenPerCore = tilingData_.rowLenPerTailCore;
            basicRowLen = tilingData_.basicRowLenTailCore;
            rowLoop = CeilDiv(rowLenPerCore, basicRowLen);
            baseRow = headCoreNum * tilingData_.rowLenPerHeadCore + (coreIdx - headCoreNum) * rowLenPerCore;
        }

        outAlignLen = AlignUp(basicColLen, SWI_GLU_QUANT_THIRTY_TWO);
        outLen = basicRowLen * (outAlignLen == 0 ? (BLOCK_SIZE / sizeof(outType)) : outAlignLen);

        alignedGroupLen = AlignUp(tilingData_.groupLen, SWI_GLU_QUANT_EIGHT);

        uint32_t alignedNum = BLOCK_SIZE / sizeof(inType);
        sizeHalfLen = AlignUp(basicColLen, alignedNum);
        tileLength = basicRowLen * (sizeHalfLen == 0 ? (BLOCK_SIZE / sizeof(inType)) : sizeHalfLen);
        rightPadding = sizeHalfLen - basicColLen;
        isPad = (rightPadding > 0);
        blockUnit = (isPad) ? 1 : BLOCK_SIZE;

        smoothSizeFloatLen = AlignUp(basicColLen, SWI_GLU_QUANT_EIGHT);
        smoothRightPadding = smoothSizeFloatLen - tilingData_.basicColLen;
        smoothIsPad = (smoothRightPadding > 0);
    }

    __aicore__ inline void InitAndSetBuffer(GM_ADDR input_gm, GM_ADDR smooth_scales, GM_ADDR offsets,
        GM_ADDR group_index, GM_ADDR y_gm, GM_ADDR scale_gm)
    {
        // gm数据
        xGm.SetGlobalBuffer((__gm__ inType *)input_gm, SPLIT_NUM * tilingData_.rowLen * tilingData_.colLen);
        yGm.SetGlobalBuffer((__gm__ outType *)y_gm, tilingData_.rowLen * tilingData_.colLen);
        smooth_scales_Gm.SetGlobalBuffer((__gm__ float *)smooth_scales, tilingData_.groupLen * tilingData_.colLen);
        group_index_Gm.SetGlobalBuffer((__gm__ int32_t *)group_index, tilingData_.groupLen);
        offsetsGm.SetGlobalBuffer((__gm__ float *)offsets, tilingData_.groupLen);
        scale_Gm.SetGlobalBuffer((__gm__ float *)scale_gm, tilingData_.rowLen);

        // queue
        pPipe->InitBuffer(inQueueA, BUFFER_NUM, tileLength * sizeof(inType));
        pPipe->InitBuffer(inQueueB, BUFFER_NUM, tileLength * sizeof(inType));
        // todo
        pPipe->InitBuffer(outQueueY, BUFFER_NUM, outLen * sizeof(outType));
        pPipe->InitBuffer(scaleQueue, BUFFER_NUM, basicRowLen * sizeof(float));
        pPipe->InitBuffer(groupQueue, BUFFER_NUM, alignedGroupLen * sizeof(int32_t));
        pPipe->InitBuffer(smoothQueue, BUFFER_NUM, sizeHalfLen * sizeof(float));

        // 定义过程变量
        pPipe->InitBuffer(sharedTempBuf, tileLength * sizeof(float));
        pPipe->InitBuffer(tempBufferY, tileLength * sizeof(float));
        pPipe->InitBuffer(tempYUnit, sizeHalfLen * sizeof(float));
    }

    __aicore__ inline uint32_t GetSmoothIndex(uint32_t realRowNum, int32_t &groupNum, uint32_t smoothIndex)
    {
        // 获取符合条件的smooth_scales的index
        for (size_t index = smoothIndex; index < tilingData_.groupLen; index++) {
            groupNum = groupLocal.GetValue(index);
            if (groupNum >= realRowNum) {
                return index;
            }
        }
        return SMOOTH_INDEX_UPBOUND;
    }

    __aicore__ inline void GroupCopyIn()
    {
        LocalTensor<int32_t> groupLocal = groupQueue.AllocTensor<int32_t>();
        uint8_t rightPadding = alignedGroupLen - tilingData_.groupLen;
        DataCopyParams copyParams{1, (uint16_t)(tilingData_.groupLen * sizeof(int32_t)), 0, 0};
        DataCopyPadParams padParams{true, 0, rightPadding, 0};
        DataCopyPad(groupLocal, group_index_Gm, copyParams, padParams);
        groupQueue.EnQue(groupLocal);
    }

    __aicore__ inline void SmoothCopyIn(uint32_t offset)
    {
        LocalTensor<float> smoothLocal = smoothQueue.AllocTensor<float>();
        if (smoothIsPad) {
            DataCopyParams copyParams{1, (uint16_t)(basicColLen * sizeof(float)), 0, 0};
            DataCopyPadParams padParams{false, 0, smoothRightPadding, 0};
            DataCopyPad(smoothLocal, smooth_scales_Gm[offset], copyParams, padParams);
        } else {
            DataCopy(smoothLocal, smooth_scales_Gm[offset], basicColLen);
        }
        smoothQueue.EnQue(smoothLocal);
    }

    __aicore__ inline void ProcessCoreMultiUbMulti()
    {
        uint32_t smoothIndex = 0;
        uint32_t offsetRow = 0;

        for (uint32_t ridx = 0; ridx < rowLoop; ridx++) {
            offsetRow = baseRow + ridx * basicRowLen;

            // 处理最后一行
            basicRowLenCal = static_cast<uint16_t>((ridx == rowLoop - 1)
                                                       ? (rowLenPerCore - (rowLoop - 1) * basicRowLen)
                                                       : basicRowLen);  // 每核处理的最后一个行循环单独处理
            ProcessCoreMultiUbMultiAlign(ridx, smoothIndex, offsetRow);
        }
    }

    __aicore__ inline void ComputeVecInGmOffset(uint32_t ridx)
    {
        if (coreIdx < headCoreNum) {
            offsetParam.tmpVecGmOffset = coreIdx * rowLenPerCore * mergedColLen + ridx * basicRowLen * mergedColLen;
            splitCopyoutOffset = coreIdx * rowLenPerCore * colLen + ridx * basicRowLen * basicColLen;
        }
        else {
            offsetParam.tmpVecGmOffset = headCoreNum * tilingData_.rowLenPerHeadCore * mergedColLen +
                                         (coreIdx - headCoreNum) * rowLenPerCore * mergedColLen +
                                         ridx * basicRowLen * mergedColLen;
            splitCopyoutOffset = headCoreNum * tilingData_.rowLenPerHeadCore * colLen +
                                 (coreIdx - headCoreNum) * rowLenPerCore * colLen +
                                 ridx * basicRowLen * basicColLen;
        }
    }

    __aicore__ inline void ProcessCoreMultiUbMultiAlign(uint32_t ridx, uint32_t &smoothIndex, uint16_t offsetRow)
    {
        DataCopyParams splitCopyinParams;
        DataCopyParams splitCopyoutParams;

        splitCopyinParams = {basicRowLenCal,
            (uint16_t)(basicColLen * sizeof(inType) / blockUnit),
            (uint16_t)((mergedColLen - basicColLen) * sizeof(inType) / blockUnit),
            0};

        splitCopyoutParams = {basicRowLenCal,
            (uint16_t)(basicColLen * sizeof(outType)),
            0,
            (uint16_t)((colLen - basicColLen) * sizeof(outType))};

        ComputeVecInGmOffset(ridx);

        if (tilingData_.activateLeft == 1){
            offsetParam.splitVecGmOffset1 = offsetParam.tmpVecGmOffset;
            offsetParam.splitVecGmOffset2 = offsetParam.splitVecGmOffset1 + tilingData_.colLen;
        }
        else{
            offsetParam.splitVecGmOffset2 = offsetParam.tmpVecGmOffset;
            offsetParam.splitVecGmOffset1 = offsetParam.splitVecGmOffset2 + tilingData_.colLen;
        }

        uint32_t smoothScalesOffset = smoothIndex * tilingData_.colLen;

        CopyIn(offsetParam, smoothScalesOffset, splitCopyinParams);
        Compute(offsetRow, smoothIndex);
        CopyOut(splitCopyoutOffset, splitCopyoutParams, ridx, basicRowLenCal);
    }

    __aicore__ inline void CopyIn(
        XxGluSingleTileOffsetParam &offsetParam, uint32_t smoothScalesOffset, DataCopyParams &splitCopyinParams)
    {
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();

        if (isPad) {
            // Copy A
            DataCopyPadParams padParams{false, 0, rightPadding, 0};
            DataCopyPad(aLocal, this->xGm[offsetParam.splitVecGmOffset1], splitCopyinParams, padParams);
            // Copy B
            DataCopyPad(bLocal, this->xGm[offsetParam.splitVecGmOffset2], splitCopyinParams, padParams);
        }else {
            // Copy A
            DataCopy(aLocal, this->xGm[offsetParam.splitVecGmOffset1], splitCopyinParams);
            // Copy B
            DataCopy(bLocal, this->xGm[offsetParam.splitVecGmOffset2], splitCopyinParams);
        }

        this->inQueueA.template EnQue(aLocal);
        this->inQueueB.template EnQue(bLocal);
        // Copy Scales
        SmoothCopyIn(smoothScalesOffset);
    }

    __aicore__ inline void CopyOut(
        uint64_t splitCopyoutOffset, DataCopyParams &splitCopyoutParams, uint32_t ridx, uint32_t basicRowLenCal)
    {
        LocalTensor<outType> outLocal = outQueueY.DeQue<outType>();

        DataCopyPad(yGm[splitCopyoutOffset], outLocal, splitCopyoutParams);

        outQueueY.FreeTensor(outLocal);

        LocalTensor<float> scaleLocal = scaleQueue.DeQue<float>();
        DataCopyParams copyParams1{1, (uint16_t)(basicRowLenCal * sizeof(float)), 0, 0};

        DataCopyPad(scale_Gm[baseRow + basicRowLen * ridx], scaleLocal, copyParams1);
        scaleQueue.FreeTensor(scaleLocal);
    }

    __aicore__ inline void Compute(uint32_t offsetRow, uint32_t &smoothIndex)
    {
        LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
        LocalTensor<float> tmpALocal = sharedTempBuf.Get<float>();
        LocalTensor<float> tmpYLocal = tempBufferY.Get<float>();
        LocalTensor<inType> aLocal = inQueueA.template DeQue<inType>();

        if constexpr (sizeof(inType) == sizeof(float)) {
            DataCopy(tmpALocal, aLocal, tileLength);
        } else {
            Cast(tmpALocal, aLocal, RoundMode::CAST_NONE, tileLength);
        }

        inQueueA.template FreeTensor(aLocal);
        Muls(tmpYLocal, tmpALocal, static_cast<float>(-1.0), tileLength);
        PipeBarrier<PIPE_V>();
        Exp(tmpYLocal, tmpYLocal, tileLength);
        PipeBarrier<PIPE_V>();
        Adds(tmpYLocal, tmpYLocal, static_cast<float>(1.0), tileLength);
        PipeBarrier<PIPE_V>();
        Div(tmpYLocal, tmpALocal, tmpYLocal, tileLength);
        PipeBarrier<PIPE_V>();

        LocalTensor<inType> bLocal = inQueueB.template DeQue<inType>();
        LocalTensor<float> tmpBLocal = sharedTempBuf.Get<float>();
        if constexpr (sizeof(inType) == sizeof(float)) {
            DataCopy(tmpBLocal, bLocal, tileLength);
        } else {
            Cast(tmpBLocal, bLocal, RoundMode::CAST_NONE, tileLength);
        }

        inQueueB.template FreeTensor(bLocal);
        // PipeBarrier<PIPE_V>();
        Mul(tmpYLocal, tmpYLocal, tmpBLocal, tileLength);
        PipeBarrier<PIPE_V>();

        /**  quant相关的计算  */
        uint32_t index = 0;
        uint32_t smoothOffset = 0;
        uint32_t realRowNum = 0;
        int32_t groupValue = groupLocal.GetValue(smoothIndex);
        LocalTensor<float> smoothLocal = smoothQueue.DeQue<float>();

        LocalTensor<float> tempFp32 = tempYUnit.Get<float>();
        LocalTensor<float> tmpLocal = sharedTempBuf.Get<float>(sizeHalfLen);
        LocalTensor<int32_t> tempInt32 = sharedTempBuf.Get<int32_t>(sizeHalfLen);
        auto tempHalf = tempFp32.ReinterpretCast<half>();

        LocalTensor<outType> outLocal = outQueueY.AllocTensor<outType>();
        for (int32_t i = 0; i < basicRowLenCal; i++) {
            index = i * sizeHalfLen;
            DataCopy(tempFp32, tmpYLocal[index], sizeHalfLen);

            realRowNum = offsetRow + i + 1;
            if (groupValue < realRowNum && smoothIndex != SMOOTH_INDEX_UPBOUND) {
                smoothIndex = GetSmoothIndex(realRowNum, groupValue, smoothIndex + 1);
                if (smoothIndex != SMOOTH_INDEX_UPBOUND) {
                    smoothQueue.FreeTensor(smoothLocal);
                    smoothOffset = smoothIndex * basicColLen;
                    SmoothCopyIn(smoothOffset);
                    smoothLocal = smoothQueue.DeQue<float>();
                }
            }

            if (smoothIndex != SMOOTH_INDEX_UPBOUND) {
                Mul(tempFp32, tempFp32, smoothLocal, basicColLen);
                pipe_barrier(PIPE_V);
            }

            Abs(tmpLocal, tempFp32, basicColLen);
            pipe_barrier(PIPE_V);
            ReduceMax(tmpLocal, tmpLocal, tmpLocal, basicColLen, false);
            pipe_barrier(PIPE_V);

            Div(tmpLocal, constScale, tmpLocal, MAX_VALUE_NUM);
            pipe_barrier(PIPE_V);

            float scale = tmpLocal.GetValue(0);
            scaleLocal.SetValue(i, scale);

            // y_tmp * dynamic_scale
            Muls(tempFp32, tempFp32, scale, basicColLen);
            pipe_barrier(PIPE_V);
            Cast(tempInt32, tempFp32, RoundMode::CAST_RINT, basicColLen);
            pipe_barrier(PIPE_V);

            SetDeqScale(static_cast<half>(1.0));
            pipe_barrier(PIPE_V);
            Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, basicColLen);
            pipe_barrier(PIPE_V);

            Cast(outLocal[i * outAlignLen], tempHalf, RoundMode::CAST_TRUNC, basicColLen);
        }

        smoothQueue.FreeTensor(smoothLocal);
        outQueueY.template EnQue<outType>(outLocal);
        scaleQueue.EnQue<float>(scaleLocal);
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    // quant
    TQue<QuePosition::VECIN, BUFFER_NUM> groupQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scaleQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> smoothQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> offsetsQueue;

    TBuf<TPosition::VECCALC> sharedTempBuf;
    TBuf<TPosition::VECCALC> tempBufferY;
    TBuf<TPosition::VECCALC> fp32_buf_;
    TBuf<TPosition::VECCALC> tempYUnit;

    GlobalTensor<inType> xGm;
    GlobalTensor<outType> yGm;
    GlobalTensor<float> smooth_scales_Gm;
    GlobalTensor<float> offsetsGm;
    GlobalTensor<float> scale_Gm;
    GlobalTensor<int32_t> group_index_Gm;

    LocalTensor<int32_t> groupLocal;

    uint64_t splitCopyoutOffset;
};
}  // namespace SwiGluQuantOpt
#endif  // SWI_GLU_QUANT_H