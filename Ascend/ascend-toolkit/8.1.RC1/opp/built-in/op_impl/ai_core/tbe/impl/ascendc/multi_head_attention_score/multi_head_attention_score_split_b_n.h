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
 * \file multi_head_attention_score_split_b_n.h
 * \brief
 */
#ifndef MULTI_HEAD_ATTENTION_SCORE_SPLIT_B_N_H
#define MULTI_HEAD_ATTENTION_SCORE_SPLIT_B_N_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

template <typename T, bool isBasicBlock = false>
class AttentionScoreSplitBN {
public:
    __aicore__ inline AttentionScoreSplitBN() {};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t*  value,
                                __gm__ uint8_t* pse_shift, __gm__ uint8_t* dropMask, __gm__ uint8_t* attenMask,
                                __gm__ uint8_t* softmax_out, __gm__ uint8_t* attention_out, __gm__ uint8_t* workspace,
                                const MultiHeadAttentionScoreTilingData* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void Process();
    // define matmul1
    using a1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using b1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using bias1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = matmul::MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, T>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type> bmm1;
   // define matmul2
    using a2Type = matmul::MatmulType<TPosition::TSCM, CubeFormat::NZ, T, false>;
    using b2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using bias2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    matmul::Matmul<a2Type, b2Type, c2Type, bias2Type> bmm2;

protected:
    const MultiHeadAttentionScoreTilingData* __restrict tilingData;
    TPipe* pipe;
    // define the que
    TSCM<TPosition::VECCALC> scm;
    TQue<QuePosition::VECIN, 1> attenMaskQueue;
    TQue<QuePosition::VECIN, 1> pseQueue;
    TQue<QuePosition::VECOUT, 1> bmm1ResQueue;
    TBuf<> tmpSoftmaxUbBuf;

    GlobalTensor<T> queryGm;
    GlobalTensor<T> keyGm;
    GlobalTensor<T> valueGm;
    GlobalTensor<T> attenMaskGm;
    GlobalTensor<T> pseGm;
    GlobalTensor<T> softmaxOutGm;
    GlobalTensor<T> attentionOutGm;
 
    // tiling data
    uint32_t n;
    uint32_t bnRange;
    uint32_t bnRangeTail;
    uint32_t formerNum;
    uint32_t tailNum;
    uint32_t sOuterLoopTimes;
    uint32_t singleProcessSOuterSize;
    uint32_t singleProcessSOuterSizeTail;
    uint32_t singleProcessSInnerSize;
    uint32_t sInnerSizeAlign;
    uint32_t attenMaskSOuter;
    uint32_t mmResUbSize;
    uint32_t attenMaskUbSize;
    uint32_t pseUbSize;
    uint32_t softmaxMaxSize;
    uint32_t scmTmpSize;
    uint32_t oneBlockElems;

    uint32_t padSize;
    uint32_t curBlockIdx;
    uint64_t sOuterOffset;
    uint64_t queryOffset;
    uint64_t keyOffset;
    uint64_t valueOffset;
    uint64_t attenMaskOffset;
    uint64_t pseOffset;
    uint64_t softmaxOutOffset;
    uint64_t attentionOutOffset;

    __aicore__ inline void Bmm1ComputeFirstLoop(uint32_t eachSOuter);

    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t sInnerSize, uint32_t sInnerSizeAlign,
                                           uint32_t eachSOuter);

    __aicore__ inline void PseCopyIn(uint64_t offset, uint32_t sInnerSize, uint32_t sInnerSizeAlign,
                                     uint32_t eachSOuter);

    __aicore__ inline void Elewise2Compute(LocalTensor<T>& mmResUb, LocalTensor<T>& attenMaskUb, uint32_t computeSize);

    __aicore__ inline void SoftmaxCompute(LocalTensor<T>& mmResUb, uint32_t eachSOuter);

    __aicore__ inline void SoftmaxResCopyOut(LocalTensor<T>& mmResUb, uint64_t offset, uint32_t sInnerSize,
                                             uint32_t eachSOuter);

    __aicore__ inline void Bmm1ResCopy2L1(LocalTensor<T>& mmResUb, uint32_t sInnerSize, uint32_t eachSOuter);

    __aicore__ inline void Bmm2Compute(LocalTensor<T>& bmm1ResL1, uint64_t offset, uint32_t eachSOuter);

    __aicore__ inline void Bmm1ResDoVecBmm2Compute(LocalTensor<T>& mmResUb, uint64_t valueOffset, uint32_t eachSOuter);

    __aicore__ inline void EachLoopProcess(uint32_t eachSOuter);

    __aicore__ inline void EachLoopOffsetInit(uint64_t bIdx, uint64_t nIdx);

    __aicore__ inline void ComputeEachCore(uint32_t coreIdx);

    __aicore__ inline void ComputeEachCoreImpl(uint64_t bIdx, uint64_t nIdx);

    __aicore__ inline void initTilingData();
};

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::Init(__gm__ uint8_t* query, __gm__ uint8_t* key,
    __gm__ uint8_t* value, __gm__ uint8_t* pse_shift, __gm__ uint8_t* dropMask, __gm__ uint8_t* attenMask,
    __gm__ uint8_t* softmax_out, __gm__ uint8_t*  attention_out, __gm__ uint8_t*  workspace,
    const MultiHeadAttentionScoreTilingData* __restrict tiling, TPipe* tPipe) {
    curBlockIdx = GetBlockIdx();
    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ T*)query);
    keyGm.SetGlobalBuffer((__gm__ T*)key);
    valueGm.SetGlobalBuffer((__gm__ T*)value);
    attenMaskGm.SetGlobalBuffer((__gm__ T*)attenMask);
    pseGm.SetGlobalBuffer((__gm__ T*)pse_shift);
    softmaxOutGm.SetGlobalBuffer((__gm__ T*)softmax_out);
    attentionOutGm.SetGlobalBuffer((__gm__ T*)attention_out);

    tilingData = tiling;
    pipe = tPipe;

    initTilingData();
    padSize = sInnerSizeAlign - singleProcessSInnerSize;
    pipe->InitBuffer(attenMaskQueue, 1, attenMaskUbSize * sizeof(T));
    pipe->InitBuffer(pseQueue, 1, pseUbSize * sizeof(T));
    pipe->InitBuffer(bmm1ResQueue, 1, mmResUbSize * sizeof(T));
    pipe->InitBuffer(tmpSoftmaxUbBuf, 2 * softmaxMaxSize * sizeof(T));
    pipe->InitBuffer(scm, 1, scmTmpSize * sizeof(T));

}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::initTilingData() {
    n = tilingData->multiHeadAttentionBaseParams.headNumSize;
    bnRange = tilingData->multiHeadAttentionScoreCoreParams.bnRange;
    bnRangeTail = tilingData->multiHeadAttentionScoreCoreParams.bnRangeTail;
    formerNum = tilingData->multiHeadAttentionScoreCoreParams.formerNum;
    tailNum = tilingData->multiHeadAttentionScoreCoreParams.tailNum;
    sOuterLoopTimes = tilingData->multiHeadAttentionScoreSingleCoreParams.sOuterLoopTimes;
    singleProcessSOuterSize = tilingData->multiHeadAttentionScoreSingleCoreParams.singleProcessSOuterSize;
    singleProcessSOuterSizeTail = tilingData->multiHeadAttentionScoreSingleCoreParams.singleProcessSOuterSizeTail;
    singleProcessSInnerSize = tilingData->multiHeadAttentionBaseParams.seqInnerSize;
    sInnerSizeAlign = tilingData->multiHeadAttentionBaseParams.seqInnerSizeAlign;
    attenMaskSOuter = tilingData->multiHeadAttentionBaseParams.attenMaskSOuter;
    oneBlockElems = tilingData->multiHeadAttentionScoreOffestStrideParams.typeByteNum;

    mmResUbSize = tilingData->multiHeadAttentionScoreSingleCoreTensorSize.mmResUbSize;
    attenMaskUbSize = tilingData->multiHeadAttentionScoreSingleCoreTensorSize.attenMaskUbSize;
    pseUbSize = tilingData->multiHeadAttentionScoreSingleCoreTensorSize.pseUbSize;
    softmaxMaxSize = tilingData->multiHeadAttentionScoreSingleCoreTensorSize.softmaxMaxSize;
    scmTmpSize = tilingData->multiHeadAttentionScoreSingleCoreTensorSize.scmTmpSize;
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::Process() {
    ComputeEachCore(curBlockIdx);
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::Bmm1ComputeFirstLoop(uint32_t eachSOuter) {
    bmm1.SetTensorA(queryGm[queryOffset]);
    bmm1.SetTensorB(keyGm[keyOffset], true);
    bmm1.SetTail(eachSOuter, singleProcessSInnerSize);
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::AttenMaskCopyIn(uint64_t offset, uint32_t sInnerSize,
    uint32_t sInnerSizeAlign, uint32_t eachSOuter) {
    LocalTensor<T> attenMaskUb = attenMaskQueue.AllocTensor<T>();
    attenMaskUb.SetSize(eachSOuter * sInnerSizeAlign);
    uint32_t shapeArray[] = {eachSOuter, sInnerSizeAlign};
    attenMaskUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));

    if (attenMaskSOuter == 1) {
        for (uint32_t loopSOuterIdx = 0; loopSOuterIdx < eachSOuter; loopSOuterIdx++) {
            DataCopy(attenMaskUb[loopSOuterIdx * sInnerSizeAlign], attenMaskGm[offset], sInnerSizeAlign);
        }
    } else {
        DataCopyParams intriParams;
        intriParams.blockCount = eachSOuter;
        intriParams.blockLen = sInnerSize * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        DataCopyPadParams padParams;
        padParams.isPad = true;
        padParams.paddingValue = 0;
        padParams.leftPadding = 0;
        padParams.rightPadding = padSize;
        DataCopyPad(attenMaskUb, attenMaskGm[offset], intriParams, padParams);
    }

    attenMaskQueue.EnQue(attenMaskUb);
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::PseCopyIn(uint64_t offset, uint32_t sInnerSize,
    uint32_t sInnerSizeAlign, uint32_t eachSOuter) {
    LocalTensor<T> pseUb = pseQueue.AllocTensor<T>();
    pseUb.SetSize(eachSOuter * sInnerSizeAlign);
    uint32_t shapeArray[] = {eachSOuter, sInnerSizeAlign};
    pseUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));

    DataCopyParams intriParams;
    intriParams.blockCount = eachSOuter;
    intriParams.blockLen = sInnerSize * sizeof(T);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;

    DataCopyPadParams padParams;
    padParams.isPad = true;
    padParams.paddingValue = 0;
    padParams.leftPadding = 0;
    padParams.rightPadding = padSize;
    DataCopyPad(pseUb, pseGm[offset], intriParams, padParams);
    pseQueue.EnQue(pseUb);
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::Elewise2Compute(LocalTensor<T>& mmResUb,
    LocalTensor<T>& attenMaskUb, uint32_t computeSize) {
    LocalTensor<T> pseUb = pseQueue.DeQue<T>();
    Add(mmResUb, pseUb, mmResUb, computeSize);
    pipe_barrier(PIPE_V);
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->multiHeadAttentionBaseParams.scaleValue), computeSize);
    pipe_barrier(PIPE_V);
    Add(mmResUb, attenMaskUb, mmResUb, computeSize);

    pseQueue.FreeTensor(pseUb);
    attenMaskQueue.FreeTensor(attenMaskUb);
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::SoftmaxCompute(LocalTensor<T>& mmResUb,
    uint32_t eachSOuter) {
    uint32_t shapeArray[] = {eachSOuter, oneBlockElems};
    LocalTensor<T> softmaxMaxUb = tmpSoftmaxUbBuf.Get<T>(softmaxMaxSize * 2);
    LocalTensor<T> softmaxSumUb = softmaxMaxUb[softmaxMaxSize];
    softmaxMaxUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    softmaxSumUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    
    SoftMax<T, true> (mmResUb, softmaxSumUb, softmaxMaxUb, mmResUb, tilingData->softmaxTilingData);
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::SoftmaxResCopyOut(LocalTensor<T>& mmResUb,
    uint64_t offset, uint32_t sInnerSize, uint32_t eachSOuter) {
    DataCopyParams intriParams;
    intriParams.blockCount = eachSOuter;
    intriParams.blockLen = sInnerSize * sizeof(T);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    DataCopyPad(softmaxOutGm[offset], mmResUb, intriParams);
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::Bmm1ResCopy2L1(LocalTensor<T>& mmResUb,
    uint32_t sInnerSize, uint32_t eachSOuter) {
    auto bmm1ResL1 = scm.AllocTensor<T>();

    DataCopyParams intriParams;
    intriParams.blockCount = eachSOuter;
    intriParams.blockLen = sInnerSize * sizeof(T);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;

    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = eachSOuter;
    nd2nzParams.dValue = sInnerSize;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue = sInnerSize;
    nd2nzParams.dstNzC0Stride = (eachSOuter + 15) / 16 * 16;
    nd2nzParams.dstNzNStride = 1;

    DataCopyPad(bmm1ResL1, mmResUb, intriParams, nd2nzParams);
    scm.EnQue(bmm1ResL1);
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::Bmm2Compute(LocalTensor<T>& bmm1ResL1,
    uint64_t offset, uint32_t eachSOuter) {
    bmm2.SetTensorA(bmm1ResL1);
    bmm2.SetTensorB(valueGm[offset]);
    bmm2.SetTail(eachSOuter, -1, singleProcessSInnerSize);
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::Bmm1ResDoVecBmm2Compute(LocalTensor<T>& mmResUb,
    uint64_t valueOffset, uint32_t eachSOuter) {
    if (bmm1.Iterate()) {
        // atten_mask
        AttenMaskCopyIn(attenMaskOffset, singleProcessSInnerSize, sInnerSizeAlign, eachSOuter);

        uint32_t computeSize = eachSOuter * sInnerSizeAlign;
        LocalTensor<T> attenMaskUb = attenMaskQueue.DeQue<T>();
    
        // elewise1 compute
        Muls(attenMaskUb, attenMaskUb, static_cast<T>(-10000.0), computeSize);

        // pse_shift
        PseCopyIn(pseOffset, singleProcessSInnerSize, sInnerSizeAlign, eachSOuter);

        bmm1.GetTensorC(mmResUb, false, true);
        bmm1.End();

        // elewise2 compute
        Elewise2Compute(mmResUb, attenMaskUb, computeSize);

        // softmax
        SoftmaxCompute(mmResUb, eachSOuter);
        bmm1ResQueue.EnQue(mmResUb);
        bmm1ResQueue.DeQue<T>();

        // softmax out
        SoftmaxResCopyOut(mmResUb, softmaxOutOffset, singleProcessSInnerSize, eachSOuter);
    
        Bmm1ResCopy2L1(mmResUb, singleProcessSInnerSize, eachSOuter);
        bmm1ResQueue.FreeTensor(mmResUb);
    
        LocalTensor<T> bmm1ResL1 = scm.DeQue<T>();
        Bmm2Compute(bmm1ResL1, valueOffset, eachSOuter);
        bmm2.IterateAll(attentionOutGm[attentionOutOffset], false);
        scm.FreeTensor(bmm1ResL1);
    }
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::EachLoopProcess(uint32_t eachSOuter) {
    LocalTensor<T> mmResUb = bmm1ResQueue.AllocTensor<T>();
    uint32_t mm1ResShapeArray[] = {eachSOuter, sInnerSizeAlign};
    uint32_t mm1ResOirShapeArray[] = {eachSOuter, singleProcessSInnerSize};
    mmResUb.SetShapeInfo(ShapeInfo(2, mm1ResShapeArray, 2, mm1ResOirShapeArray, DataFormat::ND));

    Bmm1ComputeFirstLoop(eachSOuter);
    Bmm1ResDoVecBmm2Compute(mmResUb, valueOffset, eachSOuter);
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::EachLoopOffsetInit(uint64_t bIdx, uint64_t nIdx) {
    if (attenMaskSOuter == 1) {
        attenMaskOffset = bIdx * tilingData->multiHeadAttentionBaseParams.seqInnerSize;
    } else {
        attenMaskOffset = bIdx * tilingData->multiHeadAttentionBaseParams.seqSize * tilingData->multiHeadAttentionBaseParams.seqInnerSize +
                          sOuterOffset * tilingData->multiHeadAttentionBaseParams.seqInnerSize;
    }

    pseOffset = bIdx * tilingData->multiHeadAttentionBaseParams.headNumSize * tilingData->multiHeadAttentionBaseParams.seqSize *
        tilingData->multiHeadAttentionBaseParams.seqInnerSize +
        nIdx * tilingData->multiHeadAttentionBaseParams.seqSize * tilingData->multiHeadAttentionBaseParams.seqInnerSize +
        sOuterOffset * tilingData->multiHeadAttentionBaseParams.seqInnerSize;
    queryOffset = bIdx * tilingData->multiHeadAttentionScoreOffestStrideParams.matmulHead * tilingData->multiHeadAttentionBaseParams.seqSize +
        nIdx * tilingData->multiHeadAttentionBaseParams.headSize + sOuterOffset * tilingData->multiHeadAttentionScoreOffestStrideParams.matmulHead;
    keyOffset = bIdx * tilingData->multiHeadAttentionScoreOffestStrideParams.matmulHead * tilingData->multiHeadAttentionBaseParams.seqInnerSize +
        nIdx * tilingData->multiHeadAttentionBaseParams.headSize;
    valueOffset = keyOffset;

    softmaxOutOffset = pseOffset;
    attentionOutOffset = queryOffset;
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::ComputeEachCoreImpl(uint64_t bIdx, uint64_t nIdx) {
    uint32_t eachSOuter = singleProcessSOuterSize;

    for (uint64_t sOuterLoopIdx = 0; sOuterLoopIdx < sOuterLoopTimes; sOuterLoopIdx++) {
        // sOuter offset
        if (sOuterLoopIdx == sOuterLoopTimes - 1) {
            eachSOuter = singleProcessSOuterSizeTail;
        }
        sOuterOffset = sOuterLoopIdx * singleProcessSOuterSize;
        EachLoopOffsetInit(bIdx, nIdx);
        EachLoopProcess(eachSOuter);
    }
}

template<typename T, bool isBasicBlock>
__aicore__ inline void AttentionScoreSplitBN<T, isBasicBlock>::ComputeEachCore(uint32_t coreIdx) {
    uint64_t bIdx = 0;
    uint64_t nIdx = 0;
    if (coreIdx < formerNum) {
        for (uint64_t loopBnIdx = 0; loopBnIdx < bnRange; loopBnIdx++) {
            bIdx = (coreIdx * bnRange + loopBnIdx) / n;
            nIdx = (coreIdx * bnRange + loopBnIdx) % n;
            ComputeEachCoreImpl(bIdx, nIdx);
        }
    } else {
        for (uint64_t loopBnIdx = 0; loopBnIdx < bnRangeTail; loopBnIdx++) {
            bIdx = (formerNum * bnRange + (coreIdx - formerNum) * bnRangeTail + loopBnIdx) / n;
            nIdx = (formerNum * bnRange + (coreIdx - formerNum) * bnRangeTail + loopBnIdx) % n;
            ComputeEachCoreImpl(bIdx, nIdx);
        }
    }
}

#endif  // ATTENTION_SCORE_SPLIT_B_N_H