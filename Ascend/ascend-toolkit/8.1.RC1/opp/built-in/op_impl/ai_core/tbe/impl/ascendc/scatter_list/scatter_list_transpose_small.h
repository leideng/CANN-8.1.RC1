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
 * \file scatter_list_transpose_small.h
 * \brief
 */
#ifndef SCATTER_LIST_TRANSPOSE_SMALL_H_
#define SCATTER_LIST_TRANSPOSE_SMALL_H_

#include "kernel_operator.h"
#include "scatter_list_base.h"

namespace ScatterList {
using namespace AscendC;

template <typename T1, typename T2>
class ScatterListTransposeSmall : public ScatterListBase<T1> {
public:
    __aicore__ inline ScatterListTransposeSmall(){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask, GM_ADDR tempOut,
                                GM_ADDR workspace, const ScatterListTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(const uint64_t &CoreBatchNum);
    __aicore__ inline void CopyOut(const uint64_t &CoreBatchNum);
    __aicore__ inline void CopyOutTransSmall(LocalTensor<T1> &dataUbSize, LocalTensor<T1> &transposeUbSize,
                                             LocalTensor<T1> &updatesUb, const uint64_t &srcGmOffset,
                                             const uint64_t &dstGmOffset);
    __aicore__ inline void TransposeB2(LocalTensor<T1> &dataUbSize, LocalTensor<T1> &transposeUbSize,
                                       LocalTensor<T1> &updatesUb, const uint64_t &srcGmOffset,
                                       const uint64_t &dstGmOffset);
    __aicore__ inline void TransposeB4(LocalTensor<T1> &dataUbSize, LocalTensor<T1> &transposeUbSize,
                                       LocalTensor<T1> &updatesUb, const uint64_t &srcGmOffset,
                                       const uint64_t &dstGmOffset);
    __aicore__ inline void TransposeB1(LocalTensor<T1> &dataUbSize, LocalTensor<T1> &transposeUbSize,
                                       LocalTensor<T1> &updatesUb, const uint64_t &srcGmOffset,
                                       const uint64_t &dstGmOffset);
    constexpr static int32_t bufferNum = 1;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> indiceInQueue;
    TQue<QuePosition::VECIN, bufferNum> updatesInQueue;
    TQue<QuePosition::VECIN, bufferNum> maskInQueue;

    GlobalTensor<T1> varGm;
    GlobalTensor<T2> indiceGm;
    GlobalTensor<T1> updatesGm;
    GlobalTensor<uint8_t> maskGm;
    TBuf<QuePosition::VECCALC> dataUb;
    TBuf<QuePosition::VECCALC> transposeUb;
    uint64_t blockIdx = 0;
    GM_ADDR varPtr = nullptr;
    bool maskIsNull = false;
    ScatterListTilingData m_tilingData;
};

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeSmall<T1, T2>::Init(GM_ADDR var, GM_ADDR indice, GM_ADDR updates,
                                                               GM_ADDR mask, GM_ADDR tempOut, GM_ADDR workspace,
                                                               const ScatterListTilingData* tilingData) {
    blockIdx = GetBlockIdx();
    this->ParseTilingData(tilingData, m_tilingData);
    varPtr = var;
    indiceGm.SetGlobalBuffer((__gm__ T2*)indice);
    updatesGm.SetGlobalBuffer((__gm__ T1*)updates);
    if (mask == nullptr) {
        maskIsNull = true;
    } else {
        maskGm.SetGlobalBuffer((__gm__ uint8_t*)mask);
        pipe.InitBuffer(maskInQueue, bufferNum, m_tilingData.maskUbSize);
    }

    pipe.InitBuffer(indiceInQueue, bufferNum, m_tilingData.indiceUbSize);
    pipe.InitBuffer(updatesInQueue, bufferNum, m_tilingData.updatesUbSize);
    pipe.InitBuffer(dataUb, m_tilingData.dataUbSize);
    pipe.InitBuffer(transposeUb, m_tilingData.transposeUbSize);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeSmall<T1, T2>::Process() {
    if (blockIdx >= m_tilingData.useCoreNum) {
        return;
    }
    if (blockIdx == m_tilingData.useCoreNum - 1) {
        CopyIn(m_tilingData.lastCoreBatchNum);
        CopyOut(m_tilingData.lastCoreBatchNum);
    } else {
        CopyIn(m_tilingData.preCoreBatchNum);
        CopyOut(m_tilingData.preCoreBatchNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeSmall<T1, T2>::CopyIn(const uint64_t &CoreBatchNum) {
    LocalTensor<T2> indiceUb = indiceInQueue.AllocTensor<T2>();
    LocalTensor<T1> updatesUb = updatesInQueue.AllocTensor<T1>();
    DataCopy(indiceUb, indiceGm, m_tilingData.indiceCount);
    DataCopy(updatesUb, updatesGm[blockIdx * m_tilingData.preCoreUpdateDim23],
             CoreBatchNum*m_tilingData.srcBatchStride);
    if (!maskIsNull) {
        LocalTensor<uint8_t> maskUb = maskInQueue.AllocTensor<uint8_t>();
        DataCopy(maskUb, maskGm, m_tilingData.maskCount);
        maskInQueue.EnQue(maskUb);
    }
    indiceInQueue.EnQue(indiceUb);
    updatesInQueue.EnQue(updatesUb);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeSmall<T1, T2>::CopyOutTransSmall(LocalTensor<T1> &dataUbSize,
                                                                            LocalTensor<T1> &transposeUbSize,
                                                                            LocalTensor<T1> &updatesUb,
                                                                            const uint64_t &srcGmOffset,
                                                                            const uint64_t &dstGmOffset) {
    if (sizeof(T1) == 2) {
        TransposeB2(dataUbSize, transposeUbSize, updatesUb, srcGmOffset, dstGmOffset);
    } else if (sizeof(T1) == 4) {
        TransposeB4(dataUbSize, transposeUbSize, updatesUb, srcGmOffset, dstGmOffset);
    } else if (sizeof(T1) == 1) {
        TransposeB1(dataUbSize, transposeUbSize, updatesUb, srcGmOffset, dstGmOffset);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeSmall<T1, T2>::TransposeB2(LocalTensor<T1> &dataUbSize,
                                                                      LocalTensor<T1> &transposeUbSize,
                                                                      LocalTensor<T1> &updatesUb,
                                                                      const uint64_t &srcGmOffset,
                                                                      const uint64_t &dstGmOffset) {
    TransposeParams params;
    TransposeParams paramsBack;
    params.src = 16;
    params.dst = m_tilingData.eachLastSize;
    params.src_rep = 16;
    params.dst_rep = 1;
    params.repeat_times = m_tilingData.transRepeatTimes;
    paramsBack.src = m_tilingData.eachLastSize;
    paramsBack.dst = 16;
    paramsBack.src_rep = 1;
    paramsBack.dst_rep = 16;
    paramsBack.repeat_times = m_tilingData.transRepeatTimes;
    if (params.repeat_times == 1) {
        params.src_rep = 0;
        params.dst_rep = 0;
        paramsBack.src_rep = 0;
        paramsBack.dst_rep = 0;
    }
    this->TransposeB16(params, dataUbSize, transposeUbSize);
    DataCopy(transposeUbSize, updatesUb[srcGmOffset], m_tilingData.srcBatchStride);
    this->TransposeB16(paramsBack, transposeUbSize, dataUbSize);
    DataCopyParams copyParams;
    copyParams.blockCount = m_tilingData.varDim2Count;
    copyParams.blockLen = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = m_tilingData.varDim3Stride;
    this->VToMte3();
    DataCopy(varGm[dstGmOffset], dataUbSize, copyParams);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeSmall<T1, T2>::TransposeB4(LocalTensor<T1> &dataUbSize,
                                                                      LocalTensor<T1> &transposeUbSize,
                                                                      LocalTensor<T1> &updatesUb,
                                                                      const uint64_t &srcGmOffset,
                                                                      const uint64_t &dstGmOffset) {
    TransposeParams params;
    TransposeParams paramsBack;
    params.src = 8;
    params.dst = m_tilingData.eachLastSize;
    params.src_rep = 16;
    params.dst_rep = 2;
    params.repeat_times = m_tilingData.transRepeatTimes;

    paramsBack.src = m_tilingData.eachLastSize;
    paramsBack.dst = 8;
    paramsBack.src_rep = 2;
    paramsBack.dst_rep = 16;
    paramsBack.repeat_times = m_tilingData.transRepeatTimes;
    if (params.repeat_times == 1) {
        params.src_rep = 0;
        params.dst_rep = 0;
        paramsBack.src_rep = 0;
        paramsBack.dst_rep = 0;
    }
    this->TransposeB32(params, dataUbSize, transposeUbSize);
    DataCopy(transposeUbSize, updatesUb[srcGmOffset], m_tilingData.srcBatchStride);
    this->TransposeB32Back(paramsBack, transposeUbSize, dataUbSize);
    DataCopyParams copyParams;
    copyParams.blockCount = m_tilingData.varDim2Count;
    copyParams.blockLen = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = m_tilingData.varDim3Stride;
    this->VToMte3();
    DataCopy(varGm[dstGmOffset], dataUbSize, copyParams);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeSmall<T1, T2>::TransposeB1(LocalTensor<T1> &dataUbSize,
                                                                      LocalTensor<T1> &transposeUbSize,
                                                                      LocalTensor<T1> &updatesUb,
                                                                      const uint64_t &srcGmOffset,
                                                                      const uint64_t &dstGmOffset) {
    TransposeParams params;
    TransposeParams paramsBack;
    params.src = 32;
    params.dst = m_tilingData.eachLastSize;
    params.src_rep = 32;
    params.dst_rep = 1;
    params.repeat_times = m_tilingData.transRepeatTimes;

    paramsBack.src = m_tilingData.eachLastSize;
    paramsBack.dst = 32;
    paramsBack.src_rep = 1;
    paramsBack.dst_rep = 32;
    paramsBack.repeat_times = m_tilingData.transRepeatTimes;
    if (params.repeat_times == 1) {
        params.src_rep = 0;
        params.dst_rep = 0;
        paramsBack.src_rep = 0;
        paramsBack.dst_rep = 0;
    }
    this->TransposeB8(params, dataUbSize, transposeUbSize);
    DataCopy(transposeUbSize, updatesUb[srcGmOffset], m_tilingData.srcBatchStride);
    this->TransposeB8(paramsBack, transposeUbSize, dataUbSize);
    DataCopyParams copyParams;
    copyParams.blockCount = m_tilingData.varDim2Count;
    copyParams.blockLen = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = m_tilingData.varDim3Stride;
    this->VToMte3();
    DataCopy(varGm[dstGmOffset], dataUbSize, copyParams);
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListTransposeSmall<T1, T2>::CopyOut(const uint64_t &CoreBatchNum) {
    LocalTensor<T2> indiceUb = indiceInQueue.DeQue<T2>();
    LocalTensor<T1> updatesUb = updatesInQueue.DeQue<T1>();

    LocalTensor<T1> dataUbSize = dataUb.Get<T1>();
    LocalTensor<T1> transposeUbSize = transposeUb.Get<T1>();
    LocalTensor<uint8_t> maskUb;
    if (!maskIsNull) {
        maskUb = maskInQueue.DeQue<uint8_t>();
    }

    this->Mte2ToS();

    for (uint64_t eachCoreBatchIdx = 0; eachCoreBatchIdx < CoreBatchNum; eachCoreBatchIdx++) {
        uint64_t allCoreBatchIdx = blockIdx * m_tilingData.preCoreBatchNum + eachCoreBatchIdx;
        uint64_t dim0Idx = allCoreBatchIdx / m_tilingData.dim1Count;
        if ((!maskIsNull) && (maskUb.GetValue(dim0Idx) == 0)) {
            continue;
        }
        varGm.SetGlobalBuffer(this->GetTensorAddr(varPtr, dim0Idx));
        uint64_t dim1Idx = allCoreBatchIdx % m_tilingData.dim1Count;
        uint64_t dim2OffsetIdx = indiceUb.GetValue(dim0Idx);
        uint64_t dstGmOffset = dim1Idx * m_tilingData.dstBatchStride + dim2OffsetIdx;
        uint64_t srcGmOffset = eachCoreBatchIdx * m_tilingData.srcBatchStride;

        DataCopyParams copyParams;
        copyParams.blockCount = m_tilingData.varDim2Count;
        copyParams.blockLen = 1;
        copyParams.srcStride = m_tilingData.varDim3Stride;
        copyParams.dstStride = 0;
        if (eachCoreBatchIdx > 0) {
            this->Mte3ToMte2();
        }
        DataCopy(dataUbSize, varGm[dstGmOffset], copyParams);
        this->Mte2ToV();
        CopyOutTransSmall(dataUbSize, transposeUbSize, updatesUb, srcGmOffset, dstGmOffset);
    }
    indiceInQueue.FreeTensor(indiceUb);
    updatesInQueue.FreeTensor(updatesUb);
    if (!maskIsNull) {
        maskInQueue.FreeTensor(maskUb);
    }
}
}  // namespace ScatterList

#endif  // SCATTER_LIST_TRANSPOSE_SMALL_H_
