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
 * \file scatter_list_rlbse_pad.h
 * \brief
 */
// 此文件实现所有芯片、-2轴更新、大batch小element、updates的后两维乘积不对齐block场景，对应模板211
// R: row, L: large, B: batch(updates的前两维除以核数得到), S: small, E: element(updates的后两维乘积), PAD: 填充
#ifndef SCATTER_LIST_RLBSE_PAD_H_
#define SCATTER_LIST_RLBSE_PAD_H_

#include "kernel_operator.h"
#include "scatter_list_base.h"

namespace ScatterList {
using namespace AscendC;

template <typename T1, typename T2>
class ScatterListRLBSEPad : public ScatterListBase<T1> {
public:
    __aicore__ inline ScatterListRLBSEPad(){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask, GM_ADDR varOut,
                                GM_ADDR workspace, const ScatterListTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessEqLen(const int64_t &eachCoreBatchNum);
    __aicore__ inline void ProcessNotEqLen(const int64_t &eachCoreBatchNum);
    __aicore__ inline void CopyIn(const int64_t &eachCoreBatchIdx);
    __aicore__ inline void CopyOutPad(const int64_t &copyCount, const int64_t &copyCountAlign,
                                      LocalTensor<T1> &updatesUb);
    __aicore__ inline void CopyOutEqLen(const int64_t &eachCoreBatchIdx);
    __aicore__ inline void CopyOutNotEqLen(const int64_t &eachCoreBatchIdx);

    constexpr static uint8_t bufferNum = 1;
    constexpr static uint8_t blockSize = 32;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> indiceInQueue;
    TQue<QuePosition::VECIN, bufferNum> updatesInQueue;
    TQue<QuePosition::VECIN, 1> maskInQueue;
    TBuf<QuePosition::VECCALC> tempBuf;

    GlobalTensor<T1> varGm;
    GlobalTensor<T2> indiceGm;
    GlobalTensor<T1> updatesGm;
    GlobalTensor<uint8_t> maskGm;

    int64_t blockIdx = 0;
    GM_ADDR varPtr = nullptr;
    bool maskIsNull = false;
    int64_t preCoreBatchIdx = 0;
    int64_t curCoreBatchIdx = 0;
    int64_t dim0Idx = 0;
    int64_t dim1Idx = 0;
    int64_t dim2OffsetIdx = 0;
    int64_t dim2UpdateLen = 0;
    int64_t dstGmOffset = 0;
    T1 tempVal = 0;
    int64_t copyCount = 0;
    int64_t copyCountAlign = 0;
    int64_t alignCount = 0;
    int64_t tailCount = 0;

    // tiling params
    ScatterListTilingData m_tilingData;
};

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBSEPad<T1, T2>::Init(GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask,
                                                         GM_ADDR varOut, GM_ADDR workspace,
                                                         const ScatterListTilingData* tilingData) {
    blockIdx = GetBlockIdx();
    varPtr = var;
    indiceGm.SetGlobalBuffer((__gm__ T2*)indice);
    updatesGm.SetGlobalBuffer((__gm__ T1*)updates);

    this->ParseTilingData(tilingData, m_tilingData);
    preCoreBatchIdx = blockIdx * m_tilingData.preCoreBatchNum;
    pipe.InitBuffer(indiceInQueue, 1, m_tilingData.indiceUbSize);
    pipe.InitBuffer(updatesInQueue, bufferNum, m_tilingData.updatesUbSize);
    pipe.InitBuffer(tempBuf, blockSize);

    if (mask == nullptr) {
        maskIsNull = true;
    } else {
        maskGm.SetGlobalBuffer((__gm__ uint8_t*)mask);
        pipe.InitBuffer(maskInQueue, 1, m_tilingData.maskUbSize);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBSEPad<T1, T2>::Process() {
    if (blockIdx >= m_tilingData.useCoreNum) {
        return;
    }
    if (blockIdx == m_tilingData.useCoreNum - 1) {
        if (m_tilingData.indiceDims == 1) {
            ProcessEqLen(m_tilingData.lastCoreBatchNum);
        } else {
            ProcessNotEqLen(m_tilingData.lastCoreBatchNum);
        }
    } else {
        if (m_tilingData.indiceDims == 1) {
            ProcessEqLen(m_tilingData.preCoreBatchNum);
        } else {
            ProcessNotEqLen(m_tilingData.preCoreBatchNum);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBSEPad<T1, T2>::ProcessEqLen(const int64_t &eachCoreBatchNum) {
    for (int64_t eachCoreBatchIdx = 0; eachCoreBatchIdx < eachCoreBatchNum; eachCoreBatchIdx++) {
        CopyIn(eachCoreBatchIdx);
        CopyOutEqLen(eachCoreBatchIdx);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBSEPad<T1, T2>::ProcessNotEqLen(const int64_t &eachCoreBatchNum) {
    for (int64_t eachCoreBatchIdx = 0; eachCoreBatchIdx < eachCoreBatchNum; eachCoreBatchIdx++) {
        CopyIn(eachCoreBatchIdx);
        CopyOutNotEqLen(eachCoreBatchIdx);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBSEPad<T1, T2>::CopyIn(const int64_t &eachCoreBatchIdx) {
    LocalTensor<T1> updatesUb = updatesInQueue.AllocTensor<T1>();
    DataCopy(updatesUb, updatesGm[(preCoreBatchIdx + eachCoreBatchIdx) * m_tilingData.srcBatchStride],
             m_tilingData.updatesCount);
    updatesInQueue.EnQue(updatesUb);

    LocalTensor<T2> indiceUb = indiceInQueue.AllocTensor<T2>();
    DataCopy(indiceUb, indiceGm, m_tilingData.indiceCount);
    indiceInQueue.EnQue(indiceUb);

    if (!maskIsNull) {
        LocalTensor<uint8_t> maskUb = maskInQueue.AllocTensor<uint8_t>();
        DataCopy(maskUb, maskGm, m_tilingData.maskCount);
        maskInQueue.EnQue(maskUb);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBSEPad<T1, T2>::CopyOutPad(const int64_t &copyCount, const int64_t &copyCountAlign,
                                                               LocalTensor<T1> &updatesUb) {
    if constexpr (PlatformSocInfo::IsDataCopyPadSupport()) {
        if (copyCountAlign > m_tilingData.updatesOneBlock) {
            alignCount = copyCountAlign - m_tilingData.updatesOneBlock;
            tailCount = copyCount - alignCount;
            DataCopy(varGm[dstGmOffset], updatesUb, alignCount);
            DataCopyParams copyParams{1, static_cast<uint16_t>(tailCount * sizeof(T1)), 0, 0};
            DataCopyPad(varGm[dstGmOffset + alignCount], updatesUb[alignCount], copyParams);
        } else {
            DataCopyParams copyParams{1, static_cast<uint16_t>(copyCount * sizeof(T1)), 0, 0};
            DataCopyPad(varGm[dstGmOffset], updatesUb, copyParams);
        }
    } else {
        LocalTensor<T1> tempUb = tempBuf.Get<T1>();
        if (copyCountAlign > m_tilingData.updatesOneBlock) {
            alignCount = copyCountAlign - m_tilingData.updatesOneBlock;
            tailCount = copyCount - m_tilingData.updatesOneBlock;
            DataCopy(varGm[dstGmOffset], updatesUb, alignCount);
            for (int64_t i = 0; i < m_tilingData.updatesOneBlock; i++) {
                tempVal = updatesUb.GetValue(tailCount + i);
                tempUb.SetValue(i, tempVal);
            }
            DataCopy(varGm[dstGmOffset + tailCount], tempUb, m_tilingData.updatesOneBlock);
        } else {
            DataCopy(tempUb, varGm[dstGmOffset], m_tilingData.updatesOneBlock);
            this->Mte2ToS();
            for (int64_t i = 0; i < copyCount; i++) {
                tempVal = updatesUb.GetValue(i);
                tempUb.SetValue(i, tempVal);
            }
            DataCopy(varGm[dstGmOffset], tempUb, m_tilingData.updatesOneBlock);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBSEPad<T1, T2>::CopyOutEqLen(const int64_t &eachCoreBatchIdx) {
    LocalTensor<T1> updatesUb = updatesInQueue.DeQue<T1>();
    LocalTensor<T2> indiceUb = indiceInQueue.DeQue<T2>();
    LocalTensor<uint8_t> maskUb;
    if (!maskIsNull) {
        maskUb = maskInQueue.DeQue<uint8_t>();
    }
    this->Mte2ToS();
    this->Mte2ToMte3();

    curCoreBatchIdx = preCoreBatchIdx + eachCoreBatchIdx;
    dim0Idx = curCoreBatchIdx / m_tilingData.dim1Count;
    if (maskIsNull || maskUb.GetValue(dim0Idx) == 1) {
        varGm.SetGlobalBuffer(this->GetTensorAddr(varPtr, dim0Idx));
        dim1Idx = curCoreBatchIdx % m_tilingData.dim1Count;
        dim2OffsetIdx = indiceUb.GetValue(dim0Idx);
        dstGmOffset = dim1Idx * m_tilingData.dstBatchStride + dim2OffsetIdx * m_tilingData.dim3Count;
        if (m_tilingData.srcBatchStride == m_tilingData.srcBatchStrideAlign) {
            DataCopy(varGm[dstGmOffset], updatesUb, m_tilingData.srcBatchStride);
        } else {
            CopyOutPad(m_tilingData.srcBatchStride, m_tilingData.srcBatchStrideAlign, updatesUb);
        }
        this->Mte3ToMte2();
    }

    indiceInQueue.FreeTensor(indiceUb);
    updatesInQueue.FreeTensor(updatesUb);
    if (!maskIsNull) {
        maskInQueue.FreeTensor(maskUb);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ScatterListRLBSEPad<T1, T2>::CopyOutNotEqLen(const int64_t &eachCoreBatchIdx) {
    LocalTensor<T1> updatesUb = updatesInQueue.DeQue<T1>();
    LocalTensor<T2> indiceUb = indiceInQueue.DeQue<T2>();
    LocalTensor<uint8_t> maskUb;
    if (!maskIsNull) {
        maskUb = maskInQueue.DeQue<uint8_t>();
    }
    this->Mte2ToS();
    this->Mte2ToMte3();

    curCoreBatchIdx = preCoreBatchIdx + eachCoreBatchIdx;
    dim0Idx = curCoreBatchIdx / m_tilingData.dim1Count;
    if (maskIsNull || maskUb.GetValue(dim0Idx) == 1) {
        varGm.SetGlobalBuffer(this->GetTensorAddr(varPtr, dim0Idx));
        dim1Idx = curCoreBatchIdx % m_tilingData.dim1Count;
        dim2OffsetIdx = indiceUb.GetValue(dim0Idx * 2);
        dim2UpdateLen = indiceUb.GetValue(dim0Idx * 2 + 1);
        dstGmOffset = dim1Idx * m_tilingData.dstBatchStride + dim2OffsetIdx * m_tilingData.dim3Count;
        copyCount = dim2UpdateLen * m_tilingData.dim3Count;
        copyCountAlign = this->CeilDivMul(copyCount, m_tilingData.updatesOneBlock);
        if (copyCount == copyCountAlign) {
            DataCopy(varGm[dstGmOffset], updatesUb, copyCount);
        } else {
            CopyOutPad(copyCount, copyCountAlign, updatesUb);
        }
        this->Mte3ToMte2();
    }

    indiceInQueue.FreeTensor(indiceUb);
    updatesInQueue.FreeTensor(updatesUb);
    if (!maskIsNull) {
        maskInQueue.FreeTensor(maskUb);
    }
}

}  // namespace ScatterList

#endif  // SCATTER_LIST_RLBSE_PAD_H_
