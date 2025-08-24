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
 * \file pad_infer.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "utils.h"
using namespace AscendC;

namespace{
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t ELE_PER_BLK = 16;
constexpr int32_t MAX_BATCH_NUM = 64;

class PadInfer {
public:
    __aicore__ inline PadInfer() {}
    __aicore__ inline void Init(GM_ADDR tmpOut, 
                                GM_ADDR paddingOffset, 
                                GM_ADDR seqLen, 
                                GM_ADDR inputIds, 
                                GM_ADDR out, 
                                uint32_t padLength, 
                                uint32_t batch, 
                                uint32_t hiddenDim)
    {
        padLength_ = padLength;
        batch_ = batch;
        hiddenDim_ = hiddenDim;
        hiddenDimAlign_ = (hiddenDim + ELE_PER_BLK - 1) / ELE_PER_BLK * ELE_PER_BLK;
        padLengthAlign_ = ((padLength_ + ELE_PER_BLK - 1) / ELE_PER_BLK) * ELE_PER_BLK;
        batchAlign_ = ((batch_ + ELE_PER_BLK - 1) / ELE_PER_BLK) * ELE_PER_BLK;
        uint32_t tokenNum  = 0;
        for (uint32_t i = 0; i < batch; i++) {
            tokenNum += *((__gm__ int32_t *)(seqLen) + i);
        }
        tmpOutGm.SetGlobalBuffer((__gm__ half *)tmpOut, tokenNum * hiddenDim_);
        seqLenGm.SetGlobalBuffer((__gm__ int32_t *)seqLen, tokenNum);
        outGm.SetGlobalBuffer((__gm__ half *)out, batch_ * hiddenDim_);
        pipe_.InitBuffer(tmpOutQueue_, BUFFER_NUM, hiddenDimAlign_ * sizeof(half));
        pipe_.InitBuffer(seqLenQueue_, BUFFER_NUM, MAX_BATCH_NUM * sizeof(int32_t));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, hiddenDimAlign_ * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < batch_; i++) {
            CopyOnce();
            AscendC::PipeBarrier<PIPE_ALL>();
            CopyIn(i);
            AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            ComputeRemovePadding();
            AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyOnce()
    {
        LocalTensor<int32_t> seqLenLocal = seqLenQueue_.AllocTensor<int32_t>();
        DataCopy(seqLenLocal, seqLenGm, MAX_BATCH_NUM);
        seqLenQueue_.EnQue(seqLenLocal);
    }
    __aicore__ inline void CopyIn(uint64_t progress)
    {
        LocalTensor<int32_t> seqLenLocal = seqLenQueue_.DeQue<int32_t>();
        LocalTensor<half> tmpOutLocal = tmpOutQueue_.AllocTensor<half>();
        if (seqLenLocal.GetValue(progress) == 0) {
            offsetUnusedBathch++;
        } else {
            tempVal_ = tempVal_ + seqLenLocal.GetValue(progress);
            AscendC::PipeBarrier<PIPE_ALL>();
            DataCopy(tmpOutLocal, tmpOutGm[(tempVal_ - 1) * hiddenDim_], hiddenDimAlign_);
        }
        tmpOutQueue_.EnQue(tmpOutLocal);
        seqLenQueue_.FreeTensor(seqLenLocal);
    }

    __aicore__ inline void ComputeRemovePadding()
    {
        LocalTensor<half> tmpOutLocal = tmpOutQueue_.DeQue<half>();
        LocalTensor<half> outLocal = outQueue_.AllocTensor<half>();
        DataCopy(outLocal, tmpOutLocal, hiddenDimAlign_);
        tmpOutQueue_.FreeTensor(tmpOutLocal);
        outQueue_.EnQue(outLocal);
    }
    __aicore__ inline void CopyOut(uint64_t progress)
    {
        LocalTensor<half> outLocal = outQueue_.DeQue<half>();
        DataCopy(outGm[(progress - offsetUnusedBathch) * hiddenDim_], outLocal, hiddenDimAlign_);
        outQueue_.FreeTensor(outLocal);
    }
private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> tmpOutQueue_, paddingOffsetQueue_, seqLenQueue_, inputIdsQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    GlobalTensor<half> tmpOutGm, outGm;
    GlobalTensor<int32_t> paddingOffsetGm, seqLenGm, inputIdsGm;
    uint32_t padLength_{1};
    uint32_t batch_{1};
    uint32_t padLengthAlign_{16};
    uint32_t batchAlign_{8};
    uint32_t hiddenDim_{16};
    uint32_t hiddenDimAlign_{16};
    uint32_t tempVal_ = 0;
    uint64_t offsetUnusedBathch = 0;
};
}

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, PadInferTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->padLength = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->batch = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->hiddenDim = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(PadInferTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->padLength = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->batch = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->hiddenDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg)    \
    PadInferTilingData tiling_data;      \
    InitTilingData(tiling_arg, &(tiling_data))

extern "C" __global__ __aicore__ void pad_infer(GM_ADDR tmpOut,
                                                GM_ADDR paddingOffset, 
                                                GM_ADDR seqLen,
                                                GM_ADDR inputIds, 
                                                GM_ADDR out,
                                                GM_ADDR workspace, 
                                                GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    PadInfer op;
    op.Init(tmpOut, paddingOffset, seqLen, inputIds,
            out, tilingData.padLength, tilingData.batch, tilingData.hiddenDim);
    op.Process();
}