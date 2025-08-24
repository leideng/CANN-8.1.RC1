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
 * \file index_put_with_sort_accumulate.h
 * \brief
 */

#ifndef INDEX_PUT_WITH_SORT_ACCUMULATE_H
#define INDEX_PUT_WITH_SORT_ACCUMULATE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AscendC {
template<typename T>
class IndexPutWithSortAccumulateKernel {
public:
    __aicore__ inline IndexPutWithSortAccumulateKernel() = delete;
    __aicore__ inline IndexPutWithSortAccumulateKernel(GM_ADDR self, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR values,
                                                       GM_ADDR output, GM_ADDR workSpace,
                                                       const IndexPutWithSortTilingData &tiling, TPipe &pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(self, sortIndices, posIdx, values, output, workSpace, tiling);
    }
    __aicore__ inline void Process()
    {
        for (uint32_t repeatTimeHuge = 0; repeatTimeHuge <= sliceRepeatTime_; repeatTimeHuge++) {
            sliceSize_ = (repeatTimeHuge == sliceRepeatTime_) ? sliceLeft_ : sliceSizeLimit_;
            InitTmpSum();
            for (uint32_t i = 0; i < groupNum_; i++) {
                uint32_t offset = i * availableNumOri_;
                if (i == groupNum_ - 1) {
                    availableNum_ = availableLeft_;
                }
                else {
                    availableNum_ = availableNumOri_;
                }
                CopyInIndices(offset);
                for (uint32_t j = 0; j < availableNum_; j++) {
                    CopyInValues(j, repeatTimeHuge);
                    ComputeAndCopyOut(j, repeatTimeHuge);
                }
                FreeIdx();
            }
            FreeTmp();
        }
    }

private:
    __aicore__ inline void InitParams(const IndexPutWithSortTilingData &tiling)
    {
        coreNum_ = tiling.params.coreNum;
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ == coreNum_ - 1) {
            rowNum_ = tiling.params.numLastCore;
        } else {
            rowNum_ = tiling.params.numEachCore;
        }
        sliceSizeOri_ = tiling.params.sliceSize;
        numel_ = tiling.params.numel;
        taskNum_ = tiling.params.taskNum;
        numEachCore_ = tiling.params.numEachCore;
        sliceSizeLimit_ = tiling.params.sliceSizeLimit;
        sliceSize_ = sliceSizeOri_ <= sliceSizeLimit_ ? sliceSizeOri_ : sliceSizeLimit_;
        sliceRepeatTime_ = tiling.params.sliceRepeatTime;
        sliceLeft_ = tiling.params.sliceLeft;
        availableNumOri_ = tiling.params.availableNum;
        if (availableNumOri_ < 1) {
            availableNumOri_ = 1;
        }
        availableNum_ = availableNumOri_;
        groupNum_ = (rowNum_ - 1) / availableNum_ + 1;
        availableLeft_ = rowNum_ % availableNum_ == 0 ? availableNum_ : rowNum_ % availableNum_;
    }
    __aicore__ inline void InitBuffers(TPipe &pipe)
    {
        uint32_t idxAlignNum = BLOCK_SIZE / sizeof(int);
        uint32_t alignNumT = BLOCK_SIZE / sizeof(T);
        uint32_t valuesAlignNum = ((sliceSize_ + alignNumT - 1) / alignNumT) * alignNumT;
        idxAlignNum = ((availableNum_ + idxAlignNum - 1) / idxAlignNum) * idxAlignNum;
        pipe.InitBuffer(indicesQue_, BUFFER_NUM, idxAlignNum * sizeof(int));
        pipe.InitBuffer(posIdxQue_, BUFFER_NUM, idxAlignNum * sizeof(int));
        pipe.InitBuffer(valuesQue_, BUFFER_NUM, valuesAlignNum * sizeof(T));
        pipe.InitBuffer(tmpQue_, BUFFER_NUM, valuesAlignNum * sizeof(T));
    }

    __aicore__ inline void SetGmAddr(GM_ADDR self, GM_ADDR sortIndices, GM_ADDR posIdx,
                                     GM_ADDR values, GM_ADDR output,
                                     GM_ADDR workSpace, const IndexPutWithSortTilingData &tiling)
    {
        indicesGm_.SetGlobalBuffer((__gm__ uint32_t*)sortIndices);
        posIdxGm_.SetGlobalBuffer((__gm__ uint32_t*)posIdx);
        valuesGm_.SetGlobalBuffer((__gm__ T*)values);
        outputGm_.SetGlobalBuffer((__gm__ T*)self);
    }

    __aicore__ inline void InitTmpSum()
    {
        LocalTensor<T> tmpLocal = tmpQue_.AllocTensor<T>();
        Duplicate<T>(tmpLocal, 0.0, sliceSize_);
        tmpQue_.EnQue<T>(tmpLocal);
    }

    __aicore__ inline void CopyInIndices(const uint32_t offset)
    {
        uint32_t indiceAddrOffset = numEachCore_ * blockIdx_ + offset;
        LocalTensor<uint32_t> indicesLocal = indicesQue_.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> posIdxLocal = posIdxQue_.AllocTensor<uint32_t>();
        DataCopyExtParams indicesCopyParams{1, static_cast<uint32_t>(sizeof(uint32_t) * availableNum_), 0, 0, 0};
        DataCopyPadExtParams<uint32_t> padParams{true, 0, 0, 0};
        DataCopyPad(posIdxLocal, posIdxGm_[indiceAddrOffset], indicesCopyParams, padParams);
        DataCopyPad(indicesLocal, indicesGm_[indiceAddrOffset], indicesCopyParams, padParams);
        indicesQue_.EnQue<uint32_t>(indicesLocal);
        posIdxQue_.EnQue<uint32_t>(posIdxLocal);
    }

    __aicore__ inline void CopyInValues(const uint32_t progress, const uint32_t repeatTimeHuge)
    {
        LocalTensor<uint32_t> posIdxLocal = posIdxQue_.DeQue<uint32_t>();
        LocalTensor<T> valuesLocal = valuesQue_.AllocTensor<T>();
        if (progress != availableNum_ - 1) {
            lastIdx_ = false;
        }
        else {
            lastIdx_ = true;
        }
        DataCopyExtParams valuesCopyParams{1, static_cast<uint32_t>(sliceSize_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParamsValues{true, 0, 0, 0};
        int32_t valuesAddrOffset = posIdxLocal.GetValue(progress) * sliceSizeOri_ + repeatTimeHuge * sliceSizeLimit_;
        DataCopyPad(valuesLocal, valuesGm_[valuesAddrOffset], valuesCopyParams, padParamsValues);
        valuesQue_.EnQue<T>(valuesLocal);
    }

    __aicore__ inline void ComputeAndCopyOut(const uint32_t progress, const uint32_t repeatTimeHuge)
    {
        AtomicAddToOut(progress, repeatTimeHuge);
        LocalTensor<T> valuesLocal = valuesQue_.DeQue<T>();
        valuesQue_.FreeTensor<T>(valuesLocal);
    }

    __aicore__ inline void AtomicAddToOut(const uint32_t progress, const uint32_t repeatTimeHuge)
    {
        event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        LocalTensor<uint32_t> indicesLocal = indicesQue_.DeQue<uint32_t>();
        LocalTensor<T> valuesLocal = valuesQue_.DeQue<T>();
        LocalTensor<T> tmpLocal = tmpQue_.DeQue<T>();
        int32_t indicesNow = indicesLocal.GetValue(progress);
        int32_t tmpAddrOffset = indicesNow + repeatTimeHuge * sliceSizeLimit_;
        int32_t nextIdx = indicesNow;
        if (!lastIdx_) {
            nextIdx = indicesLocal.GetValue(progress + 1);
        }
        Add(tmpLocal, tmpLocal, valuesLocal, sliceSize_);
        ++repeatTimes_;
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        if (repeatTimes_ > LIMIT_COUNT_NUM || lastIdx_ || nextIdx != indicesNow) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(sliceSize_ * sizeof(T)), 0, 0, 0};
            SetAtomicAdd<T>();
            DataCopyPad(outputGm_[tmpAddrOffset], tmpLocal, copyParams);
            SetAtomicNone();
            SetFlag<HardEvent::MTE3_V>(eventMte3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventMte3ToV);
            Duplicate<T>(tmpLocal, 0.0, sliceSize_);
            repeatTimes_ = 0;
        }
        tmpQue_.EnQue<T>(tmpLocal);
    }

    __aicore__ inline void FreeIdx()
    {
        LocalTensor<uint32_t> indicesLocal = indicesQue_.DeQue<uint32_t>();
        LocalTensor<uint32_t> posIdxLocal = posIdxQue_.DeQue<uint32_t>();
        indicesQue_.FreeTensor<uint32_t>(indicesLocal);
        posIdxQue_.FreeTensor<uint32_t>(posIdxLocal);
    }

    __aicore__ inline void FreeTmp()
    {
        LocalTensor<T> tmpLocal = tmpQue_.DeQue<T>();
        tmpQue_.FreeTensor<T>(tmpLocal);
    }

private:
    GlobalTensor<T> outputGm_;
    GlobalTensor<uint32_t> indicesGm_;
    GlobalTensor<uint32_t> posIdxGm_;
    GlobalTensor<T> valuesGm_;

    TQue<TPosition::VECIN, BUFFER_NUM> indicesQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> posIdxQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> valuesQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> tmpQue_;

    uint32_t numel_;
    uint32_t rowNum_;
    uint32_t repeatTimes_ = 0;
    uint32_t coreNum_;
    uint32_t taskNum_;
    uint32_t numEachCore_;
    uint32_t sliceSizeLimit_;
    uint32_t sliceRepeatTime_;
    uint32_t sliceLeft_;
    uint32_t availableNum_;
    uint32_t availableNumOri_;
    uint32_t groupNum_;
    uint32_t availableLeft_;

    uint32_t blockIdx_;
    // attr
    uint32_t sliceSizeOri_;
    uint32_t sliceSize_;
    bool lastIdx_ = false;
};
}

#endif // INDEX_PUT_WITH_SORT_ACCUMULATE_H