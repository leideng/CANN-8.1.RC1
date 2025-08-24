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
 * \file index_put_with_sort_determinist.h
 * \brief
 */

#ifndef INDEX_PUT_WITH_SORT_DETERMINIST_H
#define INDEX_PUT_WITH_SORT_DETERMINIST_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

constexpr int32_t INDICE_INIT_PARAM = -1;
constexpr int32_t ADDR_BACK_STEP = 1;

namespace AscendC {
template<typename T>
class IndexPutWithSortDeterministKernel {
public:
    __aicore__ inline IndexPutWithSortDeterministKernel() = delete;
    __aicore__ inline IndexPutWithSortDeterministKernel(GM_ADDR self, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR values,
                                                        GM_ADDR output, GM_ADDR workSpace,
                                                        const IndexPutWithSortTilingData &tiling, TPipe &pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(self, sortIndices, posIdx, values, output, workSpace, tiling);
    }
    __aicore__ inline void Process()
    {
        InitTmpSum();
        for (uint32_t repeatTimeHuge = 0; repeatTimeHuge <= sliceRepeatTime_; repeatTimeHuge++) {
            standIdx_ = -1;
            bool isNeedContinueCopy = true;
            sliceSize_ = (repeatTimeHuge == sliceRepeatTime_) ? sliceLeft_ : sliceSizeLimit_;
            if (blockIdx_ != 0) {
                InitStandIndices(0);
            }
            for (int32_t i = rowNum_ - 1; i >= 0; i--) {
                if (!(CopyIn(i, true, repeatTimeHuge))) {
                    isNeedContinueCopy = i == rowNum_ - 1 ? false : true;
                    if (lastIdx_ != INDICE_INIT_PARAM) {
                        CopyOut(switchId_, lastIdx_, repeatTimeHuge);
                    }
                    break;
                }
                ComputeAndCopyOut(i, 0, repeatTimeHuge);
            }
            if (isNeedContinueCopy) {
                ContinueCompute(repeatTimeHuge);
            }
        }
        FreeTmp();
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
        numEachCore_ = tiling.params.numEachCore;
        sliceSizeOri_ = tiling.params.sliceSize;
        numel_ = tiling.params.numel;
        taskNum_ = tiling.params.taskNum;
        lastIdx_ = INDICE_INIT_PARAM;
        sliceSizeLimit_ = tiling.params.sliceSizeLimit;
        sliceSize_ = sliceSizeOri_ <= sliceSizeLimit_ ? sliceSizeOri_ : sliceSizeLimit_;
        sliceRepeatTime_ = tiling.params.sliceRepeatTime;
        sliceLeft_ = tiling.params.sliceLeft;
        repeatTimes_[0] = INIT_PARAM;
        repeatTimes_[1] = INIT_PARAM;
    }
    __aicore__ inline void InitBuffers(TPipe &pipe)
    {
        uint32_t idxAlignNum = BLOCK_SIZE / sizeof(int);
        uint32_t alignNumT = BLOCK_SIZE / sizeof(T);
        uint32_t valuesAlignNum = ((sliceSize_ + alignNumT - 1) / alignNumT) * alignNumT;
        pipe.InitBuffer(indicesQue_, BUFFER_NUM, idxAlignNum * sizeof(int));
        pipe.InitBuffer(posIdxQue_, BUFFER_NUM, idxAlignNum * sizeof(int));
        pipe.InitBuffer(tmpQue_[0], BUFFER_NUM, valuesAlignNum * sizeof(T));
        pipe.InitBuffer(tmpQue_[1], BUFFER_NUM, valuesAlignNum * sizeof(T));
        pipe.InitBuffer(valuesQue_, BUFFER_NUM, valuesAlignNum * sizeof(T));
    }

    __aicore__ inline void SetGmAddr(GM_ADDR self, GM_ADDR sortIndices, GM_ADDR posIdx,
                                     GM_ADDR values, GM_ADDR output,
                                     GM_ADDR workSpace, const IndexPutWithSortTilingData &tiling)
    {
        uint32_t indiceAddrOffset = numEachCore_ * blockIdx_;
        indicesGm_.SetGlobalBuffer((__gm__ uint32_t*)sortIndices + indiceAddrOffset);
        indicesGlobalGm_.SetGlobalBuffer((__gm__ uint32_t*)sortIndices);
        posIdxGm_.SetGlobalBuffer((__gm__ uint32_t*)posIdx + indiceAddrOffset);
        valuesGm_.SetGlobalBuffer((__gm__ T*)values);
        outputGm_.SetGlobalBuffer((__gm__ T*)self);
    }

    __aicore__ inline void InitTmpSum()
    {
        LocalTensor<T> tmpLocal0 = tmpQue_[0].AllocTensor<T>();
        LocalTensor<T> tmpLocal1 = tmpQue_[1].AllocTensor<T>();
        Duplicate<T>(tmpLocal0, 0.0, sliceSize_);
        Duplicate<T>(tmpLocal1, 0.0, sliceSize_);
        tmpQue_[0].EnQue<T>(tmpLocal0);
        tmpQue_[1].EnQue<T>(tmpLocal1);
    }

    __aicore__ inline void InitStandIndices(const uint32_t offset)
    {
        LocalTensor<uint32_t> indicesLocal = indicesQue_.AllocTensor<uint32_t>();
        uint32_t idAddrOffset = numEachCore_ * blockIdx_ + offset;
        DataCopyParams indicesCopyParams{1, static_cast<uint16_t>(sizeof(uint32_t)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(indicesLocal, indicesGlobalGm_[idAddrOffset - ADDR_BACK_STEP], indicesCopyParams, padParams);
        standIdx_ = indicesLocal.GetValue(0);
        indicesQue_.FreeTensor<uint32_t>(indicesLocal);
    }

    __aicore__ inline bool CopyIn(const uint32_t progress, const bool flag, const uint32_t repeatTimeHuge)
    {
        LocalTensor<uint32_t> indicesLocal = indicesQue_.AllocTensor<uint32_t>();
        DataCopyParams indicesCopyParams{1, static_cast<uint16_t>(sizeof(uint32_t)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(indicesLocal, indicesGm_[progress], indicesCopyParams, padParams);
        event_t eventMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventMte2ToS);
        if ((indicesLocal.GetValue(0) == standIdx_) == flag){
            indicesQue_.FreeTensor<uint32_t>(indicesLocal);
            return false;
        }
        indicesQue_.EnQue<uint32_t>(indicesLocal);
        LocalTensor<uint32_t> posIdxLocal = posIdxQue_.AllocTensor<uint32_t>();
        LocalTensor<T> valuesLocal = valuesQue_.AllocTensor<T>();
        DataCopyExtParams valuesCopyParams{1, static_cast<uint32_t>(sliceSize_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParamsValues{true, 0, 0, 0};
        DataCopyPad(posIdxLocal, posIdxGm_[progress], indicesCopyParams, padParams);
        int32_t valuesAddrOffset = posIdxLocal.GetValue(0) * sliceSizeOri_ + repeatTimeHuge * sliceSizeLimit_;
        DataCopyPad(valuesLocal, valuesGm_[valuesAddrOffset], valuesCopyParams, padParamsValues);
        pipe_barrier(PIPE_ALL);
        valuesQue_.EnQue<T>(valuesLocal);
        posIdxQue_.FreeTensor<uint32_t>(posIdxLocal);
        return true;
    }

    __aicore__ inline bool CheckIsNeedCopyOut(const int32_t currentId)
    {
        if (lastIdx_ != INDICE_INIT_PARAM && lastIdx_ != currentId) {
            switchId_ = !switchId_;
            ResetRepeatTimes(switchId_);
            return true;
        }
        return false;
    }

    __aicore__ inline void ComputeAndCopyOut(const uint32_t progress, const uint32_t loopLimit, const uint32_t repeatTimeHuge)
    {
        LocalTensor<uint32_t> indicesLocal = indicesQue_.DeQue<uint32_t>();
        LocalTensor<T> valuesLocal = valuesQue_.DeQue<T>();
        int32_t currentId = indicesLocal.GetValue(0);
        bool isCopyOut = CheckIsNeedCopyOut(currentId);
        AtomicAddInUb(valuesLocal);
        if (repeatTimes_[switchId_] > LIMIT_COUNT_NUM || progress == loopLimit) {
            CopyOut(switchId_, currentId, repeatTimeHuge);
        }
        if (isCopyOut) {
            CopyOut(!switchId_, lastIdx_, repeatTimeHuge);
        }
        lastIdx_ = currentId;
        indicesQue_.FreeTensor<uint32_t>(indicesLocal);
        valuesQue_.FreeTensor<T>(valuesLocal);
    }

    __aicore__ inline void AtomicAddInUb(LocalTensor<T> &valuesLocal)
    {
        LocalTensor<T> tmpLocal = tmpQue_[switchId_].DeQue<T>();
        Add(tmpLocal, tmpLocal, valuesLocal, sliceSize_);
        repeatTimes_[switchId_]++;
        tmpQue_[switchId_].EnQue<T>(tmpLocal);
    }

    __aicore__ inline void CopyOut(const uint32_t addrOffset, const uint32_t index, const uint32_t repeatTimeHuge)
    {
        LocalTensor<T> tmpLocal = tmpQue_[addrOffset].DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sliceSize_ * sizeof(T)), 0, 0, 0};
        SetAtomicAdd<T>();
        DataCopyPad(outputGm_[index + repeatTimeHuge * sliceSizeLimit_], tmpLocal, copyParams);
        SetAtomicNone();
        pipe_barrier(PIPE_ALL);
        Duplicate<T>(tmpLocal, 0.0, sliceSize_);
        ResetRepeatTimes(addrOffset);
        tmpQue_[addrOffset].EnQue<T>(tmpLocal);
    }

    __aicore__ inline void ContinueCompute(const uint32_t repeatTimeHuge)
    {
        lastIdx_ = INDICE_INIT_PARAM;
        InitStandIndices(rowNum_);
        uint32_t numelIdx = taskNum_;
        uint32_t preProcessRowNum = blockIdx_ * numEachCore_;
        for (uint32_t i = rowNum_; i < numelIdx - preProcessRowNum; i++) {
            if (!CopyIn(i, false, repeatTimeHuge)) {
                if (lastIdx_ != INDICE_INIT_PARAM) {
                    CopyOut(switchId_, lastIdx_, repeatTimeHuge);
                }
                break;
            }
            ComputeAndCopyOut(i, numelIdx - preProcessRowNum - 1, repeatTimeHuge);
        }
    }

    __aicore__ inline void FreeTmp()
    {
        LocalTensor<T> tmpLocal0 = tmpQue_[0].DeQue<T>();
        tmpQue_[0].FreeTensor<T>(tmpLocal0);
        LocalTensor<T> tmpLocal1 = tmpQue_[1].DeQue<T>();
        tmpQue_[1].FreeTensor<T>(tmpLocal1);
    }

    __aicore__ inline void ResetRepeatTimes(const uint64_t id)
    {
        repeatTimes_[id] = INIT_PARAM;
    }

private:
    GlobalTensor<T> outputGm_;
    GlobalTensor<uint32_t> indicesGm_;
    GlobalTensor<uint32_t> indicesGlobalGm_;
    GlobalTensor<uint32_t> posIdxGm_;
    GlobalTensor<T> valuesGm_;

    TQue<TPosition::VECIN, BUFFER_NUM> indicesQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> posIdxQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> valuesQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> tmpQue_[2];

    uint32_t numel_;
    uint32_t rowNum_;
    uint32_t repeatTimes_[2];
    uint32_t coreNum_;
    uint32_t numEachCore_;
    int32_t standIdx_ = -1;
    uint32_t taskNum_;
    uint32_t sliceSizeLimit_;
    uint32_t sliceSizeOri_;
    uint32_t sliceRepeatTime_;
    uint32_t sliceLeft_;
    int32_t lastIdx_;
    bool switchId_ = false;

    uint32_t blockIdx_;
    // attr
    uint32_t sliceSize_;
};
}

#endif // INDEX_PUT_WITH_SORT_DETERMINIST_H