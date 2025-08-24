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
 * \file index_put_with_sort_determinist_cast.h
 * \brief
 */

#ifndef INDEX_PUT_WITH_SORT_DETERMINIST_CAST_H
#define INDEX_PUT_WITH_SORT_DETERMINIST_CAST_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AscendC {
template<typename T>
class IndexPutWithSortDeterministCastKernel {
public:
    __aicore__ inline IndexPutWithSortDeterministCastKernel() = delete;
    __aicore__ inline IndexPutWithSortDeterministCastKernel(GM_ADDR self, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR values,
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
        uint32_t alignNum = BLOCK_SIZE / sizeof(float);
        uint32_t alignNumT = BLOCK_SIZE / sizeof(T);
        uint32_t valuesAlignNum = ((sliceSize_ + alignNumT - 1) / alignNumT) * alignNumT;
        uint32_t valuesCastAlignNum = ((sliceSize_ + alignNum - 1) / alignNum) * alignNum;
        pipe.InitBuffer(indicesQue_, BUFFER_NUM, idxAlignNum * sizeof(int));
        pipe.InitBuffer(posIdxQue_, BUFFER_NUM, idxAlignNum * sizeof(int));
        pipe.InitBuffer(tmpQue_[0], BUFFER_NUM, valuesCastAlignNum * sizeof(float));
        pipe.InitBuffer(tmpQue_[1], BUFFER_NUM, valuesCastAlignNum * sizeof(float));
        pipe.InitBuffer(valuesQue_, BUFFER_NUM, valuesAlignNum * sizeof(T));
        pipe.InitBuffer(valuesCastQue_, BUFFER_NUM, valuesCastAlignNum * sizeof(float));
        pipe.InitBuffer(selfQue_, BUFFER_NUM, valuesAlignNum * sizeof(T));
        pipe.InitBuffer(selfCastQue_, BUFFER_NUM, valuesCastAlignNum * sizeof(float));
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
        LocalTensor<float> tmpLocal0 = tmpQue_[0].AllocTensor<float>();
        LocalTensor<float> tmpLocal1 = tmpQue_[1].AllocTensor<float>();
        Duplicate<float>(tmpLocal0, 0.0, sliceSize_);
        Duplicate<float>(tmpLocal1, 0.0, sliceSize_);
        tmpQue_[0].EnQue<float>(tmpLocal0);
        tmpQue_[1].EnQue<float>(tmpLocal1);
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
        LocalTensor<float> valuesCastLocal = valuesCastQue_.AllocTensor<float>();
        DataCopyExtParams valuesCopyParams{1, static_cast<uint32_t>(sliceSize_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParamsValues{true, 0, 0, 0};
        DataCopyPad(posIdxLocal, posIdxGm_[progress], indicesCopyParams, padParams);
        SetFlag<HardEvent::MTE2_S>(eventMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventMte2ToS);
        int32_t valuesAddrOffset = posIdxLocal.GetValue(0) * sliceSizeOri_ + repeatTimeHuge * sliceSizeLimit_;
        DataCopyPad(valuesLocal, valuesGm_[valuesAddrOffset], valuesCopyParams, padParamsValues);
        SetFlag<HardEvent::MTE2_S>(eventMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventMte2ToS);
        Cast(valuesCastLocal, valuesLocal, RoundMode::CAST_NONE, sliceSize_);
        valuesCastQue_.EnQue<float>(valuesCastLocal);
        posIdxQue_.FreeTensor<uint32_t>(posIdxLocal);
        valuesQue_.FreeTensor<T>(valuesLocal);
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
        LocalTensor<float> valuesLocal = valuesCastQue_.DeQue<float>();
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
        valuesCastQue_.FreeTensor<float>(valuesLocal);
    }

    __aicore__ inline void AtomicAddInUb(LocalTensor<float> &valuesLocal)
    {
        LocalTensor<float> tmpLocal = tmpQue_[switchId_].DeQue<float>();
        Add(tmpLocal, tmpLocal, valuesLocal, sliceSize_);
        repeatTimes_[switchId_]++;
        tmpQue_[switchId_].EnQue<float>(tmpLocal);
    }

    __aicore__ inline void CopyOut(const uint32_t addrOffset, const uint32_t index, const uint32_t repeatTimeHuge)
    {
        event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        event_t eventVToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_V));
        event_t eventMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        LocalTensor<T> selfLocal = selfQue_.AllocTensor<T>();
        LocalTensor<float> selfCastLocal = selfCastQue_.AllocTensor<float>();
        LocalTensor<float> tmpLocal = tmpQue_[addrOffset].DeQue<float>();
        int32_t offset = index + repeatTimeHuge * sliceSizeLimit_;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sliceSize_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParamsValues{true, 0, 0, 0};
        DataCopyPad(selfLocal, outputGm_[offset], copyParams, padParamsValues);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
        Cast(selfCastLocal, selfLocal, RoundMode::CAST_NONE, sliceSize_);
        SetFlag<HardEvent::V_V>(eventVToV);
        WaitFlag<HardEvent::V_V>(eventVToV);
        Add(tmpLocal, tmpLocal, selfCastLocal, sliceSize_);
        SetFlag<HardEvent::V_V>(eventVToV);
        WaitFlag<HardEvent::V_V>(eventVToV);
        Cast(selfLocal, tmpLocal, RoundMode::CAST_RINT, sliceSize_);
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        DataCopyPad(outputGm_[offset], selfLocal, copyParams);
        SetFlag<HardEvent::MTE3_V>(eventMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventMte3ToV);
        Duplicate<float>(tmpLocal, 0.0, sliceSize_);
        ResetRepeatTimes(addrOffset);
        tmpQue_[addrOffset].EnQue<float>(tmpLocal);
        selfQue_.FreeTensor<T>(selfLocal);
        selfCastQue_.FreeTensor<float>(selfCastLocal);
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
        LocalTensor<float> tmpLocal0 = tmpQue_[0].DeQue<float>();
        tmpQue_[0].FreeTensor<float>(tmpLocal0);
        LocalTensor<float> tmpLocal1 = tmpQue_[1].DeQue<float>();
        tmpQue_[1].FreeTensor<float>(tmpLocal1);
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
    TQue<TPosition::VECIN, BUFFER_NUM> valuesCastQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> tmpQue_[2];
    TQue<TPosition::VECIN, BUFFER_NUM> selfQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> selfCastQue_;

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

#endif // INDEX_PUT_WITH_SORT_DETERMINIST_CAST_H