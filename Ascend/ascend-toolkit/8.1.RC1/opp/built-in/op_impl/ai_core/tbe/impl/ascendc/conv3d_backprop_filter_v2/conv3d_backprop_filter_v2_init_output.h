/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file conv3d_backprop_filter_v2_init_output.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_FILTER_V2_INIT_OUTPUT_H
#define CONV3D_BACKPROP_FILTER_V2_INIT_OUTPUT_H

namespace AscendC {
template <typename yType>
class Conv3dDwInitOutput {
public:
    __aicore__ inline Conv3dDwInitOutput() {}
    __aicore__ inline void Init(GM_ADDR y, const Conv3DBackpropFilterV2TilingData *tilingData)
    {
        InitTilingData(tilingData);
        // init global buffer
        yGm_.SetGlobalBuffer((__gm__ yType *)y);
        pipe_.InitBuffer(localBuffer_, totalL1Size_);
    }

    /** main logical function
     */
    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            return;
        }

        uint32_t usedCoreNum = GetBlockNum();
        uint64_t clearSizePerCore = AlignUp((outputSize_ + usedCoreNum - 1) / usedCoreNum, BLOCK_CUBE * BLOCK_CUBE); // Ceil(outputSize_, usedCoreNum)
        usedCoreNum = (outputSize_ + clearSizePerCore - 1) / clearSizePerCore; // Ceil(outputSize_, clearSizePerCore)
        if (GetBlockIdx() >= usedCoreNum) {
            SyncAllCores();
            return;
        }

        uint64_t offset = clearSizePerCore * GetBlockIdx();
        uint64_t realClearSize = outputSize_ - offset;
        realClearSize = realClearSize > clearSizePerCore ? clearSizePerCore : realClearSize;

        LocalTensor<yType> popBuffer = localBuffer_.template Get<yType>();
        uint32_t localSize = static_cast<uint64_t>(popBuffer.GetSize()) < realClearSize
                                 ? popBuffer.GetSize()
                                 : static_cast<uint32_t>(realClearSize);
        constexpr uint16_t MAX_BLOCK_NUM = (1 << 15) - 1;
        uint16_t repeatTime = Ceil(localSize * sizeof(yType) / ONE_BLK_SIZE, MAX_BLOCK_NUM);
        uint16_t blockNum = Ceil(localSize * sizeof(yType) / ONE_BLK_SIZE, repeatTime);
        localSize = static_cast<uint32_t>(blockNum) * repeatTime * ONE_BLK_SIZE / sizeof(yType);
        InitConstValue(popBuffer, {repeatTime, blockNum, 0, 0U});

        TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE2_MTE3>();
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);

        uint64_t round = realClearSize / localSize;
        uint32_t tail = realClearSize % localSize;
        // set 2d block num 15bit, move_L1_to_out support 16 bit
        uint16_t blockLen = localSize * sizeof(yType) / ONE_BLK_SIZE;
        struct DataCopyParams dataCopyParams(1, blockLen, 0, 0);
        for (uint64_t idx = 0; idx < round; ++idx) {
            DataCopy(yGm_[offset], popBuffer, dataCopyParams);
            offset += localSize;
        }

        if (tail != 0) {
            dataCopyParams.blockLen = tail * sizeof(yType) / ONE_BLK_SIZE;
            DataCopy(yGm_[offset], popBuffer, dataCopyParams);
        }

        SyncAllCores();
    }

    __aicore__ inline void SyncAllCores()
    {
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_AIC_FLAG);
        CrossCoreWaitFlag(SYNC_AIC_FLAG);
    }

    __aicore__ inline void Destroy()
    {
        pipe_.Destroy();
    }

protected:
    uint32_t totalL1Size_;
    uint64_t outputSize_;
    TPipe pipe_;
    TBuf<TPosition::TSCM> localBuffer_;
    GlobalTensor<yType> yGm_;

    __aicore__ inline void InitTilingData(const Conv3DBackpropFilterV2TilingData *tilingData)
    {
        totalL1Size_ = tilingData->params.totalL1Size;
        uint32_t cout1 = tilingData->dwTiling.cout1G;
        uint32_t cin1 = tilingData->dwTiling.cin1G;
        uint64_t mSize = static_cast<uint64_t>(cout1) * tilingData->dwTiling.k0;
        uint64_t nSize = static_cast<uint64_t>(cin1) * tilingData->dwTiling.k0 * tilingData->dwTiling.dk *
                         tilingData->dwTiling.hk * tilingData->dwTiling.wk;
        outputSize_ = mSize * nSize * tilingData->dwTiling.group;
    }
};
}  // namespace AscendC

#endif  // CONV3D_BACKPROP_FILTER_V2_INIT_OUTPUT_H