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
 * \file add_lora_normal_core.h
 * \brief
 */
#ifndef ADD_LORA_NORMAL_CORE_H
#define ADD_LORA_NORMAL_CORE_H
#include "add_lora_base.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

class AddLoraNormalCoreBatchKernel: public AddLoraKernelBase {
public:
    __aicore__ inline AddLoraNormalCoreBatchKernel() {};
    __aicore__ inline void Process();

protected:
   __aicore__ inline void CopyIn2(uint32_t mInCore, uint32_t mInCoreOffset, uint32_t nInCore, uint32_t nId, uint32_t pingPongFlag);

   __aicore__ inline void CopyL0C2A1(uint32_t mInCore, uint32_t mInCoreOffset, uint32_t nInCore);

   __aicore__ inline void ComputeWeightB(uint32_t index, uint32_t nSplit, uint32_t mToProcess, uint32_t mProcessOffset, uint32_t pingPongFlag);

   __aicore__ inline void ComputeWeightA(uint32_t index, uint32_t nSplit, uint32_t mToProcess, uint32_t mProcessOffset, uint32_t pingPongFlag);
};

__aicore__ inline void AddLoraNormalCoreBatchKernel::CopyL0C2A1(uint32_t mInCore, uint32_t mInCoreOffset, uint32_t nInCore){
    int32_t eventIDM_FIX = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::M_FIX));
    SetFlag<HardEvent::M_FIX>(eventIDM_FIX);
    WaitFlag<HardEvent::M_FIX>(eventIDM_FIX);
    FixpipeParams<float> fixpipeParams;
    fixpipeParams.cburstNum = CeilCubeBlock(nInCore); // coutBlocks = (n + 16 - 1) / 16;
    fixpipeParams.burstLen = static_cast<uint16_t>(mInCore * CUBE_BLOCK * sizeof(float) / 32); // static_cast<uint16_t>(m * 16 * sizeof(float) / 32)
    fixpipeParams.srcStride = 0;
    fixpipeParams.dstStride = 0;//static_cast<uint16_t>((CeilCubeBlock(mInCore) * CUBE_BLOCK -mInCore) * CUBE_BLOCK * sizeof(half) /
            //32);
    fixpipeParams.quantParams = { QuantMode_t::F322F16 };
    Fixpipe(tmpGm_[mInCoreOffset * nInCore], matmull0C_, fixpipeParams);

    int32_t eventIDFIX_MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIX_MTE2);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIX_MTE2);
}

__aicore__ inline void AddLoraNormalCoreBatchKernel::CopyIn2(uint32_t mInCore, uint32_t mInCoreOffset, uint32_t nInCore, uint32_t nId, uint32_t pingPongFlag){
    eventId = pingPongFlag ? EVENT_ID0 : EVENT_ID1;
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(eventId);
    SetFlag<HardEvent::M_MTE2>(eventId);
    WaitFlag<HardEvent::M_MTE2>(eventId);
    if (R == nInCore){
       DataCopy(queryL1_[pingPongFlag], tmpGm_[mInCoreOffset * R], CeilCubeBlock(mInCore) * CeilCubeBlock(nInCore) * CUBE_BLOCK * CUBE_BLOCK);
    } else {
        DataCopy(queryL1_[pingPongFlag], tmpGm_[mInCoreOffset * R + mInCore*CUBE_BLOCK*nId], CeilCubeBlock(mInCore)* CeilCubeBlock(nInCore) * CUBE_BLOCK* CUBE_BLOCK);
    } //CUBE_BLOCK_SIZE 16 RSplit
    PipeBarrier<PIPE_MTE2>();
    SetFlag<HardEvent::MTE2_MTE1>(eventId);
    WaitFlag<HardEvent::MTE2_MTE1>(eventId);
}

 __aicore__ inline void AddLoraNormalCoreBatchKernel::Process()
{
    ComputeBatchIndice();
    if ASCEND_IS_AIV {
        InitDataMoveBuffer();
        QueryDataMove();
        pipe_->Reset();
        InitVectorBuffer();
        VectorNotifyCube<PIPE_MTE3>(SYNC_AIV_ONLY_ALL_FLAG, SYNC_AIV_AIC_FLAG);
    }
    if ASCEND_IS_AIC {
        InitMatmulBuffer();
        CubeWaitVector(SYNC_AIV_AIC_FLAG);
    }
    if ASCEND_IS_AIV {
        coreId = GetBlockIdx() / 2;
    }
    uint32_t mInCore = batch / coreNum;
    uint32_t mInCoreOffset = 0;
    mInCore = (coreId < batch%coreNum)? mInCore+1: mInCore;
    mInCoreOffset = (coreId < batch%coreNum)?mInCore* coreId:batch -((coreNum - coreId)* mInCore);
    uint32_t nSplit = (SINGLE_L0_SIZE/R /CUBE_BLOCK_SIZE)*CUBE_BLOCK_SIZE;
    nSplit = nSplit>MIN_SPLIT? nSplit:MIN_SPLIT; //MIN_SPLIT 128
    if (mInCore<=0){
        return;
    }
    uint32_t startIndex = IndicebymOffset.GetValue(mInCoreOffset);
    for(int i = startIndex; i<=IndicebymOffset.GetValue(mInCoreOffset+mInCore-1); ++i){
        uint32_t pingPongFlag = 0;
        uint32_t m =  sumIndiceCount[i];
        uint32_t mIndiceCoreOffset = (i==startIndex)? mInCoreOffset: sumIndiceCount[i-1];
        uint32_t mIndiceCore = (mInCoreOffset+mInCore - mIndiceCoreOffset) <(m - mIndiceCoreOffset) ? (mInCoreOffset+mInCore - mIndiceCoreOffset) : (m- mIndiceCoreOffset);
        uint32_t mSplit = MIN_SPLIT;
        if (mIndiceCore == 0 || i == wBatch){
            continue;
        }
        for (int mId=0; mId< DivCeil(mIndiceCore, mSplit); ++mId){
            uint32_t mToProcess = (mId < (mIndiceCore/mSplit)) ? mSplit: mIndiceCore -mId *mSplit;
            uint32_t mProcessOffset = mId*mSplit+mIndiceCoreOffset;
            nSplit = (nSplit* mToProcess)> SINGLE_L0_SIZE? MIN_SPLIT : nSplit;
            if ASCEND_IS_AIC {
                if(addLoraFlag){
                    ComputeWeightA(i, nSplit, mToProcess, mProcessOffset, pingPongFlag);
                }
                ComputeWeightB(i, nSplit, mToProcess, mProcessOffset, pingPongFlag);
            }
            MoveOutMM(mToProcess, mProcessOffset, H2, 0, pingPongFlag);
        }
    }
}

__aicore__ inline void AddLoraNormalCoreBatchKernel::ComputeWeightB(uint32_t index, uint32_t nSplit, uint32_t mToProcess, uint32_t mProcessOffset, uint32_t pingPongFlag){
    uint32_t nIncoreOffset = 0;
    for (int nIn = 0; nIn < DivCeil(H2, nSplit); ++nIn){
        if (nSplit == 0) {
            continue;
        }
        uint32_t nInner = (nIn <(H2/nSplit))? nSplit: H2 - nIn*nSplit;
        uint32_t RSplit =CUBE_BLOCK;
        if (!addLoraFlag){
            RSplit = MIN_SPLIT;
            for (int Ridx =0; Ridx < DivCeil(R, RSplit); ++Ridx){
                uint32_t RInner = (Ridx  < (R/RSplit)) ? RSplit: R -Ridx *RSplit;
                CopyInA2(mProcessOffset, mToProcess, RInner, Ridx,  nSplit, pingPongFlag);
                SplitA(mToProcess, RInner, mProcessOffset, pingPongFlag);
                CopyWbB(index, nIn, nInner, RInner, Ridx, nIncoreOffset, nSplit, RSplit, pingPongFlag);
                ComputeMM2(mProcessOffset, mToProcess, RInner, nInner, nIncoreOffset, Ridx, nIn, pingPongFlag);
                pingPongFlag = !pingPongFlag;
            }
        } else {
            for (int Ridx =0; Ridx < DivCeil(R, RSplit); ++Ridx){
                uint32_t RInner = (Ridx  < (R/RSplit)) ? RSplit: R -Ridx *RSplit;
                CopyIn2(mToProcess, mProcessOffset, RInner, Ridx, pingPongFlag);
                SplitA(mToProcess, RInner, mProcessOffset, pingPongFlag);
                CopyWbB(index, nIn, nInner, RInner, Ridx, nIncoreOffset, nSplit, RSplit, pingPongFlag);
                ComputeMM2(mProcessOffset, mToProcess, RInner, nInner, nIncoreOffset, Ridx, nIn, pingPongFlag);
                pingPongFlag = !pingPongFlag;
            }
        }
        CopyL0C2GM(mProcessOffset, mToProcess, nInner, nIn, nSplit, nIncoreOffset);
        pipe_barrier(PIPE_FIX);
    }
    CubeNotifyVector<PIPE_FIX>(SYNC_AIC_AIV_FLAG2);
}


__aicore__ inline void AddLoraNormalCoreBatchKernel::ComputeWeightA(uint32_t index, uint32_t nSplit, uint32_t mToProcess, uint32_t mProcessOffset, uint32_t pingPongFlag){
    uint32_t n = R;
    for (int kidx =0; kidx < DivCeil(H1, nSplit); ++kidx){
        if (nSplit == 0) {
            continue;
        }
        uint32_t k = (kidx < (H1/nSplit)) ? nSplit: H1 -kidx *nSplit;
        CopyIn(index, kidx, mToProcess, k, mProcessOffset, nSplit, pingPongFlag);
        SplitA(mToProcess, k,  mToProcess, pingPongFlag);
        SplitB(index, kidx, mToProcess, k,  mProcessOffset, pingPongFlag);
        Compute(index, kidx, mToProcess, k, n, pingPongFlag);
        pingPongFlag =!pingPongFlag;
    }
    CopyL0C2A1(mToProcess, mProcessOffset, n);
    pipe_barrier(PIPE_FIX);
}
#endif //ADD_LORA_NORMAL_CORE_H