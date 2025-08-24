/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
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
 * \file mat_mul_optimized_fixpipe_algorithm_stress_detect.h
 * \brief
 */
#ifndef MM_STRESS_DETECT_OPTIMIZED_FIXPIPE_ALGORITHM_H
#define MM_STRESS_DETECT_OPTIMIZED_FIXPIPE_ALGORITHM_H

#include "mat_mul_block_stress_detect.h"
#include "mat_mul_kernel_stress_detect.h"
#include "mat_mul_bl1_full_load_stress_detect.h"

namespace MatmulStressDetect {

using namespace AscendC;
using namespace matmul;

#if defined(__CCE_KT_TEST__)
using namespace std;
#endif

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulBaseBlock,
    const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class MatmulBaseUnalignedNKernel : public MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG> {
public:
    using C_T = typename C_TYPE::T;
    __aicore__ inline MatmulBaseUnalignedNKernel() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
    __aicore__ inline void AicProcess(GlobalTensor<C_T>& cTensor, uint8_t enAtomic,
                                      bool aicNeedWaitAiv, uint8_t pingPongId);
    __aicore__ inline void AivProcess(GlobalTensor<C_T>& cTensor, uint8_t pingPongId);
    __aicore__ inline void Process(uint64_t index = 0UL, uint8_t enAtomic = 0UL);

protected:
    GlobalTensor<C_T> tempCGlobal_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MatmulCallBackFunc<nullptr, nullptr, CopyB1>> mm_;
    GatherMaskParams params_;
private:
    uint64_t baseSize_ = 0UL;
    uint64_t alignedN_ = 0UL;
    uint64_t c0Size_ = BLOCK_SIZE;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(GM_ADDR aGM, GM_ADDR bGM,
    GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe)
{
    GetSizeC0<C_T>(c0Size_);
    uint64_t cDtypeSize =sizeof(C_T);
    this->block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    this->InitInputs(aGM, bGM, cGM, biasGM);
    this->pipe_ = pipe;
    alignedN_ = AlignUp(this->block_.matmulTilingData_->matmulTiling.N, MM_ALIGN_SIZE / cDtypeSize);
    baseSize_ = alignedN_ * this->block_.matmulTilingData_->matmulTiling.baseM;
    if ASCEND_IS_AIV {
        params_.src0BlockStride = 1;
        params_.src0RepeatStride = alignedN_ / c0Size_;
        params_.src1RepeatStride = 0;
    }

    tempCGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(workspaceGM),
                                 baseSize_ * NUM_TWO * this->block_.matmulTilingData_->matmulTiling.usedCoreNum);
    this->pipe_->InitBuffer(tmpBuf_, TOTAL_UB_SIZE);
    mm_.SetUserDefInfo(reinterpret_cast<uint64_t>(tilingData));
    mm_.SetSubBlockIdx(0);
    mm_.Init(&this->block_.matmulTilingData_->matmulTiling, pipe);
    this->SetOrgShape();
    mm_.SetOrgShape(this->block_.matmulTilingData_->matmulTiling.M, this->block_.params_.alignedOriN,
                    this->block_.matmulTilingData_->matmulTiling.singleCoreK, this->block_.params_.alignedKbSize,
                    alignedN_);
    if (this->block_.params_.isHf32) {
        this->mm_.SetHF32(true, 1);
    } else {
        this->mm_.SetHF32(false, 0);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::AivProcess(
    GlobalTensor<C_T>& cTensor, uint8_t pingPongId)
{
    if ASCEND_IS_AIV {
        uint32_t vBlockIndex = GetBlockIdx();
        if (vBlockIndex >= (this->block_.matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
            return;
        }
        CrossCoreWaitFlag(0x4 + pingPongId);

        uint64_t cDtypeSize = sizeof(C_T);
        LocalTensor<C_T> ubTensor = tmpBuf_.Get<C_T>();
        // aic : aiv is 1 : 2, singlecore cal half of baseM.
        uint64_t vecM = min(MMV3CeilAlign(this->block_.params_.singleCoreM / NUM_TWO, c0Size_),
                            static_cast<uint64_t>(this->block_.params_.singleCoreM));
        uint64_t subIdx = GetSubBlockIdx();
        uint64_t srcOffset = 0UL;
        uint64_t dstOffset = 0UL;
        if (subIdx == 1) {
            srcOffset = alignedN_ * vecM;
            dstOffset = vecM * this->block_.matmulTilingData_->matmulTiling.N;
            vecM = this->block_.params_.singleCoreM - vecM;
        }
        if (vecM == 0UL) {
            return;
        }
        uint64_t ubOffset = (pingPongId * TOTAL_UB_SIZE >> 1) / cDtypeSize;
        DataCopy<C_T>(ubTensor[ubOffset], cTensor[srcOffset], vecM * alignedN_);
        CrossCoreSetFlag<0x2, PIPE_MTE2>(0x6 + pingPongId);

        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingPongId));
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingPongId));
        params_.repeatTimes = vecM;
        uint64_t rsvdCnt = 0UL;
        // src1Pattern is 7; mask is this->block_.matmulTilingData_->matmulTiling.N;
        GatherMask(ubTensor[ubOffset], ubTensor[ubOffset], 7, true, this->block_.matmulTilingData_->matmulTiling.N,
                   params_, rsvdCnt);
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingPongId));
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingPongId));
        DataCopy<C_T>(this->cGlobal_[this->block_.offset_.offsetC + dstOffset], ubTensor[ubOffset],
                      AlignUp(vecM * this->block_.matmulTilingData_->matmulTiling.N, c0Size_));
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingPongId));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingPongId));
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::AicProcess(
    GlobalTensor<typename C_TYPE::T>& cTensor, uint8_t enAtomic, bool aicNeedWaitAiv, uint8_t pingPongId)
{
    if ASCEND_IS_AIC {
        this->mm_.SetSingleShape(this->block_.params_.singleCoreM, this->block_.params_.singleCoreN,
        this->block_.matmulTilingData_->matmulTiling.singleCoreK);
        this->mm_.SetTensorA(this->aGlobal_[this->block_.offset_.offsetA], this->block_.params_.isTransposeA);
        this->mm_.SetTensorB(this->bGlobal_[this->block_.offset_.offsetB], this->block_.params_.isTransposeB);
        if (this->block_.matmulTilingData_->matmulTiling.isBias) {
            this->mm_.SetBias(this->biasGlobal_[this->block_.offset_.offsetBias]);
        }
        this->mm_.Iterate();
        if (aicNeedWaitAiv) {
            CrossCoreWaitFlag(0x6 + pingPongId);
        }
        this->mm_.GetTensorC(cTensor, enAtomic);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x4 + pingPongId);
#endif
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(
    uint64_t index, uint8_t enAtomic)
{
    bool reverse = true;
    int8_t pingPongId = 0;
    bool aicNeedWaitAiv = false;
    ctx.isFirst = true;
    ctx.inputDtypeSize = sizeof(typename A_TYPE::T);
    GlobalTensor<C_T> tempCGlobal = tempCGlobal_;
    for (uint64_t mTileIndex = 0; mTileIndex < this->block_.params_.mTileCntL2; mTileIndex++) {
        reverse = !reverse;
        for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < this->block_.params_.nTileCntL2; nTileIndexTemp++) {
            uint64_t nTileIndex = reverse ? (this->block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
            this->block_.UpdateBlockCnt(mTileIndex, nTileIndex);
            this->block_.InitBlockIndex(index);
            for (uint64_t j = 0; j < this->block_.params_.realRound; j++) {
                tempCGlobal = tempCGlobal_[baseSize_ * (block_idx * 2 + pingPongId)];
                if (this->block_.params_.rowOrder == 0) {
                    this->block_.UpdateBasicIndex(j); // 使能错位分核更新Index
                }
                if (this->block_.params_.index < this->block_.params_.totalTileCnt) {
                    this->block_.UpdateBlockParams(mTileIndex, nTileIndex);
                    this->block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIndex, nTileIndex);
                    AicProcess(tempCGlobal, enAtomic, aicNeedWaitAiv, pingPongId);
                    AivProcess(tempCGlobal, pingPongId);
                    aicNeedWaitAiv = aicNeedWaitAiv || bool(pingPongId);
                    pingPongId = (pingPongId + 1) & 1;
                }
                this->block_.UpdateBlockIndex();
            }
        }
    }
    if (this->block_.params_.isHf32) {
        this->mm_.SetHF32(false, 0);
    }
    PipeBarrier<PIPE_ALL>();
    return;
}
}
#endif