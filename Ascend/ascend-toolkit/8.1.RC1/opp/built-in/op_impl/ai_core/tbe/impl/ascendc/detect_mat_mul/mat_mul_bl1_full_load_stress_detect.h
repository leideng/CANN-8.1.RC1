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

/* !
 * \file mat_mul_bl1_full_load_stress_detect.h
 * \brief
 */
#ifndef MM_STRESS_DETECT_BL1_FULL_LOAD_H
#define MM_STRESS_DETECT_BL1_FULL_LOAD_H

#include "mat_mul_kernel_stress_detect.h"

using namespace AscendC;
using namespace matmul;

namespace MatmulStressDetect {

struct CallBackContext {
    bool isFirst;
    uint64_t inputDtypeSize;
};

__BLOCK_LOCAL__ __inline__ CallBackContext ctx;

template<class T>
__aicore__ inline void DataCopyB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm,
                                  const Nd2NzParams &nd2nzParams, uint64_t bl1Size)
{
    LocalTensor<T> dst = bMatrix.ReinterpretCast<T>();
    GlobalTensor<T> src;
    src.SetGlobalBuffer((__gm__ T*)gm, bl1Size);
    if (ctx.isFirst) {
        DataCopy(dst, src, nd2nzParams);
        ctx.isFirst = false;
    }
}

__aicore__ inline void CopyB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col, int useK,
    int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    MatmulTilingData matmulTilingData;
    MatmulTilingData* tilingDataPtr = reinterpret_cast<MatmulTilingData*>(tilingPtr);
    if (tilingDataPtr != nullptr) {
        matmulTilingData = *tilingDataPtr;
    }
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    uint64_t nDim = static_cast<uint64_t>(useK);
    uint64_t dDim = static_cast<uint64_t>(useN);
    if (matmulTilingData.matmulRunInfo.transB) {
        nDim = static_cast<uint64_t>(useN);
        dDim = static_cast<uint64_t>(useK);
    }
    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = dDim;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue = dDim;
    nd2nzParams.dstNzC0Stride = MMV3CeilAlign(nDim, static_cast<uint64_t>(BLOCK_SIZE));
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    uint64_t bl1Size = matmulTilingData.matmulTiling.Kb * matmulTilingData.matmulTiling.N;
    if (ctx.inputDtypeSize == sizeof(float)) {
        DataCopyB1<float>(bMatrix, gm, nd2nzParams, bl1Size);
    } else {
        DataCopyB1<half>(bMatrix, gm, nd2nzParams, bl1Size);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulBaseBlock,
    const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>>
class MatmulBaseKernelBL1FullLoad : public MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> {
public:
    __aicore__ inline MatmulBaseKernelBL1FullLoad() {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);

    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);

protected:
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MatmulCallBackFunc<nullptr, nullptr, CopyB1>> mm_;
};


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
    class MM_CB>
__aicore__ inline void MatmulBaseKernelBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    this->block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    this->pipe_ = pipe;
    this->InitInputs(aGM, bGM, cGM, biasGM);

    mm_.SetSubBlockIdx(0);
    mm_.Init(&this->block_.matmulTilingData_->matmulTiling, this->pipe_);
    mm_.SetUserDefInfo(reinterpret_cast<uint64_t>(tilingData));
    this->SetOrgShape();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
    class MM_CB>
__aicore__ inline void MatmulBaseKernelBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
    MM_CB>::Process(uint64_t index, uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }
    ctx.isFirst = true;
    ctx.inputDtypeSize = sizeof(typename A_TYPE::T);
    mm_.SetHF32(false, 0);
    if (this->block_.params_.isHf32) {
        mm_.SetHF32(true, 1);
    }
    for (uint64_t mTileIndex = 0; mTileIndex < this->block_.params_.mTileCntL2; mTileIndex++) {
        for (uint64_t nTileIndex = 0; nTileIndex < this->block_.params_.nTileCntL2; nTileIndex++) {
            this->block_.UpdateBlockCnt(mTileIndex, nTileIndex);
            this->block_.InitBlockIndex(index);
            for (uint64_t j = 0; j < this->block_.params_.realRound; j++) {
                if (this->block_.params_.index < this->block_.params_.totalTileCnt) {
                    this->block_.UpdateBlockParams(mTileIndex, nTileIndex);
                    this->block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIndex, nTileIndex);
                    mm_.SetSingleShape(this->block_.params_.singleCoreM, this->block_.params_.singleCoreN,
                        this->block_.matmulTilingData_->matmulTiling.singleCoreK);
                    mm_.SetTensorA(this->aGlobal_[this->block_.offset_.offsetA], this->block_.params_.isTransposeA);
                    mm_.SetTensorB(this->bGlobal_[this->block_.offset_.offsetB], this->block_.params_.isTransposeB);
                    if (this->block_.matmulTilingData_->matmulTiling.isBias) {
                        mm_.SetBias(this->biasGlobal_[this->block_.offset_.offsetBias]);
                    }
                    mm_.IterateAll(this->cGlobal_[this->block_.offset_.offsetC], enAtomic);
                }
                this->block_.UpdateBlockIndex();
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    mm_.SetHF32(false, 0);
    return;
}

// Current Kernel support only nd2nzA. No need to do nd2nz for B.
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulBaseBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class MatmulBaseUnAlignedKernelBL1FullLoad
    : public MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> {
 public:
    __aicore__ inline MatmulBaseUnAlignedKernelBL1FullLoad() {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const MatmulTilingData *tilingData, TPipe *pipe);

    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);

 protected:
    using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
    MatmulBaseKernelBL1FullLoad<aType, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                MatmulCallBackFunc<nullptr, nullptr, CopyB1>>
        mma_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatmulBaseUnAlignedKernelBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const MatmulTilingData *matmulTilingData, TPipe *pipe)
{
    this->pipe_ = pipe;
    this->pipe_->InitBuffer(this->ubBuf_, TOTAL_UB_SIZE);
    this->matmulTilingData_ = matmulTilingData;
    this->innerParams_.isTransposeA = this->matmulTilingData_->matmulRunInfo.transA;
    this->innerParams_.isTransposeB = this->matmulTilingData_->matmulRunInfo.transB;
    bool nd2nzA = this->matmulTilingData_->matmulRunInfo.nd2nzA;
    this->innerParams_.nd2nzFlag = nd2nzA ? ND2NZ_SELECT::ONLY_A : 0;
    this->innerParams_.baseAN = matmulTilingData->baseAN;
    this->innerParams_.baseAD = matmulTilingData->baseAD;
    this->CalculateabGM(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);
    mma_.Init(this->innerParams_.workspaceGMNZ, bGM, cGM, biasGM, offsetWGM, workspaceGM, this->matmulTilingData_,
                this->pipe_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatmulBaseUnAlignedKernelBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(uint64_t index,
                                                                                                     uint8_t enAtomic)
{
    this->ProcessNDtoNZ();
    if ASCEND_IS_AIV {
        return;
    }
    mma_.Process(index, enAtomic);
}

}  // namespace MatmulStressDetect
#endif  // MM_STRESS_DETECT_BL1_FULL_LOAD_H