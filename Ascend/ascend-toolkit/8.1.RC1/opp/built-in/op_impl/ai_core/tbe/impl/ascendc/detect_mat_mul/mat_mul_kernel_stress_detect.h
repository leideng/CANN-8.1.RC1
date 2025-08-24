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
 * \file mat_mul_kernel_stress_detect.h
 * \brief
 */
#ifndef MM_STRESS_DETECT_KERNEL_H
#define MM_STRESS_DETECT_KERNEL_H

#include "mat_mul_block_stress_detect.h"

namespace MatmulStressDetect {

using namespace AscendC;
using namespace matmul;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulStressDetect::MatmulBaseBlock,
    const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class MatmulBaseKernel {
public:
    __aicore__ inline MatmulBaseKernel() {}

    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM);
    __aicore__ inline void SetOrgShape();

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);

    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM);

    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);
    __aicore__ inline void End()
    {
        mm_.End();
    }
    __aicore__ inline const BLOCK_TYPE GetBlock()
    {
        return block_;
    }

protected:
    BLOCK_TYPE block_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;
    TBuf<> ubBuf_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(GM_ADDR aGM,
    GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM, const void *tilingData,
    TPipe *pipe)
{
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    pipe_ = pipe;
    InitInputs(aGM, bGM, cGM, biasGM);

    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.matmulTilingData_->matmulTiling, pipe_);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);
    LocalTensor<uint8_t> buf = ubBuf_.template Get<uint8_t>();
    mm_.SetLocalWorkspace(buf);
#endif
    SetOrgShape();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::InitInputs(GM_ADDR aGM,
    GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
        static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.M) * block_.matmulTilingData_->matmulTiling.Ka);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
        static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.Kb) * block_.matmulTilingData_->matmulTiling.N);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
        static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.M) * block_.matmulTilingData_->matmulTiling.N);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), block_.matmulTilingData_->matmulTiling.N);
    SetL2CacheEnable(block_.matmulTilingData_->l2cacheUseInfo, aGlobal_, bGlobal_, cGlobal_, biasGlobal_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::SetOrgShape()
{
    if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.params_.alignedOriN, block_.params_.alignedKaSize,
            block_.params_.alignedKbSize, block_.matmulTilingData_->matmulTiling.N);
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.matmulTilingData_->matmulTiling.N,
            block_.params_.alignedKaSize, block_.matmulTilingData_->matmulTiling.Kb,
            block_.matmulTilingData_->matmulTiling.N);
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.matmulTilingData_->matmulTiling.M, block_.params_.alignedOriN,
            block_.matmulTilingData_->matmulTiling.singleCoreK, block_.params_.alignedKbSize,
            block_.matmulTilingData_->matmulTiling.N);
    }
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::UpdateGlobalTensor(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    InitInputs(aGM, bGM, cGM, biasGM);
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(uint64_t index,
    uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }
    mm_.SetHF32(false, 0);
    if (block_.params_.isHf32) {
        mm_.SetHF32(true, 1);
    }
    bool reverse = true;
    for (uint64_t mTileIndex = 0; mTileIndex < block_.params_.mTileCntL2; mTileIndex++) {
        reverse = !reverse;
        for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < block_.params_.nTileCntL2; nTileIndexTemp++) {
            uint64_t nTileIndex = reverse ? (block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
            block_.UpdateBlockCnt(mTileIndex, nTileIndex);
            block_.InitBlockIndex(index);
            for (uint64_t j = 0; j < block_.params_.realRound; j++) {
                if (block_.params_.rowOrder == 0) {
                    block_.UpdateBasicIndex(j); // 使能错位分核更新Index
                }
                if (block_.params_.index < block_.params_.totalTileCnt) {
                    block_.UpdateBlockParams(mTileIndex, nTileIndex);

                    block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIndex, nTileIndex);

                    mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                        block_.matmulTilingData_->matmulTiling.singleCoreK);
                    mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], block_.params_.isTransposeA);
                    mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], block_.params_.isTransposeB);
                    if (block_.matmulTilingData_->matmulTiling.isBias) {
                        mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
                    }
                    mm_.Iterate();
                    mm_.GetTensorC(cGlobal_[block_.offset_.offsetC], enAtomic);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
                    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
                    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
#endif
                }
                block_.UpdateBlockIndex();
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    mm_.SetHF32(false, 0);
    return;
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulStressDetect::MatmulBaseBlock,
    const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class MatmulBaseUnAlignedKernel {
    struct BaseUnAlignedKernelParams {
        bool isTransposeA;
        bool isTransposeB;
        int nd2nzFlag; // 2表示B矩阵做nd2nz，1表示A矩阵做nd2nz
        GM_ADDR aGMNZ;
        GM_ADDR bGMNZ;
        GM_ADDR workspaceGMNZ;
        GM_ADDR workspaceGMabNZ;
        uint64_t baseAN;
        uint64_t baseAD;
        uint64_t baseBN;
        uint64_t baseBD;
    };

public:
    __aicore__ inline MatmulBaseUnAlignedKernel() {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM, const MatmulTilingData *matmulTilingData, TPipe *pipe);

    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM);

    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);

    __aicore__ inline void End()
    {
        if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
            mmb_.End();
        } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
            mma_.End();
        } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
            mmab_.End();
        }
    }

protected:
    __aicore__ inline void ProcessNDtoNZ();
    __aicore__ inline void CalculateabGM(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                         GM_ADDR workspaceGM);
    using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
    using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
    using a_T = typename aType::T;
    using b_T = typename bType::T;
    MatmulBaseKernel<aType, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mma_;
    MatmulBaseKernel<A_TYPE, bType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mmb_;
    MatmulBaseKernel<aType, bType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mmab_;
    BaseUnAlignedKernelParams innerParams_;
    TPipe *pipe_;
    TBuf<> ubBuf_;
    const MatmulTilingData *matmulTilingData_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG>
__aicore__ inline void MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const MatmulTilingData* matmulTilingData, TPipe* pipe)
{
    pipe_ = pipe;
    pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);
    matmulTilingData_ = matmulTilingData;
    innerParams_.isTransposeA = matmulTilingData_->matmulRunInfo.transA;
    innerParams_.isTransposeB = matmulTilingData_->matmulRunInfo.transB;
    bool nd2nzA = matmulTilingData_->matmulRunInfo.nd2nzA;
    bool nd2nzB = matmulTilingData_->matmulRunInfo.nd2nzB;
    innerParams_.baseAN = matmulTilingData->baseAN;
    innerParams_.baseAD = matmulTilingData->baseAD;
    innerParams_.baseBN = matmulTilingData->baseBN;
    innerParams_.baseBD = matmulTilingData->baseBD;

    if (nd2nzA) {
        innerParams_.nd2nzFlag = ND2NZ_SELECT::ONLY_A;
    }
    if (nd2nzB) {
        innerParams_.nd2nzFlag = ND2NZ_SELECT::ONLY_B;
    }
    if (nd2nzA && nd2nzB) {
        innerParams_.nd2nzFlag = ND2NZ_SELECT::BOTH_AB;
    }

    CalculateabGM(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);
    if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
        mmb_.Init(aGM, innerParams_.workspaceGMNZ, cGM, biasGM, offsetWGM, workspaceGM, matmulTilingData_, pipe_);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
        mma_.Init(innerParams_.workspaceGMNZ, bGM, cGM, biasGM, offsetWGM, workspaceGM, matmulTilingData_, pipe_);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
        mmab_.Init(innerParams_.workspaceGMNZ, innerParams_.workspaceGMabNZ, cGM, biasGM, offsetWGM, workspaceGM,
            matmulTilingData_, pipe_);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG>
__aicore__ inline void
MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::UpdateGlobalTensor(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    CalculateabGM(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);
    if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
        mmb_.UpdateGlobalTensor(aGM, innerParams_.workspaceGMNZ, cGM, biasGM, offsetWGM, workspaceGM);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
        mma_.UpdateGlobalTensor(innerParams_.workspaceGMNZ, bGM, cGM, biasGM, offsetWGM, workspaceGM);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
        mmab_.UpdateGlobalTensor(
            innerParams_.workspaceGMNZ, innerParams_.workspaceGMabNZ, cGM, biasGM, offsetWGM, workspaceGM);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::CalculateabGM(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    innerParams_.aGMNZ = aGM;
    innerParams_.bGMNZ = bGM;
    using A_T = typename A_TYPE::T;
    uint64_t c0Size;
    GetSizeC0<A_T>(c0Size);
    auto alignedMSize = MMV3DivCeil(matmulTilingData_->matmulTiling.M, ALIGNED_H) * ALIGNED_H; // N轴转换成分型
    auto alignedKSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, c0Size) * c0Size;      // K轴转换成分型
    if (innerParams_.isTransposeA) {
        alignedMSize = MMV3DivCeil(matmulTilingData_->matmulTiling.M, c0Size) * c0Size;        // N轴转换成分型
        alignedKSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, ALIGNED_H) * ALIGNED_H; // K轴转换成分型
    }
    uint64_t inputDtypeSize = sizeof(typename A_TYPE::T);
    innerParams_.workspaceGMNZ = workspaceGM + MAX_BLOCK_NUM * DEFAULT_BLOCK_LEN * sizeof(int32_t);
    innerParams_.workspaceGMabNZ = workspaceGM + MAX_BLOCK_NUM * DEFAULT_BLOCK_LEN * sizeof(int32_t) +
        alignedMSize * alignedKSize * inputDtypeSize;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG>
__aicore__ inline void
MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::ProcessNDtoNZ()
{
    // ND2NZ
    if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
        MatrixBtoNZV2<typename B_TYPE::T>(innerParams_.workspaceGMNZ, innerParams_.bGMNZ,
            matmulTilingData_->matmulTiling, innerParams_.isTransposeB, ubBuf_, innerParams_.baseBN,
            innerParams_.baseBD);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
        MatrixAtoNZV2<typename A_TYPE::T>(innerParams_.workspaceGMNZ, innerParams_.aGMNZ,
            matmulTilingData_->matmulTiling, innerParams_.isTransposeA, ubBuf_, innerParams_.baseAN,
            innerParams_.baseAD);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
        MatrixAtoNZV2<typename A_TYPE::T>(innerParams_.workspaceGMNZ, innerParams_.aGMNZ,
            matmulTilingData_->matmulTiling, innerParams_.isTransposeA, ubBuf_, innerParams_.baseAN,
            innerParams_.baseAD);
        event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        MatrixBtoNZV2<typename B_TYPE::T>(innerParams_.workspaceGMabNZ, innerParams_.bGMNZ,
            matmulTilingData_->matmulTiling, innerParams_.isTransposeB, ubBuf_, innerParams_.baseBN,
            innerParams_.baseBD);
    }
    SyncAll();
    // CV SYNC
    if ASCEND_IS_AIV {
        NotifyEvent<PIPE_MTE3>(4);
    }
    if ASCEND_IS_AIC {
        WaitEvent(4);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(
    uint64_t index, uint8_t enAtomic)
{
    ProcessNDtoNZ();
    if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
        mmb_.Process(index, enAtomic);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
        mma_.Process(index, enAtomic);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
        mmab_.Process(index, enAtomic);
    }
}

__aicore__ inline void WaitFlagDevLocal(int64_t flagID)
{
#if defined(__DAV_C310__)
    wait_flag_dev(PIPE_S, flagID);
#else
    wait_flag_dev(flagID);
#endif
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE,
    class BLOCK_TYPE = MatmulStressDetect::MatmulSingleCoreSplitKBaseBlock, const MatmulConfig &MM_CFG = MM_CFG_PRELOAD>
class MatMulBaseKernelSingleCoreSplitK {
public:
    __aicore__ inline MatMulBaseKernelSingleCoreSplitK() {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM, const MatmulTilingData *matmulTilingData, TPipe *pipe);

    __aicore__ inline void UnAlignedInit(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM, const MatmulTilingData *matmulTilingData, TPipe *pipe);

    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM);

    __aicore__ inline void Process(GM_ADDR cGM, GM_ADDR srcAddr, TBuf<TPosition::VECCALC> &ubBuf);

    __aicore__ inline void UnAlignedProcess();

    __aicore__ inline void End()
    {
        mm_.End();
    }

protected:
    BLOCK_TYPE block_;
    MatmulImpl<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE, MM_CFG> mm_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename L0C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;
    bool n128AlignFlag_ = false;

private:
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM);
    __aicore__ inline void SetOrgShape();
};

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
    const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const MatmulTilingData *matmulTilingData, TPipe *pipe)
{
    n128AlignFlag_ = (block_.matmulTilingData_->matmulTiling.N % ALIGN_128_BYTE == 0);
    block_.template Init<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(matmulTilingData);
    if ASCEND_IS_AIV {
        return;
    }
    SetAtomicNone();
    if (block_idx >= block_.matmulTilingData_->matmulTiling.usedCoreNum) {
        return;
    }
    pipe_ = pipe;
    InitInputs(aGM, bGM, cGM, biasGM, workspaceGM);

    mm_.SetSubBlockIdx(0);
    PRELOAD(4);
    mm_.Init(&block_.matmulTilingData_->matmulTiling, pipe_);
    SetOrgShape();
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
    const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE,
    BLOCK_TYPE, MM_CFG>::UnAlignedInit(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const MatmulTilingData *matmulTilingData, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        return;
    }
    SetAtomicNone();
    block_.template Init<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(matmulTilingData);
    if (block_idx >= block_.matmulTilingData_->matmulTiling.usedCoreNum) {
        return;
    }
    pipe_ = pipe;
    InitInputs(aGM, bGM, cGM, biasGM, workspaceGM);

    mm_.SetSubBlockIdx(0);
    PRELOAD(4);
    mm_.Init(&block_.matmulTilingData_->matmulTiling, pipe_);
    SetOrgShape();
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
    const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::InitInputs(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename L0C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
        static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.M) * block_.matmulTilingData_->matmulTiling.Ka);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
        static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.Kb) * block_.matmulTilingData_->matmulTiling.N);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
        static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.M) * block_.matmulTilingData_->matmulTiling.N);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), block_.matmulTilingData_->matmulTiling.N);
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
    const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::SetOrgShape()
{
    if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.params_.alignedOriN, block_.params_.alignedKaSize,
            block_.params_.alignedKbSize, block_.params_.outNAlign);
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.matmulTilingData_->matmulTiling.N,
            block_.params_.alignedKaSize, block_.matmulTilingData_->matmulTiling.Kb, block_.params_.outNAlign);
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.matmulTilingData_->matmulTiling.M, block_.params_.alignedOriN,
            block_.matmulTilingData_->matmulTiling.Ka, block_.params_.alignedKbSize, block_.params_.outNAlign);
    } else {
        if (n128AlignFlag_) {
            mm_.SetOrgShape(block_.matmulTilingData_->matmulTiling.M, block_.matmulTilingData_->matmulTiling.N,
                block_.matmulTilingData_->matmulTiling.Ka, block_.matmulTilingData_->matmulTiling.Kb,
                block_.matmulTilingData_->matmulTiling.N);
        } else {
            mm_.SetOrgShape(block_.matmulTilingData_->matmulTiling.M, block_.matmulTilingData_->matmulTiling.N,
                block_.matmulTilingData_->matmulTiling.Ka, block_.matmulTilingData_->matmulTiling.Kb,
                block_.params_.outNAlign);
        }
    }
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
    const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::UpdateGlobalTensor(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    if (block_idx >= block_.matmulTilingData_->matmulTiling.usedCoreNum) {
        return;
    }

    InitInputs(aGM, bGM, cGM, biasGM, workspaceGM);
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
    const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(
    GM_ADDR cGM, GM_ADDR srcAddr, TBuf<TPosition::VECCALC> &ubBuf)
{
    block_.InitBlockIndex();
    for (uint64_t j = 0; j < block_.params_.realRound; ++j) {
        block_.UpdateBlockCnt();
        for (uint64_t innerMIndex = 0; innerMIndex < block_.params_.innerLoopM; ++innerMIndex) {
            if ASCEND_IS_AIV {
                // Cast f322f16
                WaitFlagDevLocal(5);
                // do_cast C：innerSingleCoreM * nCoreUse
                block_.UpdateBlockParams(innerMIndex, 0);
                uint64_t singleMOffset = block_.params_.mIndex * block_.matmulTilingData_->matmulTiling.singleCoreM;
                uint64_t innerMOffset = innerMIndex * block_.params_.innerBlockM;
                uint64_t offset = (singleMOffset + innerMOffset) * block_.matmulTilingData_->matmulTiling.N +
                                  block_.params_.nIndex * block_.matmulTilingData_->matmulTiling.singleCoreN;
                uint64_t vMOffset = MMV3DivCeil(block_.params_.innerSingleCoreM, NUM_TWO);
                if (GetBlockIdx() % NUM_TWO == 1) { // 一个C核对应两个V核中的第二个V核的计算处理
                    offset = offset + vMOffset * block_.matmulTilingData_->matmulTiling.N;
                    vMOffset = block_.params_.innerSingleCoreM - vMOffset;
                }
                uint64_t singleSize = vMOffset * block_.params_.nCoreUse;
                Cast32to16V220(reinterpret_cast<__gm__ typename OUTPUT_TYPE::T *>(cGM) + offset,
                    reinterpret_cast<__gm__ float *>(srcAddr) + offset, singleSize,
                    block_.params_.nCoreUse, block_.matmulTilingData_->matmulTiling.N, ubBuf);
                PipeBarrier<PIPE_ALL>();
            }
            if ASCEND_IS_AIC {
                mm_.SetHF32(false, 0);
                if (block_.params_.isHf32) {
                    mm_.SetHF32(true, 1);
                }
                for (uint64_t kIndex = 0; kIndex < block_.params_.loopK; ++kIndex) {
                    block_.UpdateBlockParams(innerMIndex, kIndex);
                    for (uint64_t innerNIndex = 0; innerNIndex < block_.params_.innerLoopN; ++innerNIndex) {
                        block_.template CalcGMOffset<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(innerMIndex, kIndex,
                            innerNIndex);
                        mm_.SetSingleShape(block_.params_.innerSingleCoreM, block_.params_.innerSingleCoreN,
                            block_.params_.kCoreUse);
                        mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], block_.params_.isTransposeA);
                        mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], block_.params_.isTransposeB);
                        if (kIndex == 0) {
                            block_.params_.atomicAddFlag = false;
                            if (block_.matmulTilingData_->matmulTiling.isBias) {
                                mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
                            }
                        } else {
                            block_.params_.atomicAddFlag = true;
                        }
                        mm_.IterateAll(cGlobal_[block_.offset_.offsetC], block_.params_.atomicAddFlag);
                        mm_.ClearBias();
                    }
                }
                mm_.SetHF32(false, 0);
                // c侧做完才能做v侧
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
                NotifyEvent<PIPE_FIX>(5);
#endif
                PipeBarrier<PIPE_ALL>();
            }
        }
        block_.UpdateBlockIndex();
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    return;
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
    const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::UnAlignedProcess()
{
    if ASCEND_IS_AIV {
        return;
    }

    if (block_idx >= block_.matmulTilingData_->matmulTiling.usedCoreNum) {
        return;
    }
    mm_.SetHF32(false, 0);
    if (block_.params_.isHf32) {
        mm_.SetHF32(true, 1);
    }

    block_.InitBlockIndex();
    for (uint64_t j = 0; j < block_.params_.realRound; ++j) {
        block_.UpdateBlockCnt();
        for (uint64_t innerMIndex = 0; innerMIndex < block_.params_.innerLoopM; ++innerMIndex) {
            for (int kIndex = 0; kIndex < block_.params_.loopK; ++kIndex) {
                block_.UpdateBlockParams(innerMIndex, kIndex);
                for (uint64_t innerNIndex = 0; innerNIndex < block_.params_.innerLoopN; ++innerNIndex) {
                    block_.template CalcGMOffset<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(innerMIndex, kIndex,
                    innerNIndex);
                    mm_.SetSingleShape(block_.params_.innerSingleCoreM, block_.params_.innerSingleCoreN,
                        block_.params_.kCoreUse);
                    mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], block_.params_.isTransposeA);
                    mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], block_.params_.isTransposeB);
                    if (kIndex == 0) {
                        block_.params_.atomicAddFlag = false;
                        if (block_.matmulTilingData_->matmulTiling.isBias) {
                            mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
                        }
                    } else {
                        block_.params_.atomicAddFlag = true;
                    }
                    mm_.IterateAll(cGlobal_[block_.offset_.offsetC], block_.params_.atomicAddFlag);
                    mm_.ClearBias();
                }
            }
        }
        block_.UpdateBlockIndex();
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    mm_.SetHF32(false, 0);
    return;
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulStressDetect::MatmulSingleCoreSplitKBaseBlock,
    const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class MatMulSingleCoreSplitKKernel {
    struct SingleCoreSplitKParams {
        GM_ADDR alignedworkspaceGM;
        uint64_t vIndex;
        uint64_t alignedN;
        uint64_t coreSizeNum;
        uint64_t offset;
        GM_ADDR cGM;
        bool n128Align = false;
    };

public:
    __aicore__ inline MatMulSingleCoreSplitKKernel() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM, const MatmulTilingData *matmulTilingData, TPipe *pipe);
    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM);
    __aicore__ inline void Process();
    __aicore__ inline void NNot128AlignProcess();
    __aicore__ inline void End()
    {
        mmcBaseKernel_.End();
    }

protected:
    using cType = MatmulType<C_TYPE::pos, C_TYPE::format, float, C_TYPE::isTrans>;
    MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, cType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mmcBaseKernel_;

    TPipe *pipe_;
    TBuf<> ubBuf_;
    const MatmulTilingData *matmulTilingData_;
    SingleCoreSplitKParams innerParams_;
    GlobalTensor<float> cTmpGlobal_;
    GlobalTensor<float> matmulOutput_;
    GlobalTensor<typename C_TYPE::T> castCGm_;

private:
    __aicore__ inline void ProcessRemovePaddingImpl();
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM);
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const MatmulTilingData *matmulTilingData, TPipe *pipe)
{
    pipe_ = pipe;
    matmulTilingData_ = matmulTilingData;
    innerParams_.n128Align = (matmulTilingData_->matmulTiling.N % ALIGN_128_BYTE == 0);

    InitInputs(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);
    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
            return;
        }
        using C_T = typename C_TYPE::T;
        if constexpr (sizeof(C_T) == sizeof(float)) {
            innerParams_.alignedworkspaceGM = innerParams_.cGM;
        }
        if (!innerParams_.n128Align) {
            mmcBaseKernel_.UnAlignedInit(aGM, bGM, innerParams_.alignedworkspaceGM, biasGM, offsetWGM, workspaceGM,
                matmulTilingData, pipe_);
            return;
        }
    }
    mmcBaseKernel_.Init(aGM, bGM, innerParams_.alignedworkspaceGM, biasGM, offsetWGM, workspaceGM, matmulTilingData,
        pipe_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::InitInputs(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    innerParams_.alignedworkspaceGM = reinterpret_cast<GM_ADDR>(
        ((reinterpret_cast<uint64_t>(workspaceGM + MAX_BLOCK_NUM * DEFAULT_BLOCK_LEN * sizeof(int32_t)) + 511) / 512) *
        512);
    innerParams_.cGM = cGM;

    if ASCEND_IS_AIV {
        using C_T = typename C_TYPE::T;
        if constexpr (sizeof(C_T) == sizeof(float)) {
            return;
        }
        // Clear gm
        innerParams_.vIndex = GetBlockIdx();
        if (innerParams_.vIndex >= (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
            return;
        }
        uint64_t totalSize = static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) *
                             static_cast<uint64_t>(matmulTilingData_->matmulTiling.N);
        uint64_t coreSize = totalSize / (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO); // need to align
        innerParams_.coreSizeNum = coreSize;
        innerParams_.offset = innerParams_.vIndex * coreSize;
        if (innerParams_.vIndex == matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO - 1) {
            // 尾块数据量
            innerParams_.coreSizeNum =
                totalSize - (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO - 1) * coreSize;
        }
        cTmpGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(innerParams_.alignedworkspaceGM), totalSize);
        InitOutput<float>(cTmpGlobal_[innerParams_.offset], innerParams_.coreSizeNum, 0);
        pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);
        if (matmulTilingData_->matmulTiling.N * DATA_SIZE_FP32 % ALIGN_BYTE != 0) {
            innerParams_.alignedN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, 64) * 64;
            matmulOutput_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(innerParams_.alignedworkspaceGM),
                matmulTilingData_->matmulTiling.M * innerParams_.alignedN);
            castCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ typename C_TYPE::T *>(cGM),
                matmulTilingData_->matmulTiling.M * matmulTilingData_->matmulTiling.N);
        }
        return;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::UpdateGlobalTensor(GM_ADDR aGM,
    GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    InitInputs(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);

    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
            return;
        }
        using C_T = typename C_TYPE::T;
        if constexpr (sizeof(C_T) == sizeof(float)) {
            innerParams_.alignedworkspaceGM = innerParams_.cGM;
        }
        mmcBaseKernel_.UpdateGlobalTensor(aGM, bGM, innerParams_.alignedworkspaceGM, biasGM, offsetWGM, workspaceGM);
        return;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::ProcessRemovePaddingImpl()
{
    if (matmulTilingData_->matmulTiling.N * DATA_SIZE_FP32 % ALIGN_BYTE != 0) {
        uint64_t splitM = matmulTilingData_->matmulTiling.M / (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO);
        uint64_t coreMSize = splitM;
        if (matmulTilingData_->matmulTiling.M < (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
            splitM = 1;
            if (innerParams_.vIndex * splitM >= matmulTilingData_->matmulTiling.M) {
                PipeBarrier<PIPE_ALL>();
                return;
            }
            coreMSize = splitM;
        } else {
            if (innerParams_.vIndex == matmulTilingData_->matmulTiling.usedCoreNum * 2 - 1) {
                coreMSize = matmulTilingData_->matmulTiling.M - coreMSize * innerParams_.vIndex;
            }
        }
        RemovePaddingImpl<float, typename C_TYPE::T>(
            castCGm_[innerParams_.vIndex * splitM * matmulTilingData_->matmulTiling.N],
            matmulOutput_[innerParams_.vIndex * splitM * innerParams_.alignedN], coreMSize, innerParams_.alignedN,
            matmulTilingData_->matmulTiling.N, ubBuf_);
    } else {
        UnAlignedCast32to16V220(reinterpret_cast<__gm__ typename C_TYPE::T *>(innerParams_.cGM) + innerParams_.offset,
            reinterpret_cast<__gm__ float *>(innerParams_.alignedworkspaceGM) + innerParams_.offset, 0,
            innerParams_.coreSizeNum, ubBuf_);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process()
{
    if (!innerParams_.n128Align) {
        NNot128AlignProcess();
        return;
    }
    using C_T = typename C_TYPE::T;
    if ASCEND_IS_AIV {
        if constexpr (sizeof(C_T) == sizeof(float)) {
            return;
        }
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO) {
            return;
        }
        SyncAll();
        NotifyEvent<PIPE_MTE3>(6);
        PipeBarrier<PIPE_ALL>();
    }
    if constexpr (sizeof(C_T) == sizeof(float)) {
        // fp32不需要vector核
        mmcBaseKernel_.UnAlignedProcess();
        return;
    }
    if ASCEND_IS_AIC {
        WaitFlagDevLocal(6);
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
            return;
        }
    }
    mmcBaseKernel_.Process(innerParams_.cGM, innerParams_.alignedworkspaceGM, ubBuf_);
    return;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::NNot128AlignProcess()
{
    using C_T = typename C_TYPE::T;
    if ASCEND_IS_AIV {
        if constexpr (sizeof(C_T) == sizeof(float)) {
            return;
        }
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO) {
            NotifyEvent<PIPE_MTE3>(6);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        SyncAll();
        NotifyEvent<PIPE_MTE3>(6);
        PipeBarrier<PIPE_ALL>();
        // Cast f322f16
        WaitFlagDevLocal(5);
        SyncAll();
        PipeBarrier<PIPE_ALL>();

        ProcessRemovePaddingImpl();

        PipeBarrier<PIPE_ALL>();
        return;
    }
    if constexpr (sizeof(C_T) == sizeof(float)) {
        // fp32不需要vector核
        mmcBaseKernel_.UnAlignedProcess();
        return;
    }
    if ASCEND_IS_AIC {
        WaitFlagDevLocal(6);
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            NotifyEvent<PIPE_FIX>(5);
#endif
            PipeBarrier<PIPE_ALL>();
            return;
        }
        mmcBaseKernel_.UnAlignedProcess();
        // c侧做完才能做v侧
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        NotifyEvent<PIPE_FIX>(5);
#endif
        PipeBarrier<PIPE_ALL>();
        return;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulStressDetect::MatmulSingleCoreSplitKBaseBlock,
    const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class MatMulUnAlignedSingleCoreSplitKKernel {
    struct UnAlignedSingleCoreSplitKParams {
        int nd2nzFlag;
        GM_ADDR alignedworkspaceGM;
        GM_ADDR castAddr;
        uint64_t vIndex;
        uint64_t alignedN;
        uint64_t coreSizeNum;
        uint64_t offset;
        uint64_t alignedOriM;
        uint64_t alignedOriN;
        uint64_t alignedKaSize;
        uint64_t alignedKbSize;
        bool isTransposeAIn;
        bool isTransposeBIn;
        bool nd2nzA;
        bool nd2nzB;
        uint64_t inputDtypeSize;
        GM_ADDR aGM;
        GM_ADDR bGM;
        GM_ADDR cGM;
        uint64_t baseAN;
        uint64_t baseAD;
        uint64_t baseBN;
        uint64_t baseBD;
        // A B矩阵都是对齐矩阵
    };

public:
    __aicore__ inline MatMulUnAlignedSingleCoreSplitKKernel() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM, const MatmulTilingData *matmulTilingData, TPipe *pipe);
    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM);
    __aicore__ inline void Process();

    __aicore__ inline void End()
    {
        if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
            mmb_.End();
        } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
            mma_.End();
        } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
            mmab_.End();
        }
    }

protected:
    using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
    using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
    using cType = MatmulType<C_TYPE::pos, C_TYPE::format, float, C_TYPE::isTrans>;
    MatMulBaseKernelSingleCoreSplitK<aType, B_TYPE, cType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mma_;
    MatMulBaseKernelSingleCoreSplitK<A_TYPE, bType, cType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mmb_;
    MatMulBaseKernelSingleCoreSplitK<aType, bType, cType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mmab_;
    GlobalTensor<float> matmulOutput_;
    GlobalTensor<typename C_TYPE::T> castCGm_;
    TPipe *pipe_;
    TBuf<> ubBuf_;
    UnAlignedSingleCoreSplitKParams innerParams_;
    const MatmulTilingData *matmulTilingData_;

private:
    __aicore__ inline void ProcessNDtoNZ();
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM);
    __aicore__ inline void ProcessRemovePaddingImpl();
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulUnAlignedSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(GM_ADDR aGM,
    GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const MatmulTilingData *matmulTilingData, TPipe *pipe)
{
    matmulTilingData_ = matmulTilingData;
    innerParams_.isTransposeAIn = matmulTilingData_->matmulRunInfo.transA;
    innerParams_.isTransposeBIn = matmulTilingData_->matmulRunInfo.transB;
    innerParams_.nd2nzA = matmulTilingData_->matmulRunInfo.nd2nzA;
    innerParams_.nd2nzB = matmulTilingData_->matmulRunInfo.nd2nzB;
    innerParams_.baseAN = matmulTilingData->baseAN;
    innerParams_.baseAD = matmulTilingData->baseAD;
    innerParams_.baseBN = matmulTilingData->baseBN;
    innerParams_.baseBD = matmulTilingData->baseBD;

    InitInputs(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);

    pipe_ = pipe;
    pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);

    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
            return;
        }

        if constexpr (sizeof(typename C_TYPE::T) == sizeof(float)) {
            innerParams_.castAddr = innerParams_.cGM;
        }

        if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
            mmb_.UnAlignedInit(innerParams_.aGM, innerParams_.alignedworkspaceGM, innerParams_.castAddr, biasGM,
                offsetWGM, workspaceGM, matmulTilingData, pipe_);
        } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
            mma_.UnAlignedInit(innerParams_.alignedworkspaceGM, innerParams_.bGM, innerParams_.castAddr, biasGM,
                offsetWGM, workspaceGM, matmulTilingData, pipe_);
        } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
            mmab_.UnAlignedInit(innerParams_.alignedworkspaceGM,
                innerParams_.alignedworkspaceGM +
                innerParams_.alignedOriM * innerParams_.alignedKaSize * innerParams_.inputDtypeSize,
                innerParams_.castAddr, biasGM, offsetWGM, workspaceGM, matmulTilingData, pipe_);
        }
        return;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulUnAlignedSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::InitInputs(GM_ADDR aGM,
    GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    innerParams_.alignedworkspaceGM = reinterpret_cast<GM_ADDR>(
        ((reinterpret_cast<uint64_t>(workspaceGM + MAX_BLOCK_NUM * DEFAULT_BLOCK_LEN * sizeof(int32_t)) + 511) / 512) *
        512);
    innerParams_.aGM = aGM;
    innerParams_.bGM = bGM;
    innerParams_.cGM = cGM;

    using A_T = typename A_TYPE::T;
    innerParams_.inputDtypeSize = sizeof(A_T);
    uint64_t c0Size;
    GetSizeC0<A_T>(c0Size);
    innerParams_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, ALIGNED_H) * ALIGNED_H;
    innerParams_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, c0Size) * c0Size;
    innerParams_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, c0Size) * c0Size;
    innerParams_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, ALIGNED_H) * ALIGNED_H;
    // A B矩阵都是对齐矩阵
    if (innerParams_.isTransposeAIn) {
        innerParams_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, c0Size) * c0Size;
        innerParams_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, ALIGNED_H) * ALIGNED_H;
    }
    if (innerParams_.isTransposeBIn) {
        innerParams_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, ALIGNED_H) * ALIGNED_H;
        innerParams_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, c0Size) * c0Size;
    }

    innerParams_.castAddr = innerParams_.alignedworkspaceGM;
    if (innerParams_.nd2nzA) {
        innerParams_.nd2nzFlag = ND2NZ_SELECT::ONLY_A;
        innerParams_.castAddr = innerParams_.alignedworkspaceGM +
            innerParams_.alignedOriM * innerParams_.alignedKaSize * innerParams_.inputDtypeSize;
    }
    if (innerParams_.nd2nzB) {
        innerParams_.nd2nzFlag = ND2NZ_SELECT::ONLY_B;
        innerParams_.castAddr = innerParams_.alignedworkspaceGM +
            innerParams_.alignedKbSize * innerParams_.alignedOriN * innerParams_.inputDtypeSize;
    }
    if (innerParams_.nd2nzA && innerParams_.nd2nzB) {
        bool isAFullLoad = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseM) * matmulTilingData_->matmulTiling.baseK *
            matmulTilingData_->matmulTiling.depthA1 >=
            innerParams_.alignedOriM * innerParams_.alignedKaSize;
        bool isBFullLoad = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseN) * matmulTilingData_->matmulTiling.baseK *
            matmulTilingData_->matmulTiling.depthB1 >=
            innerParams_.alignedOriN * innerParams_.alignedKbSize;
        if (isAFullLoad) {
            innerParams_.nd2nzFlag = ND2NZ_SELECT::ONLY_B;
            innerParams_.castAddr = innerParams_.alignedworkspaceGM +
                innerParams_.alignedOriN * innerParams_.alignedKbSize * innerParams_.inputDtypeSize;
        } else if (isBFullLoad) {
            innerParams_.nd2nzFlag = ND2NZ_SELECT::ONLY_A;
            innerParams_.castAddr = innerParams_.alignedworkspaceGM +
                innerParams_.alignedOriM * innerParams_.alignedKaSize * innerParams_.inputDtypeSize;
        } else {
            innerParams_.nd2nzFlag = ND2NZ_SELECT::BOTH_AB;
            innerParams_.castAddr =
                innerParams_.alignedworkspaceGM + (innerParams_.alignedOriM + innerParams_.alignedOriN) *
                innerParams_.alignedKaSize * innerParams_.inputDtypeSize;
        }
    }
    if ASCEND_IS_AIV {
        // Clear gm
        innerParams_.vIndex = GetBlockIdx();
        if (innerParams_.vIndex >= (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
            return;
        }
        if constexpr (sizeof(typename C_TYPE::T) == sizeof(float)) {
            return;
        }
        uint64_t totalSize = static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) * matmulTilingData_->matmulTiling.N;
        uint64_t coreSize = totalSize / (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO); // need to align
        innerParams_.coreSizeNum = coreSize;
        innerParams_.offset = innerParams_.vIndex * coreSize;
        if (innerParams_.vIndex == matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO - 1) {
            // 尾块数据量
            innerParams_.coreSizeNum =
                totalSize - (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO - 1) * coreSize;
        }
        if (static_cast<uint64_t>(matmulTilingData_->matmulTiling.N) * DATA_SIZE_FP32 % ALIGN_BYTE != 0) {
            innerParams_.alignedN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, 64) * 64;
            matmulOutput_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(innerParams_.castAddr),
                static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) * innerParams_.alignedN);
            castCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ typename C_TYPE::T *>(cGM),
                static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) * matmulTilingData_->matmulTiling.N);
        }
        return;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulUnAlignedSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE,
    MM_CFG>::UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
    GM_ADDR workspaceGM)
{
    InitInputs(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);

    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
            return;
        }
        if constexpr (sizeof(typename A_TYPE::T) == sizeof(float)) {
            innerParams_.castAddr = innerParams_.cGM;
        }

        if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
            mmb_.UpdateGlobalTensor(innerParams_.aGM, innerParams_.alignedworkspaceGM, innerParams_.castAddr, biasGM,
                offsetWGM, workspaceGM);
        } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
            mma_.UpdateGlobalTensor(innerParams_.alignedworkspaceGM, innerParams_.bGM, innerParams_.castAddr, biasGM,
                offsetWGM, workspaceGM);
        } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
            mmab_.UpdateGlobalTensor(innerParams_.alignedworkspaceGM,
                innerParams_.alignedworkspaceGM +
                innerParams_.alignedOriM * innerParams_.alignedKaSize * innerParams_.inputDtypeSize,
                innerParams_.castAddr, biasGM, offsetWGM, workspaceGM);
        }
        return;
    }
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulUnAlignedSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::ProcessNDtoNZ()
{
    // ND2NZ
    if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
        MatrixBtoNZV2<typename B_TYPE::T>(innerParams_.alignedworkspaceGM, innerParams_.bGM, matmulTilingData_->matmulTiling, innerParams_.isTransposeBIn, ubBuf_,
                                          innerParams_.baseBN, innerParams_.baseBD);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
        MatrixAtoNZV2<typename A_TYPE::T>(innerParams_.alignedworkspaceGM, innerParams_.aGM, matmulTilingData_->matmulTiling, innerParams_.isTransposeAIn, ubBuf_,
                                          innerParams_.baseAN, innerParams_.baseAD);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
        MatrixAtoNZV2<typename A_TYPE::T>(innerParams_.alignedworkspaceGM, innerParams_.aGM, matmulTilingData_->matmulTiling, innerParams_.isTransposeAIn, ubBuf_,
                                          innerParams_.baseAN, innerParams_.baseAD);
        event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        set_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
        MatrixBtoNZV2<typename B_TYPE::T>(innerParams_.alignedworkspaceGM +
                innerParams_.alignedOriM * innerParams_.alignedKaSize * innerParams_.inputDtypeSize, innerParams_.bGM, matmulTilingData_->matmulTiling, innerParams_.isTransposeBIn, ubBuf_,
                                          innerParams_.baseBN, innerParams_.baseBD);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulUnAlignedSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::ProcessRemovePaddingImpl()
{
    if (static_cast<uint64_t>(matmulTilingData_->matmulTiling.N) * DATA_SIZE_FP32 % ALIGN_BYTE != 0) {
        uint64_t splitM = matmulTilingData_->matmulTiling.M / (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO);
        uint64_t coreMSize = splitM;
        if (matmulTilingData_->matmulTiling.M < (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
            splitM = 1;
            if (innerParams_.vIndex * splitM >= matmulTilingData_->matmulTiling.M) {
                pipe_barrier(PIPE_ALL);
                return;
            }
            coreMSize = splitM;
        } else {
            if (innerParams_.vIndex == matmulTilingData_->matmulTiling.usedCoreNum * 2 - 1) {
                coreMSize = matmulTilingData_->matmulTiling.M - coreMSize * innerParams_.vIndex;
            }
        }
        RemovePaddingImpl<float, typename C_TYPE::T>(
            castCGm_[innerParams_.vIndex * splitM * static_cast<uint64_t>(matmulTilingData_->matmulTiling.N)],
            matmulOutput_[innerParams_.vIndex * splitM * innerParams_.alignedN], coreMSize, innerParams_.alignedN,
            matmulTilingData_->matmulTiling.N, ubBuf_);
    } else {
        UnAlignedCast32to16V220(reinterpret_cast<__gm__ typename C_TYPE::T *>(innerParams_.cGM) + innerParams_.offset,
            reinterpret_cast<__gm__ float *>(innerParams_.castAddr) + innerParams_.offset, 0, innerParams_.coreSizeNum,
            ubBuf_);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulUnAlignedSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process()
{
    using C_T = typename C_TYPE::T;
    if ASCEND_IS_AIV {
        if (GetBlockIdx() >= (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
            NotifyEvent<PIPE_MTE3>(6);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        ProcessNDtoNZ();

        SyncAll();
        NotifyEvent<PIPE_MTE3>(6);
        PipeBarrier<PIPE_ALL>();
        if constexpr (sizeof(C_T) == sizeof(float)) {
            return;
        }
        WaitFlagDevLocal(5);
        SyncAll();
        PipeBarrier<PIPE_ALL>();

        ProcessRemovePaddingImpl();

        PipeBarrier<PIPE_ALL>();
        return;
    }

    if ASCEND_IS_AIC {
        WaitFlagDevLocal(6);
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            if constexpr (sizeof(C_T) != sizeof(float)) {
                NotifyEvent<PIPE_FIX>(5);
            }
#endif
            PipeBarrier<PIPE_ALL>();
            return;
        }

        if (innerParams_.nd2nzFlag == 2) {
            mmb_.UnAlignedProcess();
        } else if (innerParams_.nd2nzFlag == 1) {
            mma_.UnAlignedProcess();
        } else if (innerParams_.nd2nzFlag == 3) {
            mmab_.UnAlignedProcess();
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        if constexpr (sizeof(C_T) != sizeof(float)) {
            NotifyEvent<PIPE_FIX>(5);
        }
#endif
        PipeBarrier<PIPE_ALL>();
        return;
    }
}
}
#endif // MM_STRESS_DETECT_KERNEL_H
