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
 * \file mat_mul_block_stress_detect.h
 * \brief
 */
#ifndef MM_STRESS_DETECT_BLOCK_H
#define MM_STRESS_DETECT_BLOCK_H

#include "detect_mat_mul.h"

using namespace AscendC;
using namespace matmul;

namespace MatmulStressDetect {

struct BlockOffset {
    uint64_t offsetA = 0;
    uint64_t offsetB = 0;
    uint64_t offsetC = 0;
    uint64_t offsetBias = 0;
};

struct BaseBlockArgs {
    uint64_t index;
    uint64_t mCnt;
    uint64_t nCnt;
    uint64_t totalTileCnt;
    uint64_t blockBaseM;
    uint64_t blockBaseN;
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t nBaseTail;
    uint64_t mBaseTail;
    uint64_t mTileCntL2;
    uint64_t nTileCntL2;
    uint64_t mCntTail;
    uint64_t nCntTail;
    uint64_t mTotalCnt;
    uint64_t nTotalCnt;
    uint64_t round;
    uint64_t realRound;
    uint64_t preCoreNum;
    uint64_t mCntUse;
    uint64_t nCntUse;
    uint32_t rowOrder;
    uint64_t blockIdxStart;
    uint64_t blockIdxEnd;

    uint64_t mTileAddrOffset;
    uint64_t nTileAddrOffset;

    uint64_t c0Size;
    uint64_t alignedOriM;
    uint64_t alignedOriN;
    uint64_t alignedKaSize;
    uint64_t alignedKbSize;

    bool isTransposeA;
    bool isTransposeB;
    uint32_t isHf32;
};

__aicore__ inline uint64_t MMLcm(uint64_t m, uint64_t n) {
    if (m == 0 || n == 0) {
        return 0; // 处理输入为0的情况
    }
    uint64_t total = m * n;
    uint64_t tmp = 0;
    while (n != 0) {
        tmp = m % n;
        m = n;
        n = tmp;
    }
    return total / m;
}

class MatmulBaseBlock {
public:
    __aicore__ inline MatmulBaseBlock() {}
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void *tilingData);
    __aicore__ inline void UpdateBlockCnt(uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void InitBlockIndex(uint64_t index);
    __aicore__ inline void UpdateBlockParams(uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void UpdateBlockIndex();
    __aicore__ inline void UpdateBasicIndex(const uint64_t roundIdx);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset(uint64_t mTileIndex, uint64_t nTileIndex);

public:
    BlockOffset offset_;
    BaseBlockArgs params_;
    const MatmulTilingData *matmulTilingData_;
    bool indexInit_ = false;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulBaseBlock::Init(const void *tilingData)
{
    matmulTilingData_ = static_cast<const MatmulTilingData *>(tilingData);
    const L2cacheTilePara &tilingL2 = matmulTilingData_->tileL2cacheTiling;

    params_.isTransposeA = matmulTilingData_->matmulRunInfo.transA;
    params_.isTransposeB = matmulTilingData_->matmulRunInfo.transB;
    params_.isHf32 = matmulTilingData_->matmulRunInfo.isHf32;

    params_.blockBaseM = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseM);
    params_.blockBaseN = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseN);
    params_.mTileCntL2 = static_cast<uint64_t>(tilingL2.mTileCntL2); // M方向的Tile份数
    params_.nTileCntL2 = static_cast<uint64_t>(tilingL2.nTileCntL2); // N方向的Tile份数
    params_.mTotalCnt =
        (static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) + matmulTilingData_->matmulTiling.singleCoreM - 1) /
        matmulTilingData_->matmulTiling.singleCoreM; // 总的m方向base块个数
    params_.nTotalCnt =
        (static_cast<uint64_t>(matmulTilingData_->matmulTiling.N) + matmulTilingData_->matmulTiling.singleCoreN - 1) /
        matmulTilingData_->matmulTiling.singleCoreN; // 总的n方向base块个数
    params_.nBaseTail =
        static_cast<uint64_t>(matmulTilingData_->matmulTiling.N) - (params_.nTotalCnt - 1) *
        matmulTilingData_->matmulTiling.singleCoreN; // n方向上的base尾块
    params_.mBaseTail =
        static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) - (params_.mTotalCnt - 1) *
        matmulTilingData_->matmulTiling.singleCoreM; // m方向上的base尾块
    params_.singleCoreM = 0;
    params_.singleCoreN = 0;
    params_.blockIdxStart = 0;
    params_.blockIdxEnd = 0;
    params_.index = 0;
    // mCnt和nCnt需要添加约束，否则可能会地址超限，当前tiling保证mTileCntL2和nTileCntL2合法性
    // 需要保证mTileCntL2和nTileCntL2的切分策略正好满足整块+尾块的处理
    params_.mCnt = (params_.mTotalCnt + params_.mTileCntL2 - 1) / params_.mTileCntL2; // 每一份mTile包含的base块个数
    params_.nCnt = (params_.nTotalCnt + params_.nTileCntL2 - 1) / params_.nTileCntL2; // 每一份nTile包含的base块个数
    if (tilingL2.mTileBlock > 0 && tilingL2.nTileBlock > 0) {
        params_.mCnt = tilingL2.mTileBlock;
        params_.nCnt = tilingL2.nTileBlock;
    }
    params_.totalTileCnt = params_.mCnt * params_.nCnt;

    params_.mCntTail = params_.mTotalCnt - (params_.mTileCntL2 - 1) * params_.mCnt; // M方向上mTile尾块里的base块的个数
    params_.nCntTail = params_.nTotalCnt - (params_.nTileCntL2 - 1) * params_.nCnt; // M方向上nTile尾块里的base块的个数
    params_.round = (params_.totalTileCnt + matmulTilingData_->matmulTiling.usedCoreNum - 1) /
        matmulTilingData_->matmulTiling.usedCoreNum;
    params_.realRound = 0;
    params_.preCoreNum = params_.totalTileCnt % matmulTilingData_->matmulTiling.usedCoreNum;
    params_.mCntUse = params_.mCnt;
    params_.nCntUse = params_.nCnt;
    // 当B矩阵比A矩阵大时，列优先输出能减少数据的重复替换

    params_.rowOrder = tilingL2.calOrder;

    params_.c0Size = 0;
    using A_T = typename A_TYPE::T;
    GetSizeC0<A_T>(params_.c0Size);
    params_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, ALIGNED_H) * ALIGNED_H;
    params_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, params_.c0Size) * params_.c0Size;
    params_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, params_.c0Size) * params_.c0Size;
    params_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, ALIGNED_H) * ALIGNED_H;
    // A B矩阵都是对齐矩阵
    if (params_.isTransposeA) {
      params_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, params_.c0Size) * params_.c0Size;
      params_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, ALIGNED_H) * ALIGNED_H;
    }
    if (params_.isTransposeB) {
      params_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, ALIGNED_H) * ALIGNED_H;
      params_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, params_.c0Size) * params_.c0Size;
    }
}

__aicore__ inline void MatmulBaseBlock::UpdateBlockCnt(uint64_t mTileIndex, uint64_t nTileIndex)
{
    params_.mTileAddrOffset = mTileIndex * params_.mCnt * matmulTilingData_->matmulTiling.singleCoreM;
    params_.nTileAddrOffset = nTileIndex * params_.nCnt * matmulTilingData_->matmulTiling.singleCoreN;

    if ((mTileIndex == (params_.mTileCntL2 - 1)) && (nTileIndex == (params_.nTileCntL2 - 1))) {
        params_.totalTileCnt = params_.mCntTail * params_.nCntTail;
        params_.mCntUse = params_.mCntTail;
        params_.nCntUse = params_.nCntTail;
    } else if (mTileIndex == (params_.mTileCntL2 - 1)) {
        params_.totalTileCnt = params_.mCntTail * params_.nCnt;
        params_.mCntUse = params_.mCntTail;
        params_.nCntUse = params_.nCnt;
    } else if (nTileIndex == (params_.nTileCntL2 - 1)) {
        params_.totalTileCnt = params_.mCnt * params_.nCntTail;
        params_.mCntUse = params_.mCnt;
        params_.nCntUse = params_.nCntTail;
    } else {
        params_.totalTileCnt = params_.mCnt * params_.nCnt;
        params_.mCntUse = params_.mCnt;
        params_.nCntUse = params_.nCnt;
    }

    params_.round = DivCeil(params_.totalTileCnt, static_cast<uint64_t>(matmulTilingData_->matmulTiling.usedCoreNum));
    params_.preCoreNum = params_.totalTileCnt % matmulTilingData_->matmulTiling.usedCoreNum;
    if (params_.preCoreNum == 0) {
        params_.preCoreNum = static_cast<uint64_t>(matmulTilingData_->matmulTiling.usedCoreNum);
    }
}

__aicore__ inline void MatmulBaseBlock::UpdateBasicIndex(const uint64_t roundIdx)
{
    uint64_t newBlockIdx = (block_idx + matmulTilingData_->matmulTiling.usedCoreNum - params_.blockIdxStart) %
        matmulTilingData_->matmulTiling.usedCoreNum +
        roundIdx * matmulTilingData_->matmulTiling.usedCoreNum;
    uint64_t mIdx = newBlockIdx % params_.mCntUse;
    uint64_t nIdx = 0;
    if (params_.mCntUse != 0 && params_.nCntUse != 0) {
        nIdx = (newBlockIdx + newBlockIdx / MMLcm(params_.mCntUse, params_.nCntUse)) % params_.nCntUse;
    }
    params_.index = mIdx * params_.nCntUse + nIdx;
}


__aicore__ inline void MatmulBaseBlock::InitBlockIndex(uint64_t index)
{
    if (indexInit_) {
        params_.blockIdxStart = params_.blockIdxEnd; // 开始运算时，首核的索引
    } else {
        params_.blockIdxStart =
            index * params_.preCoreNum % matmulTilingData_->matmulTiling.usedCoreNum; // 开始运算时，首核的索引
        indexInit_ = true;
    }
    params_.blockIdxEnd = (params_.blockIdxStart + params_.preCoreNum) %
        matmulTilingData_->matmulTiling.usedCoreNum; // 结束运算时，尾核的索引
    uint64_t indexStart = params_.blockIdxStart;
    uint64_t indexEnd = params_.blockIdxEnd;

    // 利用roudCnt来解决尾块负载均衡问题
    if (indexStart < indexEnd) {
        // 正常排序, preCore在整个Cores的中间
        if (block_idx < indexStart) {
            params_.index = block_idx * (params_.round - 1);
            params_.realRound = params_.round - 1;
        } else if (block_idx < indexEnd) {
            params_.index = indexStart * (params_.round - 1) + (block_idx - indexStart) * params_.round;
            params_.realRound = params_.round;
        } else {
            params_.index = (indexStart * (params_.round - 1) + params_.preCoreNum * params_.round +
                (block_idx - indexEnd) * (params_.round - 1));
            params_.realRound = params_.round - 1;
        }
        if (params_.rowOrder == COL_FIRST) {
            // 列优先分配
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    } else if (indexEnd < indexStart) {
        // indexEnd会翻转
        if (block_idx < indexEnd) {
            params_.index = block_idx * params_.round;
            params_.realRound = params_.round;
        } else if (block_idx < indexStart) {
            params_.index = indexEnd * params_.round + (block_idx - indexEnd) * (params_.round - 1);
            params_.realRound = params_.round - 1;
        } else {
            params_.index = (indexEnd * params_.round + (indexStart - indexEnd) * (params_.round - 1) +
                (block_idx - indexStart) * params_.round);
            params_.realRound = params_.round;
        }
        if (params_.rowOrder == COL_FIRST) {
            // 列优先分配
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    } else {
        // 不存在尾核，基本块对齐
        params_.index = block_idx * params_.round;
        params_.realRound = params_.round;
        if (params_.rowOrder == COL_FIRST) {
            // 列优先分配
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    }
}

__aicore__ inline void MatmulBaseBlock::UpdateBlockParams(uint64_t mTileIndex, uint64_t nTileIndex)
{
    if ((mTileIndex == (params_.mTileCntL2 - 1)) && (nTileIndex == (params_.nTileCntL2 - 1)) &&
        (params_.index == (params_.totalTileCnt - 1))) {
        params_.singleCoreM = params_.mBaseTail;
        params_.singleCoreN = params_.nBaseTail;
    } else if ((mTileIndex == (params_.mTileCntL2 - 1)) && (params_.index >= (params_.mCntUse - 1) * params_.nCntUse)) {
        params_.singleCoreM = params_.mBaseTail;
        params_.singleCoreN = matmulTilingData_->matmulTiling.singleCoreN;
    } else if ((nTileIndex == (params_.nTileCntL2 - 1)) && ((params_.index + 1) % params_.nCntUse == 0)) {
        params_.singleCoreM = matmulTilingData_->matmulTiling.singleCoreM;
        params_.singleCoreN = params_.nBaseTail;
    } else {
        params_.singleCoreM = matmulTilingData_->matmulTiling.singleCoreM;
        params_.singleCoreN = matmulTilingData_->matmulTiling.singleCoreN;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulBaseBlock::CalcGMOffset(uint64_t mTileIndex, uint64_t nTileIndex)
{
    uint64_t mCntIndex = params_.index / params_.nCntUse;
    uint64_t nCntIndex = params_.index % params_.nCntUse;
    if constexpr (A_TYPE::format == CubeFormat::ND) {
        if (params_.isTransposeA) {
            offset_.offsetA = mCntIndex * matmulTilingData_->matmulTiling.singleCoreM + params_.mTileAddrOffset;
        } else {
            offset_.offsetA = mCntIndex * matmulTilingData_->matmulTiling.singleCoreM *
                matmulTilingData_->matmulTiling.Ka + params_.mTileAddrOffset * matmulTilingData_->matmulTiling.Ka;
        }
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        if (params_.isTransposeA) {
            offset_.offsetA = mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * params_.alignedKaSize +
                params_.mTileAddrOffset * params_.alignedKaSize;
        } else {
            offset_.offsetA = mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * params_.c0Size +
                params_.mTileAddrOffset * params_.c0Size;
        }
    }
    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if (params_.isTransposeB) {
            offset_.offsetB = nCntIndex * matmulTilingData_->matmulTiling.singleCoreN *
                matmulTilingData_->matmulTiling.Kb + params_.nTileAddrOffset * matmulTilingData_->matmulTiling.Kb;
        } else {
            offset_.offsetB = nCntIndex * matmulTilingData_->matmulTiling.singleCoreN + params_.nTileAddrOffset;
        }
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        if (params_.isTransposeB) {
            offset_.offsetB = nCntIndex * matmulTilingData_->matmulTiling.singleCoreN * params_.c0Size +
                params_.nTileAddrOffset * params_.c0Size;
        } else {
            offset_.offsetB = nCntIndex * matmulTilingData_->matmulTiling.singleCoreN * params_.alignedKbSize +
                params_.nTileAddrOffset * params_.alignedKbSize;
        }
    }
    if constexpr (C_TYPE::format == CubeFormat::ND) {
        offset_.offsetC = (nCntIndex * matmulTilingData_->matmulTiling.singleCoreN +
            mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * matmulTilingData_->matmulTiling.N +
            (params_.mTileAddrOffset * matmulTilingData_->matmulTiling.N + params_.nTileAddrOffset));
    } else {
        offset_.offsetC = (nCntIndex * matmulTilingData_->matmulTiling.singleCoreN * matmulTilingData_->matmulTiling.M +
            mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * params_.c0Size +
            (params_.mTileAddrOffset * params_.c0Size +
            params_.nTileAddrOffset * matmulTilingData_->matmulTiling.M));
    }
    if (matmulTilingData_->matmulTiling.isBias) {
        offset_.offsetBias = nCntIndex * matmulTilingData_->matmulTiling.singleCoreN + params_.nTileAddrOffset;
    }
}

__aicore__ inline void MatmulBaseBlock::UpdateBlockIndex()
{
    if (params_.rowOrder == ROW_FIRST) {
        params_.index += 1;
    } else if (params_.rowOrder == COL_FIRST) {
        params_.index += params_.nCntUse;
        if (params_.index >= params_.totalTileCnt) {
            params_.index = params_.index % params_.totalTileCnt + 1;
        }
    }
}

struct SingleCoreSplitKBaseBlockArgs {
    bool isTransposeA;
    bool isTransposeB;
    uint32_t isHf32;

    uint64_t mCnt;
    uint64_t nCnt;
    uint64_t kCnt;
    uint64_t mCoreTail;
    uint64_t nCoreTail;
    uint64_t kCoreTail;
    uint64_t loopK;

    uint64_t kTileL2;

    uint64_t innerBlockM;
    uint64_t innerBlockN;

    uint64_t innerLoopM;
    uint64_t innerLoopN;
    bool atomicAddFlag;

    uint64_t index;
    // M和N方向绑多核, 按照3*3个128*128的基本块
    uint64_t totalCnt;
    uint64_t round;
    uint64_t realRound;
    uint64_t preCoreNum;

    uint64_t mCoreUse;
    uint64_t nCoreUse;
    uint64_t kCoreUse;

    uint64_t mIndex;
    uint64_t nIndex;

    uint64_t innerSingleCoreM; // 增加
    uint64_t innerSingleCoreN; // 增加

    uint64_t outNAlign;
    uint64_t c0Size;
    uint64_t alignedOriM;
    uint64_t alignedOriN;
    uint64_t alignedKaSize;
    uint64_t alignedKbSize;
    uint32_t rowOrder; // 用于区分是否进入n轴错峰
};

class MatmulSingleCoreSplitKBaseBlock {
public:
    __aicore__ inline MatmulSingleCoreSplitKBaseBlock() {}
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const MatmulTilingData *matmulTilingData);
    __aicore__ inline void UpdateBlockCnt();
    __aicore__ inline void InitBlockIndex();
    __aicore__ inline void UpdateBlockParams(uint64_t innerMIndex, uint64_t kIndex);
    __aicore__ inline void UpdateBlockIndex();
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset(uint64_t innerMIndex, uint64_t kIndex, uint64_t innerNIndex);

public:
    BlockOffset offset_;
    SingleCoreSplitKBaseBlockArgs params_;
    const MatmulTilingData *matmulTilingData_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulSingleCoreSplitKBaseBlock::Init(const MatmulTilingData *matmulTilingData)
{
    matmulTilingData_ = matmulTilingData;
    const L2cacheTilePara &tilingL2 = matmulTilingData_->tileL2cacheTiling;
    params_.rowOrder = tilingL2.calOrder;
    params_.isTransposeA = matmulTilingData_->matmulRunInfo.transA;
    params_.isTransposeB = matmulTilingData_->matmulRunInfo.transB;
    params_.isHf32 = matmulTilingData_->matmulRunInfo.isHf32;

    params_.mCnt = MMV3DivCeil(matmulTilingData_->matmulTiling.M, matmulTilingData_->matmulTiling.singleCoreM);
    params_.nCnt = MMV3DivCeil(matmulTilingData_->matmulTiling.N, matmulTilingData_->matmulTiling.singleCoreN);
    params_.kCnt = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, matmulTilingData_->matmulTiling.singleCoreK);
    params_.mCoreTail = static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) -
                        (params_.mCnt - 1) * matmulTilingData_->matmulTiling.singleCoreM;
    params_.nCoreTail = static_cast<uint64_t>(matmulTilingData_->matmulTiling.N) -
                        (params_.nCnt - 1) * matmulTilingData_->matmulTiling.singleCoreN;
    params_.kCoreTail = static_cast<uint64_t>(matmulTilingData_->matmulTiling.Ka) -
                        (params_.kCnt - 1) * matmulTilingData_->matmulTiling.singleCoreK;
    params_.loopK = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, matmulTilingData_->matmulTiling.singleCoreK);
    params_.innerBlockM =
        static_cast<uint64_t>(matmulTilingData_->matmulTiling.stepM) * matmulTilingData_->matmulTiling.baseM;
    params_.innerBlockN =
        static_cast<uint64_t>(matmulTilingData_->matmulTiling.stepN) * matmulTilingData_->matmulTiling.baseN;
    params_.atomicAddFlag = false;
    // 记录3*3的基本块的位置
    params_.index = 0;
    // M和N方向绑多核, 按照3*3个128*128的基本块
    params_.totalCnt = params_.mCnt * params_.nCnt;
    params_.round = DivCeil(params_.totalCnt, matmulTilingData_->matmulTiling.usedCoreNum);
    params_.realRound = 0;
    params_.preCoreNum = params_.totalCnt % matmulTilingData_->matmulTiling.usedCoreNum;
    if (params_.preCoreNum == 0) {
        params_.preCoreNum = matmulTilingData_->matmulTiling.usedCoreNum;
    }

    params_.mCoreUse = matmulTilingData_->matmulTiling.singleCoreM;
    params_.nCoreUse = matmulTilingData_->matmulTiling.singleCoreN;
    params_.kCoreUse = matmulTilingData_->matmulTiling.singleCoreK;

    uint64_t cTypeSize = 64; // 64 means 256Byte
    params_.outNAlign = MMV3DivCeil(matmulTilingData_->matmulTiling.N, cTypeSize) * cTypeSize;
    using A_T = typename A_TYPE::T;
    if constexpr (sizeof(A_T) == sizeof(float)) {
        params_.outNAlign = matmulTilingData_->matmulTiling.N;
    }

    GetSizeC0<A_T>(params_.c0Size);
    params_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, ALIGNED_H) * ALIGNED_H;
    params_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, params_.c0Size) * params_.c0Size;
    params_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, params_.c0Size) * params_.c0Size;
    params_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, ALIGNED_H) * ALIGNED_H;
    // A B矩阵都是对齐矩阵
    if (params_.isTransposeA) {
        params_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, params_.c0Size) * params_.c0Size;
        params_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, ALIGNED_H) * ALIGNED_H;
    }
    if (params_.isTransposeB) {
        params_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, ALIGNED_H) * ALIGNED_H;
        params_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, params_.c0Size) * params_.c0Size;
    }
}

__aicore__ inline void MatmulSingleCoreSplitKBaseBlock::UpdateBlockCnt()
{
    params_.mIndex = params_.index % params_.mCnt;
    params_.nIndex = params_.index / params_.mCnt;
    if (params_.index == (params_.totalCnt - 1)) {
        // 最后一块是尾块
        params_.mCoreUse = params_.mCoreTail;
        params_.nCoreUse = params_.nCoreTail;
    } else if (params_.mIndex == (params_.mCnt - 1)) {
        // m方向尾块
        params_.mCoreUse = params_.mCoreTail;
        params_.nCoreUse = matmulTilingData_->matmulTiling.singleCoreN;
    } else if (params_.nIndex == (params_.nCnt - 1)) {
        // n方向尾块
        params_.mCoreUse = matmulTilingData_->matmulTiling.singleCoreM;
        params_.nCoreUse = params_.nCoreTail;
    } else {
        // 对齐整块
        params_.mCoreUse = matmulTilingData_->matmulTiling.singleCoreM;
        params_.nCoreUse = matmulTilingData_->matmulTiling.singleCoreN;
    }
    params_.innerLoopM = MMV3DivCeil(matmulTilingData_->matmulTiling.singleCoreM, params_.innerBlockM);
    params_.innerLoopN = MMV3DivCeil(matmulTilingData_->matmulTiling.singleCoreN, params_.innerBlockN);
    if (params_.mIndex == params_.mCnt - 1) {
        params_.innerLoopM = DivCeil(params_.mCoreTail, params_.innerBlockM);
    }
    if (params_.nIndex == params_.nCnt - 1) {
        params_.innerLoopN = DivCeil(params_.nCoreTail, params_.innerBlockN);
    }
    if (params_.rowOrder == 0) { // 单核切k中 l2IterateOrder 为默认值0时走原kernel
        params_.innerLoopN = 1;
    }
}

__aicore__ inline void MatmulSingleCoreSplitKBaseBlock::InitBlockIndex()
{
    if (block_idx < params_.preCoreNum) {
        params_.index = block_idx * params_.round;
        params_.realRound = params_.round;
    } else {
        params_.index = block_idx * (params_.round - 1) + params_.preCoreNum;
        params_.realRound = params_.round - 1;
    }
}

__aicore__ inline void MatmulSingleCoreSplitKBaseBlock::UpdateBlockParams(uint64_t innerMIndex, uint64_t kIndex)
{
    params_.innerSingleCoreM = params_.innerBlockM;
    if (innerMIndex == params_.innerLoopM - 1) {
        params_.innerSingleCoreM = params_.mCoreUse - (params_.innerLoopM - 1) * params_.innerBlockM;
    }
    if (kIndex == params_.loopK - 1) {
        params_.kCoreUse = params_.kCoreTail;
    } else {
        params_.kCoreUse = matmulTilingData_->matmulTiling.singleCoreK;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulSingleCoreSplitKBaseBlock::CalcGMOffset(uint64_t innerMIndex, uint64_t kIndex,
    uint64_t innerNIndex)
{
    uint64_t innerNShiftIndex = (block_idx + innerNIndex) % params_.innerLoopN;
    params_.innerSingleCoreN = params_.innerBlockN;
    if (innerNShiftIndex == params_.innerLoopN - 1) {
        params_.innerSingleCoreN = params_.nCoreUse - innerNShiftIndex * params_.innerBlockN;
    }
    if constexpr (A_TYPE::format == CubeFormat::ND) {
        if (params_.isTransposeA) {
            offset_.offsetA =
                (params_.mIndex * matmulTilingData_->matmulTiling.singleCoreM + innerMIndex * params_.innerBlockM +
                kIndex * matmulTilingData_->matmulTiling.singleCoreK *
                matmulTilingData_->matmulTiling.M);
        } else {
            offset_.offsetA =
                ((params_.mIndex * matmulTilingData_->matmulTiling.singleCoreM + innerMIndex * params_.innerBlockM) *
                matmulTilingData_->matmulTiling.Ka + kIndex * matmulTilingData_->matmulTiling.singleCoreK);
        }
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        if (params_.isTransposeA) {
            offset_.offsetA =
                ((params_.mIndex * matmulTilingData_->matmulTiling.singleCoreM + innerMIndex * params_.innerBlockM) *
                params_.alignedKaSize + kIndex * matmulTilingData_->matmulTiling.singleCoreK * params_.c0Size);
        } else {
            offset_.offsetA =
                ((params_.mIndex * matmulTilingData_->matmulTiling.singleCoreM + innerMIndex * params_.innerBlockM) *
                params_.c0Size + kIndex * matmulTilingData_->matmulTiling.singleCoreK * params_.alignedOriM);
        }
    }
    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if (params_.isTransposeB) {
            offset_.offsetB = (kIndex * matmulTilingData_->matmulTiling.singleCoreK +
                (params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN + innerNShiftIndex * params_.innerBlockN) *
                matmulTilingData_->matmulTiling.Kb);
        } else {
            offset_.offsetB = (params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN +
                innerNShiftIndex * params_.innerBlockN + kIndex *
                matmulTilingData_->matmulTiling.singleCoreK * matmulTilingData_->matmulTiling.N);
        }
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        if (params_.isTransposeB) {
            offset_.offsetB =
                (kIndex * matmulTilingData_->matmulTiling.singleCoreK * params_.alignedOriN +
                (params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN + innerNShiftIndex * params_.innerBlockN) *
                params_.c0Size);
        } else {
            offset_.offsetB = ((params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN +
                innerNShiftIndex * params_.innerBlockN) * params_.alignedKbSize +
                kIndex * matmulTilingData_->matmulTiling.singleCoreK * params_.c0Size);
        }
    }
    offset_.offsetC =
        ((params_.mIndex * matmulTilingData_->matmulTiling.singleCoreM + innerMIndex * params_.innerBlockM) *
        params_.outNAlign + params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN +
        innerNShiftIndex * params_.innerBlockN);
    if (A_TYPE::format == CubeFormat::ND && B_TYPE::format == CubeFormat::ND &&
        (matmulTilingData_->matmulTiling.N % ALIGN_128_BYTE == 0)) {
        offset_.offsetC =
            ((params_.mIndex * matmulTilingData_->matmulTiling.singleCoreM + innerMIndex * params_.innerBlockM) *
            matmulTilingData_->matmulTiling.N + params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN +
            innerNShiftIndex * params_.innerBlockN);
    }
    offset_.offsetBias = params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN;
}

__aicore__ inline void MatmulSingleCoreSplitKBaseBlock::UpdateBlockIndex()
{
    params_.index += 1;
}

}

#endif // MM_STRESS_DETECT_BLOCK_H