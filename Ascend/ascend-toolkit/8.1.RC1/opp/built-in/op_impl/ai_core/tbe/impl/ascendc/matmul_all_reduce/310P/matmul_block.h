/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file matmul_block.h
 * \brief
 */
#ifndef MC2_MATMUL_BLOCK_H
#define MC2_MATMUL_BLOCK_H

namespace AscendC {

constexpr uint32_t SHAPE_ALIGNED_SIZE = 16;

namespace MC2 {
struct BlockOffset {
    uint64_t offsetA;
    uint64_t offsetB;
    uint64_t offsetC;
    uint64_t offsetBias;
};

class MatmulBlock {
public:
    __aicore__ inline MatmulBlock() {}
    __aicore__ inline void Init(TCubeTiling& tiling, RCSTiling& cfg, L2cacheTilePara& tileL2cacheTiling);
    __aicore__ inline void UpdateBlockIndex();
    __aicore__ inline void InitBlockIndex();
    __aicore__ inline uint32_t LCM(uint32_t m, uint32_t n); // 计算最小公倍数
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset(int32_t mTileIndex = 0, int32_t nTileIndex = 0);
    __aicore__ inline void UpdateBlockCnt(uint32_t index, int32_t mTileIndex = 0, int32_t nTileIndex = 0);
    __aicore__ inline void UpdateL2cacheTail() {}

public:
    BlockOffset offset;
    uint32_t totalBlockCnt;  // C矩阵的全部基本块个数
    uint32_t blockCnt;       // 单核需要计算的基本块个数
    uint32_t blockIndex;     // 当前核需计算的基本块的起始block索引
    bool isAtomic;
    bool isTransposeA;
    bool isTransposeB;
    RCSTiling cfg;

    TCubeTiling tiling;
    bool isRowOrder;
    uint32_t preCoreNum;  // 需要多计算一次的核数
    uint32_t mBlockCnt;   // M方向的基本块个数
    uint32_t nBlockCnt;   // N方向的基本块个数
    uint32_t nBaseTail;   // N方向的尾块大小
    uint32_t mBaseTail;   // M方向的尾块大小

    uint32_t mBlockIndex;
    uint32_t nBlockIndex;
    uint32_t blockCurM;   // 当前block块M轴大小
    uint32_t blockCurN;   // 当前block块N轴大小
};

__aicore__ inline void MatmulBlock::Init(TCubeTiling& tiling, RCSTiling& cfg, L2cacheTilePara& tileL2cacheTiling)
{
    (void)tileL2cacheTiling;
    this->tiling = tiling;
    this->cfg = cfg;
    // 计算C矩阵可以切分的Block数，因为有尾块需要向上取整
    mBlockCnt = DivCeil(tiling.M, tiling.baseM);  //M方向分Base块个数
    nBlockCnt = DivCeil(tiling.N, tiling.baseN);  //N方向分Base块个数
    nBaseTail = tiling.N - (nBlockCnt - 1) * tiling.baseN;
    mBaseTail = tiling.M - (mBlockCnt - 1) * tiling.baseM;
    totalBlockCnt = mBlockCnt * nBlockCnt;

    // preCoreNum为不能被整除的部分，当作尾块分配到前preCoreNum个核中计算
    // blockCnt为总Block数/总核数向上取整
    // 10*29=290个block/20核=14余10，所以前10个核分15个block，后10个核分14个block
    preCoreNum = totalBlockCnt % tiling.usedCoreNum;
    if (preCoreNum == 0) {
        preCoreNum = tiling.usedCoreNum;
    }

    isRowOrder = true;
    // 5 is a multiple, When N is greater than 5 times M, block by column is used to improve performance
    if (tiling.N > 5 * tiling.M) {
        isRowOrder = false;
    }
    isTransposeA = cfg.isTransposeA > 0 ? true : false;
    isTransposeB = cfg.isTransposeB > 0 ? true : false;
    isAtomic = false;
    if (isTransposeA) {
        isAtomic = true;
    }
}

__aicore__ inline void MatmulBlock::UpdateBlockCnt(uint32_t index, int32_t mTileIndex, int32_t nTileIndex)
{
    (void)mTileIndex;
    (void)nTileIndex;
    blockCnt = totalBlockCnt / tiling.usedCoreNum;
    preCoreNum = totalBlockCnt % tiling.usedCoreNum;
    uint32_t offset = (tiling.usedCoreNum - preCoreNum) * index + block_idx;
    if ((offset % tiling.usedCoreNum) < preCoreNum) {
        blockCnt += 1;
    }
#if __CCE_AICORE__ == 220
    blockIndex = block_idx;
#else
    uint32_t startIdx = (preCoreNum * index) % tiling.usedCoreNum;
    uint32_t endIdx = (startIdx + preCoreNum) % tiling.usedCoreNum;
    if (startIdx > endIdx) {
        if (block_idx < endIdx) {
            blockIndex = block_idx * blockCnt;
        } else if (block_idx >= startIdx) {
            blockIndex = block_idx * blockCnt - (tiling.usedCoreNum - preCoreNum);
        } else {
            blockIndex = block_idx * blockCnt + endIdx;
        }
    } else {
        if (block_idx < startIdx) {
            blockIndex = block_idx * blockCnt;
        } else if (block_idx >= endIdx) {
            blockIndex = block_idx * blockCnt + preCoreNum;
        } else {
            blockIndex = block_idx * blockCnt - startIdx;
        }
    }

    uint32_t preTotalBlock = blockIndex;
    if (!isRowOrder) {
        blockIndex = preTotalBlock / mBlockCnt + preTotalBlock % mBlockCnt * nBlockCnt;
    }
#endif
}

__aicore__ inline void MatmulBlock::UpdateBlockIndex()
{
#if __CCE_AICORE__ == 220
    blockIndex += tiling.usedCoreNum;
#else
    // 完成一个block计算后，block需取下一个，按行取只需+1即可
    if (isRowOrder) {
        blockIndex = blockIndex + 1;
    } else {
        // 按列取，block不在最后一行，下一个block索引列不变行+1
        // 如果block在最后一行，下一个block索引是列+1行清零
        blockIndex = blockIndex + nBlockCnt;
        if (blockIndex >= totalBlockCnt) {
            blockIndex = blockIndex % totalBlockCnt + 1;
        }
    }
#endif
}

__aicore__ inline void MatmulBlock::InitBlockIndex()
{
    blockCnt = DivCeil(totalBlockCnt, tiling.usedCoreNum);
    if (block_idx >= preCoreNum) {
        blockCnt = blockCnt - 1;
    }
    preCoreNum = totalBlockCnt % tiling.usedCoreNum;
    if (preCoreNum == 0) {
        preCoreNum = tiling.usedCoreNum;
    }
    uint32_t preTotalBlock = 0;
    if (block_idx < preCoreNum) {
        if (isRowOrder) {
            // 第6个核的起始index为5*单核处理block数
            blockIndex = block_idx * blockCnt;
        } else {
            // 先计算按列排列前x个核的block个数，再重新按大Z格式计算本核的blockIndex
            // C矩阵有10*29=290个block，前5个核的总blcok数为5*15=75
            // 75/10=7列余5个数，因此第6个核的blockIndex为5*29+7(左侧7个block)
            preTotalBlock = block_idx * blockCnt;
            blockIndex = preTotalBlock / mBlockCnt + preTotalBlock % mBlockCnt * nBlockCnt;
        }
    } else {
        if (isRowOrder) {
            blockIndex = block_idx * blockCnt + preCoreNum;
        } else {
            preTotalBlock = block_idx * blockCnt + preCoreNum;
            blockIndex = preTotalBlock / mBlockCnt + preTotalBlock % mBlockCnt * nBlockCnt;
        }
    }
}

__aicore__ inline uint32_t MatmulBlock::LCM(uint32_t m, uint32_t n)
{
    uint32_t mn = m * n;
    uint32_t gcd = n; // 最大公约数
    while (m % n) {
        gcd = m % n;
        m = n;
        n = gcd;
    }
    return mn / gcd;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulBlock::CalcGMOffset(int32_t mTileIndex, int32_t nTileIndex)
{
    (void)mTileIndex;
    (void)nTileIndex;

#if __CCE_AICORE__ == 220
    uint32_t lcmCuts = LCM(mBlockCnt, nBlockCnt);
    mBlockIndex = blockIndex % mBlockCnt;
    nBlockIndex = (blockIndex + blockIndex / lcmCuts) % nBlockCnt;
#else
    mBlockIndex = blockIndex / nBlockCnt;  // 基本块所在的行索引
    nBlockIndex = blockIndex % nBlockCnt;  // 基本块所在的列索引
#endif

    auto alignedKa = AlignUp(tiling.Ka, SHAPE_ALIGNED_SIZE);
    auto alignedKb = AlignUp(tiling.Kb, SHAPE_ALIGNED_SIZE);

    if constexpr (A_TYPE::format == CubeFormat::ND) {
        if (isTransposeA) {
            offset.offsetA = mBlockIndex * tiling.baseM;
        } else {
            offset.offsetA = mBlockIndex * tiling.baseM * tiling.Ka;
        }
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        if (isTransposeA) {
            offset.offsetA = mBlockIndex * tiling.baseM * alignedKa;
        } else {
            offset.offsetA = mBlockIndex * tiling.baseM * BLOCK_CUBE;
        }
    }

    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if (isTransposeB) {
            offset.offsetB = nBlockIndex * tiling.baseN * tiling.Kb;
        } else {
            offset.offsetB = nBlockIndex * tiling.baseN;
        }
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        if (isTransposeB) {
            offset.offsetB = nBlockIndex * tiling.baseN * BLOCK_CUBE;
        } else {
            offset.offsetB = nBlockIndex * tiling.baseN * alignedKb;
        }
    }

    // C矩阵和BIAS只支持ND
    if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
        offset.offsetC = nBlockIndex * tiling.baseN + mBlockIndex * tiling.baseM * tiling.N;
    }
    if constexpr (BIAS_TYPE::format == CubeFormat::ND) {
        offset.offsetBias = nBlockIndex * tiling.baseN;
    }
}
}   // namespace mc2
}   // namespace Ascendc
#endif // MC2_MATMUL_BLOCK_H
