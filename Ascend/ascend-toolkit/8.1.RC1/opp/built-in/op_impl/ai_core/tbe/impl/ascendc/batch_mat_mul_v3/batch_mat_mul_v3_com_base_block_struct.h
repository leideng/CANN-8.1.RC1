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

/* !
 * \file batch_mat_mul_v3_com_base_block_struct.h
 * \brief
 */
#ifndef BATCH_MATMUL_V3_COM_BASE_BLOCK_STRUCT_H
#define BATCH_MATMUL_V3_COM_BASE_BLOCK_STRUCT_H

struct CommonKernelBlockOffset {
    uint64_t offsetA;
    uint64_t offsetB;
    uint64_t offsetC;
    uint64_t offsetBias;
};

struct CommonKernelBaseBlockArgs {
    uint64_t batchA1;
    uint64_t batchA2;
    uint64_t batchA3;
    uint64_t batchA4;
    uint64_t batchB1;
    uint64_t batchB2;
    uint64_t batchB3;
    uint64_t batchB4;
    uint64_t batchC1;
    uint64_t batchC2;
    uint64_t batchC3;
    uint64_t batchC4;
    uint64_t aBatchDimAll;
    uint64_t bBatchDimAll;
    uint64_t cBatchDimAll;

    uint64_t batchCnt;
    uint64_t mCnt;       // m方向上分多少个base块
    uint64_t nCnt;       // n方向上分多少个base块
    uint64_t nBaseTail;  // n方向上的baseN尾块
    uint64_t mBaseTail;  // m方向上的baseM尾块
    uint64_t mTileCntL2;
    uint64_t nTileCntL2;
    uint64_t mCntTail;
    uint64_t nCntTail;
    uint64_t mTileCnt;
    uint64_t nTileCnt;
    uint64_t totalCnt;   // 所有的基本块个数
    uint64_t totalTileCnt;
    uint64_t round;      // 每一个core最大做base块计算的次数
    uint64_t realRound;  // 单核做多少次base块计算
    uint64_t preCoreNum; // 从0core开始有多少个core会多做一次base块
    uint64_t mCntUse; // Tile块内m方向个数
    uint64_t nCntUse; // Tile块内n方向个数
    uint32_t rowOrder; // 0 错位分核 ROW_FIRST C矩阵行优先 COL_FIRST C矩阵列优先
    uint64_t blockIdxStart;
    uint64_t blockIdxEnd;
    uint64_t preTotalBlock;

    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t singleCoreK;

    uint64_t mTileAddrOffset; // Tile块m方向起始地址偏移
    uint64_t nTileAddrOffset; // Tile块n方向起始地址偏移
    uint64_t c0Size;
    uint64_t alignedOriM;
    uint64_t alignedOriN;
    uint64_t alignedKaSize;
    uint64_t alignedKbSize;

    uint64_t index;      // 当前block_idx的起始基本块Index，这个idex是按照先循环N，再循环M的次序
    bool isHf32;
    bool biasWithBatch;
    bool enableL2Cache;
};

#endif // BATCH_MATMUL_V3_COM_BASE_BLOCK_STRUCT_H