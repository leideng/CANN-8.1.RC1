/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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
 * \file swi_glu_quant_base.h
 * \brief
 */
#ifndef SWI_GLU_QUANT_BASE_H
#define SWI_GLU_QUANT_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace SwiGluQuantOpt {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t SPLIT_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t SWI_GLU_QUANT_EIGHT = 8;
constexpr uint32_t SWI_GLU_QUANT_THIRTY_TWO = 32;

constexpr float SWI_GLU_QUANT_INT8_SYM_SCALE = 127.0;
constexpr uint32_t MAX_VALUE_NUM = 8;
constexpr uint32_t SMOOTH_INDEX_UPBOUND = 65536;

// 单输入场景，一个tile需要的偏置参数
struct XxGluSingleTileOffsetParam {
    uint64_t splitVecGmOffset1; // 拼接的vector，第一个vector gm上的偏移
    uint64_t splitVecGmOffset2; // 拼接的vector，第er个vector gm上的偏移
    uint64_t tmpVecGmOffset;
};

enum class QuantType : uint8_t {
    STATIC_PER_TENSOR = 0,
    STATIC_PER_CHANNEL
};

class SwiGluQuantBase {
public:
    __aicore__ inline SwiGluQuantBase()
    {}

    __aicore__ inline void ParseTilingData(const SwiGluQuantTilingData *tilingData)
    {
        tilingData_.groupLen = tilingData->groupLen;            // group的长度
        tilingData_.rowLen = tilingData->rowLen;                // 多少行数据
        tilingData_.colLen = tilingData->colLen;                // 列数，对输入x的一半
        tilingData_.rowLenPerHeadCore = tilingData->rowLenPerHeadCore;  // 每核处理的行数
        tilingData_.rowLenPerTailCore = tilingData->rowLenPerTailCore;  // 每核处理的行数
        tilingData_.basicRowLenHeadCore = tilingData->basicRowLenHeadCore;      // 每次计算的行数
        tilingData_.basicRowLenTailCore = tilingData->basicRowLenTailCore;      // 每次计算的行数
        tilingData_.basicColLen = tilingData->basicColLen;      // 每次计算的列数
        tilingData_.headCoreNum = tilingData->headCoreNum;      // 使用的head核数
        tilingData_.realCoreNum = tilingData->realCoreNum;      // 使用的核数
        tilingData_.activateLeft = tilingData->activateLeft;
    }

    __aicore__ inline void InitBaseBuffer()
    {
        pPipe->InitBuffer(tmpConstBuffer, MAX_VALUE_NUM * sizeof(float));
    }

    __aicore__ inline void DuplicateConst()
    {
        constScale = tmpConstBuffer.Get<float>();
        Duplicate<float>(constScale, SWI_GLU_QUANT_INT8_SYM_SCALE, MAX_VALUE_NUM);
    }

    template <typename T>
    __aicore__ inline T CeilDiv(T x, T y)
    {
        return y == 0 ? 0 : (x + y - 1) / y;
    }

    __aicore__ inline float GetMax(float a, float b)
    {
        return a > b ? a : b;
    }

    template<typename T>
    __aicore__ inline T AlignUp(T num, T div)
    {
        return (div == 0) ? 0 : (num + div - 1) / div * div;
    }

protected:
    TPipe *pPipe = nullptr;
    /* tiling data */
    SwiGluQuantTilingData tilingData_;

    /* variable */
    uint32_t rowLen;
    uint32_t colLen; 
    uint32_t groupLen;
    uint32_t alignedGroupLen;
    uint32_t rowLenPerHeadCore;
    uint32_t rowLenPerTailCore;
    uint32_t basicRowLen;
    uint32_t rowLenPerCore;
    uint32_t basicRowLenHeadCore;
    uint32_t basicRowLenTailCore;
    uint32_t basicColLen;
    uint32_t headCoreNum;
    uint32_t realCoreNum;
    uint32_t outAlignLen;
    uint32_t sizeHalfLen;
    uint32_t smoothSizeFloatLen;
    uint32_t outLen;
    uint8_t rightPadding = 0;
    uint8_t smoothRightPadding = 0;
    bool isPad = false;
    bool smoothIsPad = false;
    uint16_t blockUnit;

    uint32_t coreIdx;
    uint32_t rowLoop = 1;
    uint32_t baseRow = 0;     // 记录开始处理的行数
    uint16_t basicRowLenCal;
    uint32_t mergedColLen;
    uint64_t tileLength;

    XxGluSingleTileOffsetParam offsetParam;

    TBuf<TPosition::VECCALC> tmpConstBuffer;
    TBuf<TPosition::VECCALC> tmpMaxBuffer;
    /* local memory */
    LocalTensor<float> constScale;
};
}  // namespace SwiGluQuantOpt
#endif  // SwiGluQuantBase