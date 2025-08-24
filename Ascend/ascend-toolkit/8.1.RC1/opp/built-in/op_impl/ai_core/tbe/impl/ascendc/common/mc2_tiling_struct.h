/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
 * \file mc2_tiling_struct.h
 * \brief
 */
#ifndef MC2_TILING_STRUCT_H
#define MC2_TILING_STRUCT_H

struct RCSTiling {
    uint32_t rankDim;
    uint32_t rankID;
    uint32_t commtype;
    uint32_t subtype;
    uint32_t tileCnt;
    uint32_t tailM;
    uint32_t tailCnt;
    uint32_t biasLen;
    uint32_t isAdd;
    uint32_t rankM;
    uint32_t rankN;
    uint32_t rankK;
    uint32_t gatherIndex;
    uint32_t isTransposeA;
    uint32_t isTransposeB;
    uint32_t storageGather;
    uint64_t nd2NzWorkLen;
    uint64_t cToFloatLen;
    uint64_t gatherLen;
    uint32_t workspaceAddr4;
    uint32_t aicCoreNum;
    uint32_t needUbBuffer;
    uint32_t addX3UbCnt;
    uint32_t commWorkSpaceSize;
    uint32_t isInputCommQuantScale;
    uint32_t dataType;
};

struct TileL2Tiling {
    uint32_t mL2TileCnt;
    uint32_t nL2TileCnt;
    uint32_t mTileBlocks;
    uint32_t nTileBlocks;
    uint32_t mTailBlocks;
    uint32_t nTailBlocks;
    uint32_t rankTileNum;
    uint32_t calcOrder;
    uint32_t enableL2Tile;
};

#endif // MC2_TILING_STRUCT_H
