/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_cube_out_datacopy_wrapper.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_OUT_COPY_CUBE_OUT_DATACOPY_WRAPPER_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_OUT_COPY_CUBE_OUT_DATACOPY_WRAPPER_H

#include "../../utils/matmul_module.h"
#include "../../utils/matmul_param.h"
#include "copy_cube_out_utils.h"

namespace AscendC {
namespace Impl {
namespace Detail {

constexpr int32_t DOUBLE_SPACE = 2;
constexpr int32_t TWO_TIMES = 2;
constexpr int32_t EIGHT_TIMES = 8;
constexpr int32_t SHIFT_16_BIT = 16;
constexpr int32_t SHIFT_32_BIT = 32;
constexpr int32_t SHIFT_48_BIT = 48;
constexpr uint32_t MAX_REPEAT_STRIDE = 255;
constexpr int32_t PATTERN_SIZE = 8;
constexpr int32_t PATTERN_OFFSET = 2;
/*
    CopyCubeOutWrapper is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    CopyCubeOutWrapper is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, const auto& MM_CFG>
class CopyCubeOutWrapper
{
    using SrcT = typename A_TYPE::T;
    using DstT = typename C_TYPE::T;

    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(LocalWorkspace);

public:
    __aicore__ inline CopyCubeOutWrapper() = default;
    __aicore__ inline ~CopyCubeOutWrapper() = default;

    // get blockCount by DstT
    __aicore__ inline constexpr int32_t GetBlockCount()
    {
        return sizeof(DstT) == B32_BYTE_SIZE ? BLOCK_CUBE : ONE_BLK_SIZE / sizeof(DstT);
    }

    // if target is not aligned, must copy the unalign data to trans UB
    __aicore__ inline bool IsNeedPadUnalignedToTrans(const int32_t baseWidth, const uint32_t dimN,
        const bool isComputeLineByLine, const bool isTargetAligned)
    {
        if constexpr (IsSameType<SrcT, int8_t>::value) {
            bool isOdd = false;
            if constexpr (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value) {
                if (baseWidth % TWO_TIMES > 0) {
                    isOdd = true;
                }
            }
            bool isSingleCore = (MATMUL_MODULE(MatmulShapeInfo)->GetOrgM() <= MATMUL_MODULE(MatmulShapeInfo)->
                GetSingleCoreM()) && (dimN <= MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN());
            bool isMutiCoreNeedPad = !isSingleCore && !isComputeLineByLine;
            return (!isTargetAligned && (isSingleCore || isMutiCoreNeedPad) && !isOdd);
        } else {
            return (!isTargetAligned);
        }
    }

    // copy the unalign data to trans UB
    template <bool enSequentialWrite>
    __aicore__ inline void PadUnalignedToTrans(const LocalTensor<DstT>& trans, const GlobalTensor<DstT>& gm,
        int32_t dstOffset, bool isComputeLineByLine, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        int32_t alignedSize;
        if constexpr (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value) {
            alignedSize = GetC0Size<SrcT>();
        } else {
            alignedSize = BLOCK_CUBE;
        }
        int32_t baseUseN = CeilAlign(baseWidth, alignedSize);
        int32_t gmTailOffset = dstOffset + baseUseN - GetBlockCount();
        int32_t transTailOffset = baseUseN - GetBlockCount();

        auto enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(enQueEvtID);
        WaitFlag<HardEvent::MTE3_MTE2>(enQueEvtID);

        if (isComputeLineByLine) {
            if constexpr (enSequentialWrite) {
                PadUnalignedToTransByLine(trans[transTailOffset], gm[gmTailOffset], baseUseN, baseUseN, baseHeight);
            } else {
                PadUnalignedToTransByLine(trans[transTailOffset], gm[gmTailOffset], baseUseN,
                    MATMUL_MODULE(MatmulShapeInfo)->GetOrgN(), baseHeight);
            }
        } else {
            PadUnalignedToTransWithStride(trans[transTailOffset], gm[gmTailOffset], baseHeight, baseWidth, baseBlockWidth);
        }

        // if copy gm to ub, must add the set/wait flag to wait the UB has be writed;
        event_t eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
    }

    // trans nz buffer to nd buffer
    __aicore__ inline auto TransNZ2NDByVec(const LocalTensor<DstT>& trans, const LocalTensor<DstT>& localBuf,
        int32_t blockHigh, int32_t blockWidth, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockWidth)
    {
        CopyOutEnQue();
        ASCENDC_ASSERT(((blockWidth * GetBlockCount() * sizeof(DstT) / ONE_BLK_SIZE) <= MAX_REPEAT_TIMES), {
            KERNEL_LOG(KERNEL_ERROR, "blockWidth is %d, blockCount is %d, repeat time exceed max time %d", blockWidth,
                GetBlockCount(), MAX_REPEAT_TIMES);
        });
        if constexpr (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value) {
            TransNZ2NDByVecDstB8(trans, localBuf, blockHigh, blockWidth, baseHeight, baseWidth, baseBlockWidth);
        } else {
            TransNZ2NDByVecDstNotB8(trans, localBuf, blockHigh, blockWidth, baseHeight, baseWidth, baseBlockWidth);
        }
        CopyOutDeQue();
    }

    // copy trans buffer to gm
    template <bool enSequentialWrite>
    __aicore__ inline void CopyTrans2GM(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& trans,
        int32_t curRow, int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight, int32_t baseBlockWidth,
        int32_t dstOffset, int32_t offset, int32_t dstStride, bool isComputeLineByLine, bool isTargetAligned)
    {
        int32_t blockLen = baseBlockWidth * (GetBlockCount() * sizeof(DstT) / ONE_BLK_SIZE);
        if constexpr (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value) {
            blockLen = Ceil(blockLen, TWO_TIMES);
        }
        if (!isComputeLineByLine) {
            DataCopy(gm[dstOffset], trans, { static_cast<uint16_t>(baseHeight), static_cast<uint16_t>(blockLen), 0,
                static_cast<uint16_t>(dstStride) });
            return;
        }
        if constexpr (IsSameType<SrcT, int8_t>::value) {
            if constexpr (!enSequentialWrite) {
                dstOffset = curRow * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() *
                    MATMUL_MODULE(MatmulShapeInfo)->GetOrgN() + curCol *
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
                offset = MATMUL_MODULE(MatmulShapeInfo)->GetOrgN();
            }
            int32_t newBlockCount;
            if constexpr (IsSameType<DstT, float>::value || IsSameType<DstT, int32_t>::value) {
                newBlockCount = BLOCK_CUBE;
            } else {
                newBlockCount = ONE_BLK_SIZE / sizeof(DstT);
            }
            if (isTargetAligned) {
                CopyTrans2GMByVecByLineAlign(gm[dstOffset], trans, baseHeight, blockLen, newBlockCount, offset);
            } else if (blockLen == 1) {
                CopyTrans2GMByVecByLineUnalignOneBlock(gm[dstOffset], trans, baseHeight, baseWidth, blockLen, newBlockCount, offset);
            } else {
                if constexpr (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value) {
                    CopyTrans2GMByVecByLineUnalign<true>(gm[dstOffset], trans, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth, blockLen, newBlockCount, offset);
                } else {
                    CopyTrans2GMByVecByLineUnalign<false>(gm[dstOffset], trans, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth, blockLen, newBlockCount, offset);
                }
            }
        } else {
            CopyTrans2GMByVecByLineAlign(gm[dstOffset], trans, baseHeight, blockLen, ONE_BLK_SIZE / sizeof(DstT), offset);
        }
    }

    // if baseWidth is unaligned, then copy the tail data
    __aicore__ inline void CopyLocal2GMNZ2NDOnTheFlyTail(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& localBuf,
        int32_t baseHeight, int32_t baseWidth, int32_t iterIdx, int32_t calcWidth, const event_t& eventIDMte3ToMte2)
    {
        LocalTensor<DstT> trans = MATMUL_MODULE(LocalWorkspace)->GetNZ2NDWorkspace().template ReinterpretCast<DstT>();
        trans.SetSize(GetBlockCount());
        int32_t srcTailOffset = iterIdx * GetBlockCount() + calcWidth * GetBlockCount() * CeilAlign(baseHeight, GetBlockCount());
        if (baseWidth * sizeof(DstT) > ONE_BLK_SIZE) {
            CopyLocal2GMNZ2NDRegMov(gm, localBuf, trans, baseHeight, baseWidth, iterIdx, calcWidth, srcTailOffset);
        } else {
            if (iterIdx > 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            }
            if constexpr (IsSameType<typename A_TYPE::T, half>::value &&
                IsSameType<typename B_TYPE::T, int8_t>::value) {
                event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(eventID);
                WaitFlag<HardEvent::V_MTE2>(eventID);
            }
            DataCopy(trans, gm[baseWidth], { 1, 1, 0, 0 });
            event_t eventIDMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(eventIDMte2ToMte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIDMte2ToMte3);
            DataCopy(gm, localBuf[srcTailOffset], { 1, 1, 0, 0 });
            PipeBarrier<PIPE_MTE3>();
            DataCopy(gm[baseWidth], trans, { 1, 1, 0, 0 });
            if (iterIdx <  baseHeight - 1) {
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            }
        }
    }

private:
    __aicore__ inline void PadUnalignedToTransByLine(const LocalTensor<DstT>& trans, const GlobalTensor<DstT>& gm,
        int32_t transStride, int32_t gmStride, int32_t baseHeight)
    {
        // copy gm to trans one line by one line
        int32_t dstOffset = 0;
        int32_t srcOffset = 0;
        int32_t blockLen = GetBlockCount() * sizeof(DstT) / ONE_BLK_SIZE;
        for (int32_t i = 0; i < baseHeight; ++i) {
            DataCopy(trans[dstOffset], gm[srcOffset], { static_cast<uint16_t>(1),
                static_cast<uint16_t>(blockLen), 0, 0 });
            dstOffset += transStride;
            srcOffset += gmStride;
        }
    }

    __aicore__ inline void PadUnalignedToTransWithStride(const LocalTensor<DstT>& trans, const GlobalTensor<DstT>& gm, int32_t baseHeight,
        int32_t baseWidth, int32_t baseBlockWidth)
    {
        // copy gm to trans with stride
        DataCopy(trans, gm, { static_cast<uint16_t>(baseHeight), static_cast<uint16_t>(1),
            static_cast<uint16_t>(MATMUL_MODULE(MatmulShapeInfo)->GetOrgN() / GetBlockCount() - 1),
            static_cast<uint16_t>(baseWidth / GetBlockCount()) });
    }

    __aicore__ inline void TransNZ2NDByVecDstB8(const LocalTensor<DstT>& trans, const LocalTensor<DstT>& localBuf,
        int32_t blockHigh, int32_t blockWidth, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockWidth)
    {
        struct UnaryRepeatParams intriParams;
        intriParams.dstBlkStride = Ceil(baseWidth, ONE_BLK_SIZE);
        intriParams.srcBlkStride = 1;
        uint32_t dstRepStride = Ceil(baseWidth * sizeof(DstT), ONE_BLK_SIZE) * EIGHT_TIMES;
        intriParams.dstRepStride = dstRepStride;
        intriParams.srcRepStride = (GetBlockCount() * sizeof(DstT) / ONE_BLK_SIZE) * EIGHT_TIMES;
        int32_t dstOffset = 0;
        int32_t srcOffset = 0;
        int32_t highBlocks = (blockHigh * BLOCK_CUBE) / EIGHT_TIMES / MAX_REPEAT_TIMES;
        int32_t highTail = (blockHigh * BLOCK_CUBE) / EIGHT_TIMES % MAX_REPEAT_TIMES;
        uint64_t mask[2] = {uint64_t(-1), uint64_t(-1)};
        // mov src to dst width aligned
        LocalTensor<int16_t> src = localBuf.template ReinterpretCast<int16_t>();
        LocalTensor<int16_t> dst = trans.template ReinterpretCast<int16_t>();
        SetVectorMask<int16_t>(mask[1], mask[0]);
        constexpr int64_t srcOffsetStride = BLOCK_CUBE * EIGHT_TIMES;
        const int64_t dstOffsetStride = baseBlockWidth * BLOCK_CUBE * EIGHT_TIMES / TWO_TIMES;
        for (int32_t i = 0; i < Ceil(blockWidth, TWO_TIMES); ++i) {
            if constexpr (C_TYPE::format != CubeFormat::ND_ALIGN) {
                // if the baseWidth is not aligned, set the mask value;
                if (i == (Ceil(blockWidth, TWO_TIMES) - 1) && (baseWidth % GetBlockCount() != 0)) {
                    uint64_t masktail = (1 << (Ceil(baseWidth % GetBlockCount(), TWO_TIMES))) - 1;
                    mask[0] =
                        masktail + (masktail << SHIFT_16_BIT) + (masktail << SHIFT_32_BIT) + (masktail << SHIFT_48_BIT);
                    mask[1] = mask[0];
                    SetVectorMask<int16_t>(mask[1], mask[0]);
                }
            }
            int32_t dstMulsOffset = dstOffset;
            for (int32_t j = 0; j < highBlocks; ++j) {
                Muls<int16_t, false>(
                    dst[dstMulsOffset], src[srcOffset], (int16_t)1, mask, MAX_REPEAT_TIMES, intriParams);
                srcOffset += MAX_REPEAT_TIMES * BLOCK_CUBE;
                dstMulsOffset += blockWidth * GetBlockCount() * MAX_REPEAT_TIMES;
            }
            if (highTail > 0) {
                if (dstRepStride > MAX_REPEAT_STRIDE) {
                    int32_t tmpSrcOffset = srcOffset;
                    for (int32_t j = 0; j < highTail; j++) {
                        Muls<int16_t, false>(dst[dstMulsOffset], src[tmpSrcOffset], (int16_t)1, mask, 1, intriParams);
                        dstMulsOffset += dstOffsetStride;
                        tmpSrcOffset += srcOffsetStride;
                    }
                } else {
                    Muls<int16_t, false>(dst[dstMulsOffset], src[srcOffset], (int16_t)1, mask, highTail, intriParams);
                }
                srcOffset += highTail * BLOCK_CUBE * EIGHT_TIMES;
            }
            dstOffset += BLOCK_CUBE;
        }
    }

    __aicore__ inline void TransNZ2NDByVecDstNotB8(const LocalTensor<DstT>& trans, const LocalTensor<DstT>& localBuf,
        int32_t blockHigh, int32_t blockWidth, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockWidth)
    {
        struct UnaryRepeatParams intriParams;
        intriParams.srcBlkStride = 1;
        int32_t dstOffset = 0;
        int32_t srcOffset = 0;
        int32_t highBlocks = 0;
        int32_t highTail = 0;
        int32_t srcStride = MAX_REPEAT_TIMES * GetBlockCount();
        int32_t dstStride = blockWidth * GetBlockCount() * MAX_REPEAT_TIMES;
        bool isBeyondMaxStride = false;
        uint64_t mask[2] = {uint64_t(-1), uint64_t(-1)};

        if constexpr (sizeof(DstT) == B32_BYTE_SIZE) {
            intriParams.dstBlkStride = 1;
            intriParams.dstRepStride = blockWidth * GetBlockCount() * sizeof(DstT) / ONE_BLK_SIZE;
            intriParams.srcRepStride = GetBlockCount() * sizeof(DstT) / ONE_BLK_SIZE;
            highBlocks = (blockHigh * GetBlockCount()) / MAX_REPEAT_TIMES;
            highTail = (blockHigh * GetBlockCount()) % MAX_REPEAT_TIMES;
            mask[0] = static_cast<uint64_t>((1<< GetBlockCount()) - 1);
            mask[1] = 0;
        } else {
            intriParams.dstBlkStride = blockWidth;
            uint32_t dstRepStride = (blockWidth * GetBlockCount() * sizeof(DstT) / ONE_BLK_SIZE) * EIGHT_TIMES;
            intriParams.dstRepStride = dstRepStride;
            if (dstRepStride > MAX_REPEAT_STRIDE) {
                isBeyondMaxStride = true;
            }
            intriParams.srcRepStride = (GetBlockCount() * sizeof(DstT) / ONE_BLK_SIZE) * EIGHT_TIMES;
            highBlocks = (blockHigh * GetBlockCount()) / EIGHT_TIMES / MAX_REPEAT_TIMES;
            highTail = (blockHigh * GetBlockCount()) / EIGHT_TIMES % MAX_REPEAT_TIMES;
            srcStride *= EIGHT_TIMES;
            dstStride *= EIGHT_TIMES;
        }
        SetVectorMask<DstT>(mask[1], mask[0]);
        const int64_t srcOffsetStride = GetBlockCount() * EIGHT_TIMES;
        const int64_t dstOffsetStride = baseBlockWidth * BLOCK_CUBE * EIGHT_TIMES;
        for (int32_t i = 0; i < blockWidth; ++i) {
            if constexpr (C_TYPE::format != CubeFormat::ND_ALIGN) {
                // if the baseWidth is not aligned, set the mask value;
                if (i == (blockWidth - 1) && (baseWidth % GetBlockCount() != 0)) {
                    uint64_t masktail = (1 << (baseWidth % GetBlockCount())) - 1;
                    mask[0] = masktail + (masktail << SHIFT_16_BIT) + (masktail << SHIFT_32_BIT) + (masktail << SHIFT_48_BIT);
                    mask[1] = mask[0];
                    SetVectorMask<DstT>(mask[1], mask[0]);
                }
            }
            int32_t dstMulsOffset = dstOffset;
            for (int32_t j = 0; j < highBlocks; ++j) {
                Muls<DstT, false>(trans[dstMulsOffset], localBuf[srcOffset], (DstT)1.0, mask, MAX_REPEAT_TIMES, intriParams);
                srcOffset += srcStride;
                dstMulsOffset += dstStride;
            }
            if (highTail) {
                if (isBeyondMaxStride) {
                    for (int32_t j = 0; j < highTail; j++) {
                        Muls<DstT, false>(trans[dstMulsOffset + j * dstOffsetStride],
                            localBuf[srcOffset + j * srcOffsetStride], (DstT)1.0, mask, 1, intriParams);
                    }
                } else {
                    Muls<DstT, false>(trans[dstMulsOffset], localBuf[srcOffset], (DstT)1.0, mask, highTail, intriParams);
                }
                if constexpr (sizeof(DstT) == B32_BYTE_SIZE) {
                    srcOffset += highTail * GetBlockCount();
                } else {
                    srcOffset += highTail * srcOffsetStride;
                }
            }
            dstOffset += GetBlockCount();
        }
    }

    __aicore__ inline void CopyTrans2GMByVecByLineAlign(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& trans, int32_t baseHeight,
        int32_t blockLen, int32_t blockCount, int32_t offset)
    {
        int32_t dstOffset = 0;
        int32_t srcOffset = 0;
        int32_t blockOffset = blockLen * blockCount;
        for (int32_t i = 0; i < baseHeight; ++i) {
            DataCopy(gm[dstOffset], trans[srcOffset],
                { 1, static_cast<uint16_t>(blockLen), 0, 0 });
            PipeBarrier<PIPE_MTE3>();
            dstOffset += offset;
            srcOffset += blockOffset;
        }
    }

    __aicore__ inline void CopyTrans2GMByVecByLineUnalignOneBlock(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& trans,
        int32_t baseHeight, int32_t baseWidth, int32_t blockLen, int32_t blockCount, int32_t offset)
    {
        CopyTrans2GMEnQue();
        int32_t padLen = (ONE_BLK_SIZE - baseWidth * sizeof(DstT)) / sizeof(DstT);
        SetAtomicAdd<int16_t>();
        int32_t dstOffset = 0;
        for (int32_t i = 0; i < baseHeight; ++i) {
            LocalTensor<DstT> transAligin = MATMUL_MODULE(LocalWorkspace)->template
                GetWorkspaceWithOffset<ToMatmulConfig(MM_CFG).enableUBReuse>(0)
                .template ReinterpretCast<DstT>();
            int32_t transIndex = i * blockLen * blockCount;
            for (int32_t j = 0; j < baseWidth; ++j) {
                transAligin.SetValue(j, trans.GetValue(transIndex + j));
            }
            for (int32_t j = baseWidth; j < blockCount; ++j) {
                transAligin.SetValue(j, 0);
            }
            DataCopy(gm[dstOffset], transAligin, { 1, 1, 0, 0 });
            dstOffset += offset;
            CopyLocal2GMNZ2NDDeQue();
        }
        SetAtomicNone();
    }

    template <bool isDstB8>
    __aicore__ inline auto CopyTrans2GMByVecByLineUnalign(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& trans,
        int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight, int32_t baseBlockWidth, int32_t blockLen,
        int32_t blockCount, int32_t offset) -> enable_if_t<isDstB8, void>
    {
        LocalTensor<uint16_t> transAligin = MATMUL_MODULE(LocalWorkspace)->template
            GetWorkspaceWithOffset<ToMatmulConfig(MM_CFG).enableUBReuse>(0).template ReinterpretCast<uint16_t>();
        int32_t remainLen = (baseWidth % blockCount) / TWO_TIMES;
        CopyTrans2GMEnQue();
        LocalTensor<uint16_t> src1Pattern;
        src1Pattern = MATMUL_MODULE(LocalWorkspace)->template GetWorkspaceWithOffset<
            ToMatmulConfig(MM_CFG).enableUBReuse>(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength()
            / TWO_TIMES).template ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> tmpSrc = trans.template ReinterpretCast<uint16_t>();
        src1Pattern.SetSize(PATTERN_SIZE);
        src1Pattern.SetValue(0, 0xFFFF << remainLen);
        src1Pattern.SetValue(1, (1 << remainLen) - 1);
        for (int32_t i = PATTERN_OFFSET; i < PATTERN_SIZE; ++i) {
            src1Pattern.SetValue(i, 0);
        }
        int32_t orinRemain = baseWidth % blockCount;
        int32_t gmOffset = blockCount * (blockLen - PATTERN_OFFSET);
        int32_t dstOffset = 0;
        int32_t srcOffset = 0;
        int32_t blockOffset = blockLen * blockCount;
        for (int32_t i = 0; i < baseHeight; ++i) {
            DataCopy(gm[dstOffset], trans[srcOffset], { 1, static_cast<uint16_t>(blockLen - 1), 0, 0 });
            if (baseWidth % TWO_TIMES == 0) {
                CopyOutEnQue();
                GatherMaskParams gatherMaskParams(1, 1, PATTERN_SIZE, PATTERN_SIZE);
                uint64_t rsvdCnt = 0;
                GatherMask<uint16_t>(transAligin, tmpSrc[((i + 1) * blockLen - PATTERN_OFFSET) * BLOCK_CUBE],
                    src1Pattern, false, 0, gatherMaskParams, rsvdCnt);
                LocalTensor<DstT> tmpTrans = transAligin.template ReinterpretCast<DstT>();
                DataCopy(gm[dstOffset + gmOffset + remainLen * DOUBLE_SPACE], tmpTrans, { 1, 1, 0, 0 });
            } else {
                CopyLocal2GMNZ2NDDeQue();
                LocalTensor<DstT> tmpTrans = transAligin.template ReinterpretCast<DstT>();
                for (int32_t j = 0; j < ONE_BLK_SIZE; ++j) {
                    tmpTrans.SetValue(j, trans[srcOffset + gmOffset + orinRemain].GetValue(j));
                }
                CopyLocal2GMNZ2NDEnQue();
                DataCopy(gm[dstOffset + gmOffset + orinRemain], tmpTrans, { 1, 1, 0, 0 });
            }
            PipeBarrier<PIPE_MTE3>();
            dstOffset += offset;
            srcOffset += blockOffset;
        }
    }

    template <bool isDstB8>
    __aicore__ inline auto CopyTrans2GMByVecByLineUnalign(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& trans,
        int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight, int32_t baseBlockWidth, int32_t blockLen,
        int32_t blockCount, int32_t offset) -> enable_if_t<!isDstB8, void>
    {
        LocalTensor<DstT> transAligin = MATMUL_MODULE(LocalWorkspace)->template
            GetWorkspaceWithOffset<ToMatmulConfig(MM_CFG).enableUBReuse>(0).template ReinterpretCast<DstT>();
        int32_t remainLen = baseWidth % blockCount;
        CopyTrans2GMEnQue();
        LocalTensor<uint16_t> src1Pattern;
        src1Pattern = MATMUL_MODULE(LocalWorkspace)->template
            GetWorkspaceWithOffset<ToMatmulConfig(MM_CFG).enableUBReuse>(
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength() / TWO_TIMES).template ReinterpretCast<uint16_t>();
        src1Pattern.SetSize(PATTERN_SIZE);
        src1Pattern.SetValue(0, 0xFFFF << remainLen);
        src1Pattern.SetValue(1, (1 << remainLen) - 1);
        for (int32_t i = PATTERN_OFFSET; i < PATTERN_SIZE; ++i) {
            src1Pattern.SetValue(i, 0);
        }
        int32_t gmOffset = blockCount * (blockLen - PATTERN_OFFSET);
        int32_t dstOffset = 0;
        int32_t srcOffset = 0;
        int32_t blockOffset = blockLen * blockCount;
        for (int32_t i = 0; i < baseHeight; ++i) {
            DataCopy(gm[dstOffset], trans[srcOffset], { 1, static_cast<uint16_t>(blockLen - 1), 0, 0 });
            GatherMaskParams gatherMaskParams(1, 1, PATTERN_SIZE, PATTERN_SIZE);
            uint64_t rsvdCnt = 0;
            CopyOutEnQue();
            GatherMask<DstT>(transAligin, trans[srcOffset + gmOffset],
                src1Pattern, false, 0, gatherMaskParams, rsvdCnt);
            DataCopy(gm[dstOffset + gmOffset + remainLen], transAligin, { 1, 1, 0, 0 });
            PipeBarrier<PIPE_MTE3>();
            dstOffset += offset;
            srcOffset += blockOffset;
        }
    }

    __aicore__ inline void CopyLocal2GMNZ2NDRegMov(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& localBuf,
        LocalTensor<DstT>& trans, int32_t baseHeight, int32_t baseWidth, int32_t iterIdx, int32_t calcWidth,
        int32_t srcTailOffset)
    {
        int32_t dstTailOffset = calcWidth * GetBlockCount();
        int32_t basicOffset = 0;
        if constexpr (sizeof(DstT) == B32_BYTE_SIZE) {
            DataCopy(gm[dstTailOffset], localBuf[srcTailOffset], { 1, 1, 0, 0 });
            basicOffset = ONE_BLK_SIZE / sizeof(DstT);
        }

        // reg_mov
        srcTailOffset = srcTailOffset + basicOffset -
            GetBlockCount() * CeilAlign(baseHeight, GetBlockCount()) + baseWidth % GetBlockCount();
        dstTailOffset = dstTailOffset + basicOffset + baseWidth % GetBlockCount() - GetBlockCount();
        if constexpr (IsSameType<typename A_TYPE::T, half>::value &&
            IsSameType<typename B_TYPE::T, int8_t>::value) {
            event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventID);
            WaitFlag<HardEvent::V_S>(eventID);
        }
        int32_t j = 0;
        for (int32_t k = 0; k < GetBlockCount() - baseWidth % GetBlockCount(); j++, k++) {
            DstT scalar = localBuf.GetValue(srcTailOffset + k);
            trans.SetValue(j, scalar);
        }
        srcTailOffset = iterIdx * GetBlockCount() + calcWidth * GetBlockCount() * CeilAlign(baseHeight, GetBlockCount());
        for (int32_t k = 0; k < baseWidth % GetBlockCount(); j++, k++) {
            DstT scalar = localBuf.GetValue(srcTailOffset + k);
            trans.SetValue(j, scalar);
        }

        CopyLocal2GMNZ2NDEnQue();
        // copy the tail from ub to gm
        DataCopy(gm[dstTailOffset], trans, { 1, 1, 0, 0 });
        if constexpr (IsSameType<typename A_TYPE::T, half>::value &&
            IsSameType<typename B_TYPE::T, int8_t>::value) {
            CopyLocal2GMNZ2NDDeQue();
        }
    }
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_OUT_COPY_CUBE_OUT_DATACOPY_WRAPPER_H