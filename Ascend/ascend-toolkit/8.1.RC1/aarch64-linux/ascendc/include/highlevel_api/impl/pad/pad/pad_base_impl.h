/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pad_base_impl.h
 * \brief
 */
#ifndef IMPL_PAD_PAD_PAD_BASE_IMPL_H
#define IMPL_PAD_PAD_PAD_BASE_IMPL_H
#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_transpose_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_transpose_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_transpose_impl.h"
#include "dav_c220/kernel_operator_vec_gather_mask_impl.h"
#endif

namespace AscendC {
template <typename T>
__aicore__ inline void DuplicateLastDimImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const uint32_t srcSize, const uint32_t brcbSize)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    T scalarList[BRCB_BROADCAST_NUMBER] = {0};
    const uint32_t rangeM = srcSize / BRCB_BROADCAST_NUMBER;
    const uint32_t tailM = srcSize % BRCB_BROADCAST_NUMBER;

    for (uint32_t i = 0; i < rangeM; i++) {
        for (uint32_t j = 0; j < BRCB_BROADCAST_NUMBER; j++) {
            scalarList[j] = srcTensor[i * BRCB_BROADCAST_NUMBER + j].GetValue(0);
        }
        for (uint32_t j = 0; j < BRCB_BROADCAST_NUMBER; j++) {
            Duplicate(dstTensor[(i * BRCB_BROADCAST_NUMBER + j) * brcbSize], scalarList[j], brcbSize);
        }
    }
    if (tailM != 0) {
        for (uint32_t j = 0; j < tailM; j++) {
            scalarList[j] = srcTensor[rangeM * BRCB_BROADCAST_NUMBER + j].GetValue(0);
        }
        for (uint32_t j = 0; j < tailM; j++) {
            Duplicate(dstTensor[(rangeM * BRCB_BROADCAST_NUMBER + j) * brcbSize], scalarList[j], brcbSize);
        }
    }

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

/*
32B aligned. overall solution: blocks that don't need pad use DataCopy. others use Duplicate
*/
template <typename T>
__aicore__ inline void AlignedPad(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    PadParams& padParams, PadTiling& tiling)
{
    uint16_t leftPad = padParams.leftPad;
    uint16_t rightPad = padParams.rightPad;
    int32_t padValue = padParams.padValue;

    uint32_t height = tiling.srcHeight;
    uint32_t width = tiling.srcWidth;
    uint32_t oriWidth = tiling.srcOriWidth;

    // pure 32B aligned not supported pad
    ASCENDC_ASSERT(((width != oriWidth)), {
        KERNEL_LOG(KERNEL_ERROR, "width is %u, oriWidth is %u, which can not be equal", width, oriWidth);
    });

    // not support left pad, dirty data is always in right
    ASCENDC_ASSERT(((leftPad == 0)), { KERNEL_LOG(KERNEL_ERROR, "leftPad is %u, which should be 0", leftPad); });

    // not support right pad 0
    ASCENDC_ASSERT(((rightPad != 0)), { KERNEL_LOG(KERNEL_ERROR, "rightPad is %u, which should not be 0", rightPad); });


    uint32_t elementsPerBlock = ONE_BLK_SIZE / sizeof(T);

    DataCopy(dstTensor, srcTensor, height * width);
    PipeBarrier<PIPE_V>();

    uint64_t mask[2];
    mask[0] = ((1 << rightPad) - 1) << (elementsPerBlock - (width - oriWidth));
    mask[1] = 0;

    uint32_t widthWithoutLastBlock = tiling.widthWithoutLastBlock;
    uint32_t blocksPerRow = tiling.blocksPerRow;

    uint32_t heightTiling = tiling.heightTiling;
    uint32_t heightFractal = tiling.heightFractal;
    uint32_t heightFractalTail = tiling.heightFractalTail;
    for (uint32_t i = 0; i < heightFractal; i++) {
        Duplicate<T, true>(dstTensor[i * tiling.mainLoopOffset + widthWithoutLastBlock], static_cast<T>(padValue), mask,
            heightTiling, 1, blocksPerRow);
    }
    if (heightFractalTail) {
        Duplicate<T, true>(dstTensor[tiling.tailBlockOffset], static_cast<T>(padValue), mask, heightFractalTail, 1,
            blocksPerRow);
    }
}

/*
32B unaligned.
overall solution: regard it as one column
step1: src Brcb to tmp1
step2: tmp1 DataCopy to tmp2, pad data that needs pad
step3: tmp2 TransDataTo5HD to tmp2, in-place
step4: tmp2 DataCopy to dst, only need first line(block)
*/
template <typename T>
__aicore__ inline void UnAlignedPad(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    PadParams& padParams, const LocalTensor<T>& tmpBuffer, PadTiling& tiling)
{
    uint16_t leftPad = padParams.leftPad;
    uint16_t rightPad = padParams.rightPad;
    int32_t padValue = padParams.padValue;

    uint32_t height = tiling.srcHeight;
    uint32_t width = tiling.srcWidth;
    uint32_t oriWidth = tiling.srcOriWidth;

    // must be 32B aligned with pad
    ASCENDC_ASSERT((((width + leftPad + rightPad) * sizeof(T) % ONE_BLK_SIZE == 0)), {
        KERNEL_LOG(KERNEL_ERROR,
            "width is %u, leftPad is %u, rightPad is %u, width + leftPad + rightPad should be 32B aligned", width,
            leftPad, rightPad);
    });

    uint32_t tmp1BlockNum = tiling.tmpBuffer1BlockNum;

    LocalTensor<T> tmp1 = tmpBuffer;
    LocalTensor<T> tmp2 = tmpBuffer[tiling.tmpBuffer2Offset];

    uint32_t widthTiling = tiling.widthTiling; // elements
    uint32_t widthFractal = tiling.widthFractal;
    uint32_t widthFractalTail = tiling.widthFractalTail;
    // aligned to 8 or 16
    uint32_t widthFractalTailAlingned = tiling.widthFractalTailAlingned;

    uint32_t brcbTiling = tiling.brcbTiling; // elements
    uint32_t brcbFractal = tiling.brcbFractal;
    uint32_t brcbFractalTail = tiling.brcbFractalTail;
    uint32_t brcbFractalCount = 0;

    uint32_t maxRepeatTimes = tiling.maxRepeatTimes; // 255*8 not 32B aligned, so 254
    uint32_t brcbTilingRepeatTimes = tiling.brcbTilingRepeatTimes;
    uint32_t brcbTilingRepeatTimesTail = tiling.brcbTilingRepeatTimesTail;
    uint32_t brcbFractalTailRepeatTimes = tiling.brcbFractalTailRepeatTimes;
    uint32_t brcbFractalTailRepeatTimesTail = tiling.brcbFractalTailRepeatTimesTail;

    uint32_t tmp1RowFull = tiling.tmpBuffer1RowNum;
    uint32_t tmp1RowCount = tiling.tmpBuffer1RowNum;
    uint32_t tmp1RemainRow = 0;

    uint32_t tmp2RowFull = tiling.tmpBuffer1RowNum;
    uint32_t tmp2RowCount = 0;
    uint32_t tmp2NeedRow = tiling.tmpBuffer1RowNum;

    uint32_t tmpWidth = ONE_BLK_SIZE / sizeof(T);

    // tmp2 -> dst paras
    TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = tmp1BlockNum;
    if (transDataParams.repeatTimes > 1) {
        transDataParams.dstRepStride = 16;
        transDataParams.srcRepStride = 16;
    }

    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE];

    // transdata in-place
    for (uint16_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstList[i] = (uint64_t)(tmp2[i * tmpWidth].GetPhyAddr());
        srcList[i] = (uint64_t)(tmp2[i * tmpWidth].GetPhyAddr());
    }

    // row: for each row;
    // column: for every 16*basicBlockNum column;
    // each for loop fill tmp2 once and to dst
    for (uint32_t j = 0; j < height; j++) {
        for (uint32_t i = 0; i < (widthFractal + 1); i++) {
            tmp2RowCount = 0;
            // first fractal, may have leftPad
            if (i == 0 && leftPad != 0) {
                Duplicate<T, true>(tmp2, static_cast<T>(padValue), tmpWidth, leftPad, 1, 1);
                tmp2RowCount += leftPad;
            }

            // last fractal, may have rightPad
            if (i == widthFractal) {
                if (rightPad != 0) {
                    Duplicate<T, true>(tmp2[(widthFractalTailAlingned - rightPad) * tmpWidth], static_cast<T>(padValue),
                        tmpWidth, rightPad, 1, 1);
                }
                tmp2NeedRow = widthFractalTailAlingned - tmp2RowCount - rightPad;
            } else {
                tmp2NeedRow = tmp2RowFull - tmp2RowCount;
            }

            while (tmp2NeedRow != 0) {
                PipeBarrier<PIPE_V>();
                tmp1RemainRow = tmp1RowFull - tmp1RowCount;
                if (tmp2NeedRow > tmp1RemainRow) {
                    if (tmp1RemainRow != 0) {
                        DataCopy(tmp2[tmp2RowCount * tmpWidth], tmp1[tmp1RowCount * tmpWidth],
                            { 1, static_cast<uint16_t>(tmp1RemainRow), 0, 0 });
                        tmp1RowCount += tmp1RemainRow;
                        tmp2RowCount += tmp1RemainRow;
                        tmp2NeedRow -= tmp1RemainRow;
                        PipeBarrier<PIPE_V>();
                    }
                    // load data to tmp1 from src
#if __CCE_AICORE__ == 220
                    if (brcbFractalCount == brcbFractal) {
                        for (uint32_t i = 0; i < brcbFractalTailRepeatTimes; i++) {
                            Brcb(tmp1[i * maxRepeatTimes * 8 * tmpWidth],
                                srcTensor[brcbFractalCount * brcbTiling + i * maxRepeatTimes * 8], maxRepeatTimes,
                                { 1, 8 });
                        }
                        if (brcbFractalTailRepeatTimesTail) {
                            Brcb(tmp1[brcbFractalTailRepeatTimes * maxRepeatTimes * 8 * tmpWidth],
                                srcTensor[brcbFractalCount * brcbTiling +
                                brcbFractalTailRepeatTimes * maxRepeatTimes * 8],
                                brcbFractalTailRepeatTimesTail, { 1, 8 });
                        }
                        tmp1RowFull = brcbFractalTail;
                    } else {
                        for (uint32_t i = 0; i < brcbTilingRepeatTimes; i++) {
                            Brcb(tmp1[i * maxRepeatTimes * 8 * tmpWidth],
                                srcTensor[brcbFractalCount * brcbTiling + i * maxRepeatTimes * 8], maxRepeatTimes,
                                { 1, 8 });
                        }
                        if (brcbTilingRepeatTimesTail) {
                            Brcb(tmp1[brcbTilingRepeatTimes * maxRepeatTimes * 8 * tmpWidth],
                                srcTensor[brcbFractalCount * brcbTiling + brcbTilingRepeatTimes * maxRepeatTimes * 8],
                                brcbTilingRepeatTimesTail, { 1, 8 });
                        }
                    }
#else
                    if (brcbFractalCount == brcbFractal) {
                        DuplicateLastDimImpl(tmp1, srcTensor[brcbFractal * brcbTiling], brcbFractalTail,
                            ONE_BLK_SIZE / sizeof(T));
                        tmp1RowFull = brcbFractalTail;
                    } else {
                        DuplicateLastDimImpl(tmp1, srcTensor[brcbFractalCount * brcbTiling], brcbTiling,
                            ONE_BLK_SIZE / sizeof(T));
                    }
#endif
                    brcbFractalCount += 1;
                    tmp1RowCount = 0;
                } else {
                    DataCopy(tmp2[tmp2RowCount * tmpWidth], tmp1[tmp1RowCount * tmpWidth],
                        { 1, static_cast<uint16_t>(tmp2NeedRow), 0, 0 });
                    tmp1RowCount += tmp2NeedRow;
                    tmp2RowCount += tmp2NeedRow;
                    tmp2NeedRow = 0;
                }
            }

            PipeBarrier<PIPE_V>();

            // tmp2 -> tmp2
            TransDataTo5HD<T>(dstList, srcList, transDataParams);
            PipeBarrier<PIPE_V>();
            if (i == widthFractal) {
                // copy first line to dst
                if (sizeof(T) == sizeof(half)) {
                    DataCopy(dstTensor[j * (width + leftPad + rightPad) + widthFractal * 16 * tmp1BlockNum], tmp2,
                        { static_cast<uint16_t>(widthFractalTailAlingned / 16), 1, 15, 0 });
                } else if (sizeof(T) == sizeof(float)) {
                    if (widthFractalTailAlingned / 16 != 0) {
                        DataCopy(dstTensor[j * (width + leftPad + rightPad) + widthFractal * 16 * tmp1BlockNum], tmp2,
                            { static_cast<uint16_t>(widthFractalTailAlingned / 16), 2, 14, 0 });
                    }
                    if (widthFractalTailAlingned % 16) {
                        DataCopy(dstTensor[j * (width + leftPad + rightPad) + widthFractal * 16 * tmp1BlockNum +
                            widthFractalTailAlingned / 16 * 16],
                            tmp2[widthFractalTailAlingned / 16 * 16 * 8], { 1, 1, 15, 0 });
                    }
                }
            } else {
                // copy first line to dst
                if (sizeof(T) == sizeof(half)) {
                    DataCopy(dstTensor[j * (width + leftPad + rightPad) + i * 16 * tmp1BlockNum], tmp2,
                        { static_cast<uint16_t>(tmp1BlockNum), 1, 15, 0 });
                } else if (sizeof(T) == sizeof(float)) {
                    DataCopy(dstTensor[j * (width + leftPad + rightPad) + i * 16 * tmp1BlockNum], tmp2,
                        { static_cast<uint16_t>(tmp1BlockNum), 2, 14, 0 });
                }
            }
        }
    }
}
} // namespace AscendC
#endif // IMPL_PAD_PAD_PAD_BASE_IMPL_H
