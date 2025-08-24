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
 * \file pad_v200_impl.h
 * \brief
 */
#ifndef IMPL_PAD_PAD_PAD_V200_IMPL_H
#define IMPL_PAD_PAD_PAD_V200_IMPL_H

#include "pad_base_impl.h"

namespace AscendC {
template <typename T>
__aicore__ inline void PadCompute(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    PadParams &padParams, const LocalTensor<uint8_t> &sharedTmpBuffer, PadTiling &tiling)
{
    uint32_t width = tiling.srcWidth;
    // 32B aligned
    if (width * sizeof(T) % ONE_BLK_SIZE == 0) {
        AlignedPad(dstTensor, srcTensor, padParams, tiling);
    } else {
        LocalTensor<T> tmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();
        UnAlignedPad(dstTensor, srcTensor, padParams, tmpBuffer, tiling);
    }
}

template <typename T>
__aicore__ inline void UnPadCompute(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    UnPadParams &unPadParams, LocalTensor<uint8_t> &sharedTmpBuffer, UnPadTiling &tiling)
{
    uint16_t rightPad = unPadParams.rightPad;
    uint16_t height = tiling.srcHeight;
    uint16_t width = tiling.srcWidth;

    LocalTensor<T> tmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();

    uint32_t tmp1BlockNum = tiling.tmpBuffer1BlockNum;

    LocalTensor<T> tmp1 = tmpBuffer;
    LocalTensor<T> tmp2 = tmpBuffer[tiling.tmpBuffer2Offset];

    uint16_t leftPad = 0;

    uint32_t widthTiling = tiling.widthTiling; // elements
    uint32_t widthFractal = tiling.widthFractal;
    uint32_t widthFractalTail = tiling.widthFractalTail;

    uint32_t tmp1RowFull = tiling.tmpBuffer1RowNum;
    uint32_t tmp1RowCount = tiling.tmpBuffer1RowNum;
    uint32_t tmp1RemainRow = 0;

    uint32_t tmp2RowFull = tiling.tmpBuffer1RowNum;
    uint32_t tmp2RowCount = 0;
    uint32_t tmp2NeedRow = tiling.tmpBuffer1RowNum;

    uint32_t tmpWidth = ONE_BLK_SIZE / sizeof(T);

    uint32_t tmp2ToDstTiling = tiling.tmpBuffer1RowNum; // elements
    uint32_t tmp2ToDstFractal = height * (width - leftPad - rightPad) / tmp2ToDstTiling;
    uint32_t tmp2ToDstFractalTail = height * (width - leftPad - rightPad) % tmp2ToDstTiling;
    uint32_t tmp2ToDstCount = 0;

    if (tmp2ToDstFractal == 0) {
        tmp2RowFull = tmp2ToDstFractalTail;
        tmp2NeedRow = tmp2RowFull;
    }

    /*
    overall solution: regard it as one column
    step1: src Brcb to tmp1
    step2: tmp1 DataCopy to tmp2, discard data that needs unpad
    step3: tmp2 TransDataTo5HD to tmp2, in-place
    step4: tmp2 DataCopy to dst, only need first line(block)
    */

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
    // each loop fill tmp2 and to dst(may more than once)
    for (uint32_t j = 0; j < height; j++) {
        for (uint32_t i = 0; i < (widthFractal); i++) {
            tmp2NeedRow = tmp2RowFull - tmp2RowCount;

            // load data to tmp1 from src
            if (i == (widthFractal - 1)) {
                DuplicateLastDimImpl(tmp1, srcTensor[j * width + i * widthTiling], widthFractalTail,
                    ONE_BLK_SIZE / sizeof(T));
            } else {
                DuplicateLastDimImpl(tmp1, srcTensor[j * width + i * widthTiling], widthTiling,
                    ONE_BLK_SIZE / sizeof(T));
            }

            // update tmp1RowCount and tmp1RowFull
            // first fractal, may have leftPad
            tmp1RowCount = (i == 0) ? leftPad : 0;
            // last fractal, may have rightPad
            tmp1RowFull = (i == (widthFractal - 1)) ? widthFractalTail - rightPad : 16 * tmp1BlockNum;

            PipeBarrier<PIPE_V>();
            while (tmp1RowCount != tmp1RowFull) {
                tmp1RemainRow = tmp1RowFull - tmp1RowCount;
                if (tmp2NeedRow > tmp1RemainRow) {
                    DataCopyParams repeatParams { 1, static_cast<uint16_t>(tmp1RemainRow), 0, 0 };
                    DataCopy(tmp2[tmp2RowCount * tmpWidth], tmp1[tmp1RowCount * tmpWidth], repeatParams);
                    tmp1RowCount += tmp1RemainRow;
                    tmp2RowCount += tmp1RemainRow;
                    tmp2NeedRow -= tmp1RemainRow;
                    break;
                } else {
                    DataCopyParams repeatParams { 1, static_cast<uint16_t>(tmp2NeedRow), 0, 0 };
                    DataCopy(tmp2[tmp2RowCount * tmpWidth], tmp1[tmp1RowCount * tmpWidth], repeatParams);
                    tmp1RowCount += tmp2NeedRow;
                    tmp2RowCount += tmp2NeedRow;
                    tmp2NeedRow = 0;
                }

                PipeBarrier<PIPE_V>();
                // tmp2 -> tmp2
                TransDataTo5HD<T>(dstList, srcList, transDataParams);
                PipeBarrier<PIPE_V>();
                // copy first line to dst
                if (tmp2ToDstCount == tmp2ToDstFractal) {
                    uint32_t repeatInUnit16 = tmp2ToDstFractalTail / 16;
                    if (sizeof(T) == sizeof(half)) {
                        DataCopy(dstTensor[tmp2ToDstCount * tmp1BlockNum * 16], tmp2,
                            { static_cast<uint16_t>(repeatInUnit16), 1, 15, 0 });
                    } else if (sizeof(T) == sizeof(float)) {
                        if (repeatInUnit16 != 0) {
                            DataCopy(dstTensor[tmp2ToDstCount * tmp1BlockNum * 16], tmp2,
                                { static_cast<uint16_t>(repeatInUnit16), 2, 14, 0 });
                        }
                        if (tmp2ToDstFractalTail % 16) {
                            DataCopy(dstTensor[tmp2ToDstCount * tmp1BlockNum * 16 + repeatInUnit16 * 16],
                                tmp2[repeatInUnit16 * (16 * ONE_BLK_SIZE / sizeof(T))], { 1, 1, 15, 0 });
                        }
                    }
                } else {
                    if (sizeof(T) == sizeof(half)) {
                        DataCopy(dstTensor[tmp2ToDstCount * tmp1BlockNum * 16], tmp2,
                            { static_cast<uint16_t>(tmp1BlockNum), 1, 15, 0 });
                    } else if (sizeof(T) == sizeof(float)) {
                        DataCopy(dstTensor[tmp2ToDstCount * tmp1BlockNum * 16], tmp2,
                            { static_cast<uint16_t>(tmp1BlockNum), 2, 14, 0 });
                    }
                }

                // clear tmp2RowCount and update tmp2NeedRow to full
                tmp2ToDstCount++;
                if (tmp2ToDstCount == tmp2ToDstFractal) {
                    tmp2RowFull = tmp2ToDstFractalTail;
                }
                tmp2RowCount = 0;
                tmp2NeedRow = tmp2RowFull - tmp2RowCount;
            }
        }
    }
}
} // namespace AscendC
#endif // IMPL_PAD_PAD_PAD_V200_IMPL_H
