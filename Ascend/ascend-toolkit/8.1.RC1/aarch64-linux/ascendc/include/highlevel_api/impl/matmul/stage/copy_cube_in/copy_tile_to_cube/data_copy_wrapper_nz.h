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
* \file data_copy_wrapper_nz.h
* \brief
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_NZ_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_NZ_H

#include "data_copy_wrapper_intf.h"
#include "data_copy_wrapper_utils.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, const auto& MM_CFG, class INPUT_TYPE>
class DataCopyWrapper<IMPL, MM_CFG, INPUT_TYPE, enable_if_t<INPUT_TYPE::format == CubeFormat::NZ>> {
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(LocalWorkspace);

    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    __aicore__ inline DataCopyWrapper() = default;
    __aicore__ inline ~DataCopyWrapper() = default;

    __aicore__ inline void CopyNZ2NZ(const LocalTensor<TransT>& dst, const GlobalTensor<SrcT>& src,
        const int32_t row, const int32_t col, const int32_t height, const int32_t width, const int32_t gRow,
        const bool kAlignToC0Size = false)
    {
        CopyNZ2NZImpl(dst, src, row, col, height, width, gRow, kAlignToC0Size);
    }

    __aicore__ inline void CopyNZ2NZ(const LocalTensor<TransT>& dst, const LocalTensor<SrcT>& src,
        const int32_t row, const int32_t col, const int32_t height, const int32_t width, const int32_t gRow)
    {
        CopyNZ2NZImpl(dst, src, row, col, height, width, gRow);
    }

    template <bool IS_TRANS = false>
    __aicore__ void CopyNZ2NZWithTransData(const LocalTensor<TransT>& dst, LocalTensor<SrcT>& src, int row, int col,
                                           int tileHeight, int tileWidth)
    {
        int64_t size = tileHeight * tileWidth;
        LocalTensor<TransT> trans =
            MATMUL_MODULE(LocalWorkspace)->GetND2NZWorkspace(size).template ReinterpretCast<TransT>();
        trans.SetSize(size);
        int srcOffset = row * MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>() * c0Size_ +
                        col * MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>() *
                            MATMUL_MODULE(CopyCubeInParams)->template GetOrgHeight<IS_TRANS>();
        TransDataNZBMatrix<SrcT, TransT, IS_TRANS>(trans, src[srcOffset], tileHeight, tileWidth);
        event_t eventIDVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMte3);
        CopyNZ2NZImpl(dst, trans, 0, 0, tileWidth, tileHeight, tileWidth);
    }

    template <bool IS_TRANS = false>
    __aicore__ void CopyNZ2NZWithTransData(const LocalTensor<TransT>& dst, GlobalTensor<SrcT>& src, int row, int col,
                                           int tileHeight, int tileWidth)
    {
        int calcWidth = CeilT(tileWidth, c0Size_) * c0Size_;
        int calcHigh = CeilT(tileHeight, c0Size_) * c0Size_;
        int64_t size = calcHigh * calcWidth;
        LocalTensor<TransT> rightMatrix =
            MATMUL_MODULE(LocalWorkspace)->GetND2NZWorkspace(0).template ReinterpretCast<TransT>();
        rightMatrix.SetSize(size);
        int srcOffset = row * MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>() * c0Size_ +
                        col * MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>() *
                            MATMUL_MODULE(CopyCubeInParams)->template GetOrgHeight<IS_TRANS>();
        int dstOffset = 0;
        int srcHigh = CeilT(MATMUL_MODULE(CopyCubeInParams)->template GetOrgHeight<IS_TRANS>(), 16) * 16 * c0Size_;
        int dstHigh = tileHeight < c0Size_ ? tileHeight * c0Size_ : calcHigh * c0Size_;
        for (int i = 0; i < CeilT(tileWidth, c0Size_); i++) {
            DataCopy(rightMatrix[dstOffset], src[srcOffset], dstHigh);
            srcOffset += srcHigh;
            dstOffset += dstHigh;
        }
        LocalTensor<TransT> trans =
            MATMUL_MODULE(LocalWorkspace)->GetND2NZWorkspace(size).template ReinterpretCast<TransT>();
        trans.SetSize(size);
        event_t eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
        TransDataNZBMatrix<SrcT, TransT, IS_TRANS>(trans, rightMatrix, tileHeight, tileWidth);
        event_t eventIDVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMte3);
        CopyNZ2NZImpl(dst, trans, 0, 0, calcWidth, calcHigh, calcWidth);
    }

private:
    constexpr static int32_t c0Size_ = AuxGetC0Size<SrcT>();
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_NZ_H
