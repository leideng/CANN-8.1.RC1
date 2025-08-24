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
* \file data_copy_wrapper_intf.h
* \brief
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_INTF_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_INTF_H

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, const auto& MM_CFG, class INPUT_TYPE, typename = void>
class DataCopyWrapper {
    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    __aicore__ inline DataCopyWrapper() = default;
    __aicore__ inline ~DataCopyWrapper() = default;

    // CopyND2NZ, support for V200
    template <bool IS_TRANS, typename SrcTensor, typename DstTensor>
    __aicore__ inline void CopyND2NZWithTransData(const DstTensor& dst, SrcTensor& src, const int row, const int col,
                                                  const int tileHeight, const int tileWidth) {}
    template <typename SrcTensor, typename DstTensor>
    __aicore__ inline void CopyND2NZOnTheFly(const DstTensor& dst, const SrcTensor& src, int row, int col, int height,
                                             int width, int gCol) {}
    template <bool IS_TRANS = false>
    __aicore__ inline void CopyND2NZWithVecOp(const LocalTensor<TransT>& dst, const GlobalTensor<SrcT>& src, const int row,
                                              const int col, const int height, const int width, const int gCol) {}

    // CopyND2NZ, support for V220
    __aicore__ inline void CopyND2NZ(const LocalTensor<TransT>& dst, const GlobalTensor<SrcT>& src,
        const int32_t row, const int32_t col, const int32_t height, const int32_t width, const int32_t gCol,
        const int32_t ndNum = 1, const int32_t srcNdMatrixStride = 0, const int32_t dstNzMatrixStride = 0,
        const bool kAlignToC0Size = false) {}
    __aicore__ inline void CopyND2NZ(const LocalTensor<TransT>& dst, const LocalTensor<SrcT>& src,
        const int32_t row, const int32_t col, const int32_t height, const int32_t width, const int32_t gCol) {}

    // CopyNZ2NZ, support for V200/V220
    __aicore__ inline void CopyNZ2NZ(const LocalTensor<TransT>& dst, const LocalTensor<SrcT>& src,
        const int32_t row, const int32_t col, const int32_t height, const int32_t width, const int32_t gRow) {}
    __aicore__ inline void CopyNZ2NZ(const LocalTensor<TransT>& dst, const GlobalTensor<SrcT>& src,
        const int32_t row, const int32_t col, const int32_t height, const int32_t width, const int32_t gRow,
        const bool kAlignToC0Size = false) {}

    // CopyVector2A1, support for V200/V220
    __aicore__ inline void CopyVector2A1(const LocalTensor<TransT>& dst, const GlobalTensor<SrcT>& src,
        const int32_t col, const int32_t blockLen) {}
    __aicore__ inline void CopyVector2A1(const LocalTensor<TransT>& dst, const LocalTensor<SrcT>& src,
        const int32_t col, const int32_t blockLen) {}
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_INTF_H
