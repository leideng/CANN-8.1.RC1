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
* \file data_copy_wrapper_vec.h
* \brief
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_VEC_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_VEC_H

#include "data_copy_wrapper_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, const auto& MM_CFG, class INPUT_TYPE>
class DataCopyWrapper<IMPL, MM_CFG, INPUT_TYPE, enable_if_t<INPUT_TYPE::format == CubeFormat::VECTOR>> {
    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    __aicore__ inline DataCopyWrapper() = default;
    __aicore__ inline ~DataCopyWrapper() = default;

    __aicore__ inline void CopyVector2A1(
        const LocalTensor<TransT>& dst, const GlobalTensor<SrcT>& src, const int32_t col, const int32_t blockLen)
    {
        ASCENDC_ASSERT((col >= 0), { KERNEL_LOG(KERNEL_ERROR, "col is %d, which should be no less than 0.", col); });
        ASCENDC_ASSERT((INPUT_TYPE::format == CubeFormat::VECTOR),
                    { KERNEL_LOG(KERNEL_ERROR, "INPUT_TYPE::format should be CubeFormat::VECTOR."); });

        DataCopyParams dataCopyInfo;
        dataCopyInfo.blockCount = 1;
        dataCopyInfo.blockLen = blockLen;
        dataCopyInfo.srcStride = 0;
        dataCopyInfo.dstStride = 0;
        DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = BlockMode::BLOCK_MODE_VECTOR;
        DataCopy(dst, src[col], dataCopyInfo, enhancedParams);
        return;
    }

    __aicore__ inline void CopyVector2A1(const LocalTensor<TransT>& dst, const LocalTensor<SrcT>& src,
        const int32_t col, const int32_t blockLen)
    {
        ASCENDC_ASSERT((col >= 0), { KERNEL_LOG(KERNEL_ERROR, "col is %d, which should be no less than 0.", col); });
        ASCENDC_ASSERT((INPUT_TYPE::format == CubeFormat::VECTOR),
                    { KERNEL_LOG(KERNEL_ERROR, "INPUT_TYPE::format should be CubeFormat::VECTOR."); });

        DataCopyParams dataCopyInfo;
        dataCopyInfo.blockCount = 1;
        dataCopyInfo.blockLen = blockLen;
        dataCopyInfo.srcStride = 0;
        dataCopyInfo.dstStride = 0;
        DataCopy(dst, src[col], dataCopyInfo);
        return;
    }
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_VEC_H
