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
 * \file copy_cube_in_mdl.h
 * \brief
 */


#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_BASE_COPY_CUBE_IN_MDL_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_BASE_COPY_CUBE_IN_MDL_H

#include "../copy_tile_to_cube/copy_tile_to_cube.h"
#include "copy_cube_in_intf.h"
#include "copy_cube_in_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    CopyCubeIn is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    CopyCubeIn is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class CopyCubeIn<IMPL, INPUT_TYPE, MM_CFG, enable_if_t<
!MatmulFeatureTrait<MM_CFG>::IsNeedUB() && GetCopyCubeInType<INPUT_TYPE, MM_CFG>() == CopyCubeInType::MDL>>
: public CopyCubeInBase<IMPL, MM_CFG, INPUT_TYPE>
{
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(DataCopyUtils, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    using SrcT = typename INPUT_TYPE::T;
    using TransT = typename INPUT_TYPE::TRANS_T;

public:
    using BASE_MODULE = AscendC::Impl::Detail::CopyCubeInBase<IMPL, MM_CFG, INPUT_TYPE>;
    __aicore__ inline CopyCubeIn() = default;
    __aicore__ inline ~CopyCubeIn() = default;

    template <typename ScheduleContext = int>
    __aicore__ inline LocalTensor<TransT> LoadData(
        int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth, const ScheduleContext& context = {})
    {
        LocalTensor<TransT> l1;
        int32_t posL1 = BASE_MODULE::GetIterIndex(curRow, curCol);
        int32_t bufferPos = MATMUL_MODULE(CopyCubeInParams)->GetBufferPos();
        if (MATMUL_MODULE(CubeInBuffer)->Hit(posL1, bufferPos)) {
            l1 = MATMUL_MODULE(CubeInBuffer)->GetBuffer(posL1, bufferPos);
        } else {
            l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(bufferPos);
            MATMUL_MODULE(DataCopyUtils)->CopyTileToCube(l1, curRow, curCol, tileHeight, tileWidth);
            MATMUL_MODULE(CubeInBuffer)->EnQue(l1);
            MATMUL_MODULE(CubeInBuffer)->DeQue();
        }
        return l1;
    }

    template <typename ScheduleContext = int>
    __aicore__ inline LocalTensor<TransT> AsyncLoadData(
        int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth, const ScheduleContext& context = {})
    {
        if constexpr (PhyPosIsL1(INPUT_TYPE::pos) || INPUT_TYPE::layout != LayoutMode::NONE) {
            ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR,
                "Matching error. MDL AsyncLoadData doesn't support BMM && Src L1"); });
        }

        LocalTensor<TransT> l1;
        int32_t posL1 = BASE_MODULE::GetIterIndex(curRow, curCol);
        int32_t bufferPos;
        if constexpr (DoMatmulMDL(MM_CFG)) {
            bufferPos = MATMUL_MODULE(CopyCubeInParams)->GetBufferPos() + 1;
        } else {
            bufferPos = MATMUL_MODULE(CopyCubeInParams)->GetBufferPos();
        }
        if (MATMUL_MODULE(CubeInBuffer)->Hit(posL1, bufferPos)) {
            return MATMUL_MODULE(CubeInBuffer)->GetBuffer(posL1, bufferPos);
        } else {
            l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(bufferPos);
            MATMUL_MODULE(DataCopyUtils)->CopyTileToCube(l1, curRow, curCol, tileHeight, tileWidth);
            MATMUL_MODULE(CubeInBuffer)->EnQue(l1);
            return l1;
        }
    }

    __aicore__ inline void ClearLoadData(const LocalTensor<TransT>& tensor = NULL_TENSOR<TransT>,
        int32_t curRow = 0, int32_t curCol = 0)
    {
        auto bufferPos = MATMUL_MODULE(CopyCubeInParams)->GetBufferPos();
        MATMUL_MODULE(CubeInBuffer)->FreeTensor(bufferPos);
    }

    __aicore__ inline void AwaitLoadData()
    {
        MATMUL_MODULE(CubeInBuffer)->DeQue();
    }
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _COPY_CUBE_IN_MDL_H_
