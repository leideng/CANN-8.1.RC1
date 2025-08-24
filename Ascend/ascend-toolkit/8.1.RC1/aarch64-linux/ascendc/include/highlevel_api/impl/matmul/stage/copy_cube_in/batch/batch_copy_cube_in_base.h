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
* \file batch_copy_cube_in_base.h
* \brief
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_BASE_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_BASE_H

namespace AscendC {
namespace Impl {
namespace Detail {

#include "../../../utils/matmul_module.h"

template <typename IMPL, const auto &MM_CFG, class INPUT_TYPE>
class BatchCopyCubeInBase
{
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);

    using SrcT = typename INPUT_TYPE::T;
    using TransT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline void SetInput(const GlobalTensor<SrcT>& globalMatrix, bool isTranspose = false)
    {
        MATMUL_MODULE(MatmulTensorInfo)->SetGlobalTensor(globalMatrix, isTranspose);
        MATMUL_MODULE(CubeInBuffer)->Reset();
    }

    __aicore__ inline void SetInput(const LocalTensor<SrcT>& localMatrix, bool isTranspose = false)
    {}

    __aicore__ inline void Reset()
    {
        MATMUL_MODULE(CubeInBuffer)->Reset();
    }

    template <typename ScheduleContext = int>
    __aicore__ inline LocalTensor<TransT> LoadData(
        int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth, const ScheduleContext& context = 0)
    {
        LocalTensor<TransT> localTensor;
        localTensor.SetAddr(MATMUL_MODULE(MatmulTensorInfo)->GetLocalTensor().address_);
        return localTensor;
    }

    __aicore__ inline void ClearLoadData(const LocalTensor<TransT>& tensor = NULL_TENSOR<TransT>,
        int32_t curRow = 0, int32_t curCol = 0)
    {}

    __aicore__ inline void BatchDestroy()
    {
        MATMUL_MODULE(CubeInBuffer)->FreeTensor();
        MATMUL_MODULE(CubeInBuffer)->Destroy();
    }

    __aicore__ inline LocalTensor<TransT> AllocTensor(int32_t iterIndex = 0)
    {
        return MATMUL_MODULE(CubeInBuffer)->AllocTensor(iterIndex);
    }

    __aicore__ inline void Destroy()
    {
        MATMUL_MODULE(CubeInBuffer)->Destroy();
    }
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _BATCH_COPY_CUBE_IN_BASE_H_
