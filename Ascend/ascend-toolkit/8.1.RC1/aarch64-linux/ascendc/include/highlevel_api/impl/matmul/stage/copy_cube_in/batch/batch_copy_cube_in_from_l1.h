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
 * \file batch_copy_cube_in_from_l1.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_FROM_L1_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_FROM_L1_H

#include "batch_copy_cube_in_intf.h"
#include "batch_copy_cube_in_params.h"
#include "../copy_tile_to_cube/copy_tile_to_cube.h"
#include "../base/copy_cube_in_params.h"

namespace AscendC {
namespace Impl {
namespace Detail {
// Specialized Template Class of Batch Matmul CopyIn
// Batch Matmul ND Format Data CopyIn From L1
template <typename IMPL, const auto& MM_CFG, class INPUT_TYPE>
class BatchCopyCubeIn<IMPL, MM_CFG, INPUT_TYPE, enable_if_t<
    !MatmulFeatureTrait<MM_CFG>::IsNeedUB() &&
    IsBMMFromL1<INPUT_TYPE, MM_CFG>()>>
{
    MATMUL_USE_MODULE_ON(BatchCopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeInfo);

    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    inline __aicore__ BatchCopyCubeIn() = default;
    inline __aicore__ ~BatchCopyCubeIn() = default;

    __aicore__ inline void Init() {}

    __aicore__ inline void SetInput(const GlobalTensor<SrcT>& globalMatrix, bool isTranspose = false)
    {}

    __aicore__ inline void SetInput(const LocalTensor<SrcT>& localMatrix, bool isTranspose = false)
    {}

    __aicore__ inline void BatchLoad(LocalTensor<TransT>& dstTensor, const uint32_t matrixStride,
                                     const int32_t outerIdx, const int32_t splitIdx, const int32_t splitSize)
    {
        if (MATMUL_MODULE(BatchCopyCubeInParams)->IsTranspose()) {
            GetBatchMatrix<true, INPUT_TYPE::TAG == InputTypeTag::A>(
                dstTensor, matrixStride, outerIdx, splitIdx, splitSize);
        } else {
            GetBatchMatrix<false, INPUT_TYPE::TAG == InputTypeTag::B>(
                dstTensor, matrixStride, outerIdx, splitIdx, splitSize);
        }
    }

    template <typename ScheduleContext = int>
    __aicore__ inline LocalTensor<TransT> LoadData(
        int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth, const ScheduleContext& context = 0)
    {}

    __aicore__ inline void BatchDestroy() {}

    __aicore__ inline LocalTensor<TransT> AllocTensor(int32_t iterIndex = 0)
    {
        LocalTensor<TransT> localTensor;
        localTensor.SetAddr(MATMUL_MODULE(MatmulTensorInfo)->GetLocalTensor().address_);
        return localTensor;
    }

    __aicore__ inline void ClearLoadData(const LocalTensor<TransT>& tensor = NULL_TENSOR<TransT>,
        int32_t curRow = 0, int32_t curCol = 0)
    {}

    __aicore__ inline void Destroy() {}

    __aicore__ inline void Reset() {}

private:
    template <bool IS_TRANS = false, bool IS_KROW = false>
    __aicore__ inline void GetBatchMatrix(LocalTensor<TransT>& dstTensor, const uint32_t matrixStride,
                                           const int32_t outerIdx, const int32_t splitIdx, const int32_t splitSize)
    {
        // L1 input will get data at once, so no need to spilt
        if (splitIdx > 0) {
            return;
        }
        // Calculate batch outer loop offset
        int64_t batchOffset = outerIdx * GetBatchSize();

        dstTensor = dstTensor[batchOffset];
        dstTensor.SetSize(GetBatchSize());
    }

    __aicore__ inline int32_t GetBatchSize()
    {
        if constexpr (INPUT_TYPE::format == CubeFormat::ND) {
            return MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() * GetSingleSizeAlign<INPUT_TYPE::isTrans>();
        } else if constexpr (INPUT_TYPE::format == CubeFormat::NZ) {
            if constexpr (INPUT_TYPE::isTrans) {
                return MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() *
                    GetSingleSizeAlign<true, INPUT_TYPE::TAG == InputTypeTag::A>();
            } else {
                return MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() *
                    GetSingleSizeAlign<false, INPUT_TYPE::TAG == InputTypeTag::B>();
            }
        }
    }

    template <bool IS_TRANS = false, bool IS_KROW = false>
    __aicore__ inline int64_t GetSingleSizeAlign() const
    {
        if constexpr (INPUT_TYPE::format == CubeFormat::ND) {
            return MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleSizeAlign<IS_TRANS, false>();
        } else if constexpr (INPUT_TYPE::format == CubeFormat::NZ) {
            return MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleSizeAlign<IS_TRANS, IS_KROW>();
        }
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_FROM_L1_H