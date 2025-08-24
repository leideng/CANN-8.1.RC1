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
 * \file batch_copy_cube_in_params.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_PARAMS_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_PARAMS_H

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, const auto &MM_CFG, class INPUT_TYPE>
class BatchCopyCubeInParams {
    using SrcT = typename INPUT_TYPE::T;
    using TransT = typename INPUT_TYPE::TRANS_T;
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(BatchLoop);
public:
    __aicore__ inline uint32_t GetBatchNum()
    {
        if (INPUT_TYPE::TAG == InputTypeTag::A) {
            return MATMUL_MODULE(BatchLoop)->GetBatchA();
        } else {
            return MATMUL_MODULE(BatchLoop)->GetBatchB();
        }
    }

    template <bool IS_TRANS = false>
    __aicore__ inline int32_t GetBatchOrgWidth()
    {
        // Get Head length of BSH or SBH layout
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            return GetBatchOrgWidthA<IS_TRANS>();
        } else{
            return GetBatchOrgWidthB<IS_TRANS>();
        }
    }

    __aicore__ inline bool IsTranspose()
    {
        if (INPUT_TYPE::TAG == InputTypeTag::A) {
            return MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
        } else {
            return MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();
        }
    }

    template <bool IS_TRANS = false, bool NEED_BASIC = true>
    __aicore__ inline int32_t GetSingleHeight() const
    {
        if constexpr (NEED_BASIC && IsBasic(MM_CFG)) {
            // false: not support intraBlock, true: is basic constantized scenario
            return MATMUL_MODULE(CopyCubeInParams)->template GetSingleHeight<IS_TRANS, false, true>();
        } else {
            return MATMUL_MODULE(CopyCubeInParams)->template GetSingleHeight<IS_TRANS, false, false>();
        }
    }

    template <bool IS_TRANS = false, bool NEED_BASIC = true>
    __aicore__ inline int32_t GetSingleWidth() const
    {
        if constexpr (NEED_BASIC && IsBasic(MM_CFG)) {
            // false: not support intraBlock, true: is basic constantized scenario
            return MATMUL_MODULE(CopyCubeInParams)->template GetSingleWidth<IS_TRANS, false, true>();
        } else {
            return MATMUL_MODULE(CopyCubeInParams)->template GetSingleWidth<IS_TRANS, false, false>();
        }
    }

    template <bool IS_TRANS = false, bool IS_KROW = false, bool NEED_BASIC = true>
    __aicore__ inline int64_t GetSingleSizeAlign() const
    {
        if constexpr (IS_KROW && IsSameTypeV<TransT, int8_t>) {
            return CeilAlign(GetSingleHeight<IS_TRANS, NEED_BASIC>(), c0Size_) *
                   CeilAlign(GetSingleWidth<IS_TRANS, NEED_BASIC>(), c0Size_);
        } else {
            return CeilAlign(GetSingleHeight<IS_TRANS, NEED_BASIC>(), BLOCK_CUBE) *
                   CeilAlign(GetSingleWidth<IS_TRANS, NEED_BASIC>(), c0Size_);
        }
    }

private:
    template <bool IS_TRANS = false>
    __aicore__ inline int32_t GetBatchOrgWidthA()
    {
        // Get Head length of BSH or SBH layout
        if constexpr (INPUT_TYPE::layout == LayoutMode::BSNGD) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetALayoutInfoD() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetALayoutInfoN() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetALayoutInfoG();
        } else if constexpr (INPUT_TYPE::layout == LayoutMode::SBNGD) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetALayoutInfoD() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetALayoutInfoN() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetALayoutInfoG() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetALayoutInfoB();
        } else {
            // Some operators does not set LayoutInfoS/D parameters for NORMAL/BNGS1S2 layout
            return MATMUL_MODULE(CopyCubeInParams)->template GetSingleWidth<IS_TRANS>();
        }
    }

    template <bool IS_TRANS = false>
    __aicore__ inline int32_t GetBatchOrgWidthB()
    {
        // Get Head length of BSH or SBH layout
        if constexpr (INPUT_TYPE::layout == LayoutMode::BSNGD) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBLayoutInfoD() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBLayoutInfoN() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBLayoutInfoG();
        } else if constexpr (INPUT_TYPE::layout == LayoutMode::SBNGD) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBLayoutInfoD() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBLayoutInfoN() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBLayoutInfoG() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBLayoutInfoB();
        } else {
            // Some operators does not set LayoutInfoS/D parameters for NORMAL/BNGS1S2 layout
            return MATMUL_MODULE(CopyCubeInParams)->template GetSingleWidth<IS_TRANS>();
        }
    }
    constexpr static int32_t c0Size_ = AuxGetC0Size<TransT>();
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_PARAMS_H
