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
 * \file batch_copy_cube_in.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_H

#include "batch_copy_cube_in_intf.h"
#include "batch_copy_cube_in_params.h"
#include "batch_copy_cube_in_base.h"
#include "../copy_tile_to_cube/copy_tile_to_cube.h"
#include "../base/copy_cube_in_params.h"

namespace AscendC {
namespace Impl {
namespace Detail {
// Specialized Template Class of Batch Matmul CopyIn
// Batch Matmul ND Format Data CopyIn From GM/UB
template <typename IMPL, const auto& MM_CFG, class INPUT_TYPE>
class BatchCopyCubeIn<IMPL, MM_CFG, INPUT_TYPE, enable_if_t<
    !MatmulFeatureTrait<MM_CFG>::IsNeedUB() &&
    GetCopyCubeInType<INPUT_TYPE, MM_CFG>() == CopyCubeInType::BMM &&
    INPUT_TYPE::format == CubeFormat::ND>>
    : public BatchCopyCubeInBase<IMPL, MM_CFG, INPUT_TYPE>
{
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(BatchCopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(DataCopyWrapper, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulShapeTiling);

    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    using BASE_MODULE = AscendC::Impl::Detail::BatchCopyCubeInBase<IMPL, MM_CFG, INPUT_TYPE>;

    inline __aicore__ BatchCopyCubeIn() = default;
    inline __aicore__ ~BatchCopyCubeIn() = default;

    __aicore__ inline void Init()
    {
        MATMUL_MODULE(CubeInBuffer)->Init(
            MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() * MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleSizeAlign<INPUT_TYPE::isTrans>(), 1);
    }

    __aicore__ inline void BatchLoad(LocalTensor<TransT>& dstTensor, const uint32_t matrixStride,
                                     const int32_t outerIdx, const int32_t splitIdx, const int32_t splitSize)
    {
        if (MATMUL_MODULE(BatchCopyCubeInParams)->IsTranspose()) {
            return CopyBatchToCubeND<true, INPUT_TYPE::TAG == InputTypeTag::A>(
                    dstTensor, matrixStride, outerIdx, splitIdx, splitSize);
        } else {
            return CopyBatchToCubeND<false, INPUT_TYPE::TAG == InputTypeTag::B>(
                dstTensor, matrixStride, outerIdx, splitIdx, splitSize);
        }
    }

private:
    template <bool IS_TRANS = false, bool IS_KROW = false>
    __aicore__ inline void CopyBatchToCubeND(LocalTensor<TransT>& dstTensor, const uint32_t matrixStride,
                                           const int32_t outerIdx, const int32_t splitIdx, const int32_t splitSize )
    {
        // Calculate batch outer loop offset
        // the parameter false means don't need to use constant parameters
        int64_t batchOffset = outerIdx * GetSingleSize<IS_TRANS, false>() *
                              MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum();

        // Calculate iter numbers by line of BSNGD layout
        int32_t batchNum = MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum(); // batchA_ or batchB_
        int32_t iterNum = 1;
        UpdataBatchNum(batchNum, iterNum);
        batchNum /= splitSize;

        // Calculate srcDValue for ND copy
        auto srcDValue = MATMUL_MODULE(BatchCopyCubeInParams)->template GetBatchOrgWidth<IS_TRANS>();

        // Calculate src and dst stride of one step
        // if user input matrixStride, use matrixStride as srcStride
        auto srcStride = matrixStride != 0 ? matrixStride : GetSrcStride<IS_TRANS, false>();
        auto dstStride =  MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleSizeAlign<IS_TRANS>();
        int64_t srcOffset = batchNum * splitIdx * srcStride;
        int64_t dstOffset = batchNum * splitIdx * dstStride;

        // Calculate src and dst stride of one line
        auto iterSrcStride = batchNum * GetSingleSize<IS_TRANS, false>();
        auto iterDstStride = batchNum * GetSingleSize<IS_TRANS>();

        // Complete datacopy by line
        GlobalTensor<SrcT> srcGlobal;
        srcGlobal.SetGlobalBuffer(MATMUL_MODULE(MatmulTensorInfo)->GetGlobalTensor().address_);
        srcGlobal.SetAddr(batchOffset);
        for (int32_t idx = 0; idx < iterNum; ++idx) {
            if (srcStride >= UINT16_MAX) {
                for (int i = 0; i < batchNum; ++i) {
                    MATMUL_MODULE(DataCopyWrapper)->CopyND2NZ(
                        dstTensor[dstOffset], srcGlobal[srcOffset], 0, 0,
                        MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<IS_TRANS>(),
                        MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<IS_TRANS>(), srcDValue);
                    dstOffset += dstStride;
                    srcOffset += srcStride;
                }
            } else {
                MATMUL_MODULE(DataCopyWrapper)->CopyND2NZ(
                    dstTensor[dstOffset], srcGlobal[srcOffset], 0, 0,
                    MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<IS_TRANS>(),
                    MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<IS_TRANS>(),
                    srcDValue, batchNum, srcStride, dstStride);
            }
            dstOffset += iterDstStride;
            srcOffset += iterSrcStride;
        }
    }

     __aicore__ inline void UpdataBatchNum(int32_t &batchNum, int32_t &iterNum)
     {
        if constexpr (INPUT_TYPE::layout == LayoutMode::BSNGD) {
            ASCENDC_ASSERT((IsLayoutGValid()), {
                KERNEL_LOG(KERNEL_ERROR, "multi batch calculation of multiple lines of S is not supported");
            });
            // if batchNum > LayoutN * LayoutG, need copy by single line
            if (batchNum > GetLayoutInfoNG()) {
                // update batchnum to single line batch number
                batchNum = GetLayoutInfoNG();
                iterNum = Ceil(MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum(), batchNum);
            }
        }
     }

    __aicore__ inline int32_t GetLayoutInfoNG()
    {
        if constexpr(INPUT_TYPE::TAG == InputTypeTag::A) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetALayoutInfoN() *
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetALayoutInfoG();
        } else {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBLayoutInfoN() *
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBLayoutInfoG();
        }
    }

    template <bool IS_TRANS = false, bool NEED_BASIC = true>
    __aicore__ inline int64_t GetSingleSize() const
    {
        return MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<IS_TRANS, NEED_BASIC>() *
            MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<IS_TRANS, NEED_BASIC>();
    }

    // ND format, src data default don't need to use constant parameters
    template <bool IS_TRANS = false, bool NEED_BASIC = true>
    __aicore__ inline int64_t GetSrcStride()
    {
        if constexpr (INPUT_TYPE::layout == LayoutMode::BSNGD || INPUT_TYPE::layout == LayoutMode::SBNGD) {
            // BSNGD/SBNGD layout memory is not contiguous
            if constexpr (PhyPosIsUB(INPUT_TYPE::pos)) {
                return CeilAlign(MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<IS_TRANS, NEED_BASIC>(), c0Size_);
            } else {
                return MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<IS_TRANS, NEED_BASIC>();
            }
        } else {
            // NORMAL/BNGS1S2 layout memory is contiguous
            if constexpr (PhyPosIsUB(INPUT_TYPE::pos)) {
                return MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleSizeAlign<IS_TRANS, false, NEED_BASIC>();
            } else {
                return GetSingleSize<IS_TRANS, NEED_BASIC>();
            }
        }
    }

    __aicore__ inline bool IsLayoutGValid()
    {
        auto maxLayoutInfoG = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetALayoutInfoG() >
                              MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBLayoutInfoG() ?
                              MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetALayoutInfoG() :
                              MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBLayoutInfoG();
        if constexpr(INPUT_TYPE::TAG == InputTypeTag::A) {
            return MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() <=
                   (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetALayoutInfoN() * maxLayoutInfoG);
        } else {
            return MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() <=
                   (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBLayoutInfoN() * maxLayoutInfoG);
        }
    }
private:
    constexpr static int32_t c0Size_ = AuxGetC0Size<TransT>();
};

// Specialized Template Class of Batch Matmul CopyIn
// Batch Matmul NZ Format Data CopyIn From GM/UB, support LayoutMode NORMAL/BNGS1S2
template <typename IMPL, const auto& MM_CFG, class INPUT_TYPE>
class BatchCopyCubeIn<IMPL, MM_CFG, INPUT_TYPE, enable_if_t<
    (!MatmulFeatureTrait<MM_CFG>::IsNeedUB()) &&
    GetCopyCubeInType<INPUT_TYPE, MM_CFG>() == CopyCubeInType::BMM &&
    INPUT_TYPE::format == CubeFormat::NZ &&
    ((INPUT_TYPE::layout == LayoutMode::NORMAL) || (INPUT_TYPE::layout == LayoutMode::BNGS1S2))>>
    : public BatchCopyCubeInBase<IMPL, MM_CFG, INPUT_TYPE>
{
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(BatchCopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(DataCopyWrapper, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);

    using SrcT = typename INPUT_TYPE::T;
    using TransT = typename INPUT_TYPE::TRANS_T;

public:
    using BASE_MODULE = AscendC::Impl::Detail::BatchCopyCubeInBase<IMPL, MM_CFG, INPUT_TYPE>;

    inline __aicore__ BatchCopyCubeIn() = default;
    inline __aicore__ ~BatchCopyCubeIn() = default;

    __aicore__ inline void Init()
    {
        if constexpr (INPUT_TYPE::isTrans) {
            MATMUL_MODULE(CubeInBuffer)->Init(MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() *
                MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleSizeAlign<true, INPUT_TYPE::TAG == InputTypeTag::A>(), 1);
        } else {
            MATMUL_MODULE(CubeInBuffer)->Init(MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() *
                MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleSizeAlign<false, INPUT_TYPE::TAG == InputTypeTag::B>(), 1);
        }
    }

    __aicore__ inline void BatchLoad(LocalTensor<TransT>& dstTensor, const uint32_t matrixStride,
                                     const int32_t outerIdx, const int32_t splitIdx, const int32_t splitSize)
    {
        if (MATMUL_MODULE(BatchCopyCubeInParams)->IsTranspose()) {
            CopyBatchToCubeNZ<true, INPUT_TYPE::TAG == InputTypeTag::A>(
                dstTensor, outerIdx, splitIdx, splitSize);
        } else {
            CopyBatchToCubeNZ<false, INPUT_TYPE::TAG == InputTypeTag::B>(
                dstTensor, outerIdx, splitIdx, splitSize);
        }
    }

private:
    template <bool IS_TRANS = false, bool IS_KROW = false>
    __aicore__ inline void CopyBatchToCubeNZ(LocalTensor<TransT>& dstTensor, const int32_t outerIdx,
                                           const int32_t splitIdx, const int32_t splitSize)
    {
        // 1. Calculate batch outer loop offset
        // NZ does not support tail block scenarios, src also uses constantized data
        auto alignHeight = CeilAlign(MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<IS_TRANS>(), BLOCK_CUBE);
        auto alignWidth = CeilAlign(MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<IS_TRANS>(), c0Size_);

        // 2. Calculate src and dst stride of one step
        auto batchNum = MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() / splitSize;
        int64_t srcStride = alignWidth * alignHeight;
        int64_t dstStride = MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleSizeAlign<IS_TRANS, IS_KROW>();
        int64_t srcOffset = batchNum * splitIdx * srcStride;
        int64_t dstOffset = batchNum * splitIdx * dstStride;

        // 3. loop copy NZ data by batch
        bool iskRowDirec = IS_KROW && IsSameTypeV<TransT, int8_t>;
        auto batchOffset = outerIdx * MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() * srcStride;
        GlobalTensor<SrcT> srcGlobal;
        srcGlobal.SetGlobalBuffer(MATMUL_MODULE(MatmulTensorInfo)->GetGlobalTensor().address_);
        srcGlobal.SetAddr(batchOffset);
        for (int i = 0; i < batchNum; ++i) {
            MATMUL_MODULE(DataCopyWrapper)->CopyNZ2NZ(
                dstTensor[dstOffset], srcGlobal[srcOffset], 0, 0,
                alignHeight, alignWidth, alignHeight, iskRowDirec);
            dstOffset += dstStride;
            srcOffset += srcStride;
        }
    }
private:
    constexpr static int32_t c0Size_ = AuxGetC0Size<TransT>();
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_H