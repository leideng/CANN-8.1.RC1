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
 * \file antiquant_processor_using_ub.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_ANTIQUANT_ANTIQUANT_PROCESSOR_USING_UB_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_ANTIQUANT_ANTIQUANT_PROCESSOR_USING_UB_H

#include "antiquant_processor_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
constexpr int32_t ANTI_QUANT_ALIGN_SIZE_MM_API = 32;

template <typename IMPL, class A_TYPE, class B_TYPE, const auto& MM_CFG>
class MatmulAntiQuantProcessor<IMPL, A_TYPE, B_TYPE, MM_CFG, enable_if_t<MatmulFeatureTrait<MM_CFG>::IsNeedUB() &&
    IsSameTypeV<typename A_TYPE::T, half> && IsSameTypeV<typename B_TYPE::T, int8_t>>>
{
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(LocalWorkspace);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);

    using TransT = typename A_TYPE::T;
    using SrcT = typename B_TYPE::T;
public:
    __aicore__ inline MatmulAntiQuantProcessor() {}
    __aicore__ inline ~MatmulAntiQuantProcessor() {}

    __aicore__ inline void SetAntiQuantScalar(const TransT offsetScalar, const TransT scaleScalar)
    {
        antiQuantOffsetScalar_ = offsetScalar;
        antiQuantScaleScalar_ = scaleScalar;
    }

    __aicore__ inline void SetAntiQuantVector(
        const LocalTensor<TransT>& offsetTensor, const LocalTensor<TransT>& scaleTensor)
    {
        antiQuantOffsetTensor_ = offsetTensor;
        antiQuantScaleTensor_ = scaleTensor;
    }

    __aicore__ inline void AntiQuantCompute(const LocalTensor<TransT>& quantOut, const LocalTensor<SrcT>& quantIn,
                                            bool isBankConflict, bool isTranspose, int32_t baseWidth)
    {
#if __CCE_AICORE__ == 200
        LocalTensor<uint8_t> sharedLocal = GetSharedLocal(isTranspose, baseWidth);
        if constexpr (ToMatmulConfig(MM_CFG).isPerTensor) {
            AntiQuantComputePerTensor(quantOut, quantIn, sharedLocal, isTranspose);
        } else {
            uint32_t groupNum = 1;
            AntiQuantShapeInfo shapeInfo;
            if (isTranspose) {
                uint32_t quantN = CeilAlignT<int32_t>(MATMUL_MODULE(NLoop)->GetBaseShape(), ANTI_QUANT_ALIGN_SIZE_MM_API);
                if constexpr (ToMatmulConfig(MM_CFG).hasAntiQuantOffset) {
                    shapeInfo.offsetHeight = quantN;
                    shapeInfo.offsetWidth = groupNum;
                }
                shapeInfo.scaleHeight = quantN;
                shapeInfo.scaleWidth = groupNum;
            } else {
                int quantN = CeilAlignT<int32_t>(MATMUL_MODULE(NLoop)->GetBaseShape(), ANTI_QUANT_ALIGN_SIZE_MM_API);
                uint32_t padNSize = isBankConflict ? quantN + ANTI_QUANT_ALIGN_SIZE_MM_API : quantN;
                if constexpr (ToMatmulConfig(MM_CFG).hasAntiQuantOffset) {
                    shapeInfo.offsetHeight = groupNum;
                    shapeInfo.offsetWidth = padNSize;
                }
                shapeInfo.scaleHeight = groupNum;
                shapeInfo.scaleWidth = padNSize;
            }
            if constexpr (ToMatmulConfig(MM_CFG).hasAntiQuantOffset) {
                if (isTranspose) {
                    AscendAntiQuant<SrcT, TransT, true>(quantOut, quantIn, antiQuantOffsetTensor_,
                        antiQuantScaleTensor_, sharedLocal, CeilAlignT<int32_t>(
                            MATMUL_MODULE(KLoop)->GetTileShapeB(), ANTI_QUANT_ALIGN_SIZE_MM_API), shapeInfo);
                } else {
                    AscendAntiQuant<SrcT, TransT, false>(quantOut, quantIn, antiQuantOffsetTensor_,
                        antiQuantScaleTensor_, sharedLocal, CeilAlignT<int32_t>(
                            MATMUL_MODULE(KLoop)->GetTileShapeB(), ANTI_QUANT_ALIGN_SIZE_MM_API), shapeInfo);
                }
            } else {
                if (isTranspose) {
                    AscendAntiQuant<SrcT, TransT, true>(quantOut, quantIn, antiQuantScaleTensor_, sharedLocal,
                        CeilAlignT<int32_t>(MATMUL_MODULE(KLoop)->GetTileShapeB(), ANTI_QUANT_ALIGN_SIZE_MM_API),
                        shapeInfo);
                } else {
                    AscendAntiQuant<SrcT, TransT, false>(quantOut, quantIn, antiQuantScaleTensor_, sharedLocal,
                        CeilAlignT<int32_t>(MATMUL_MODULE(KLoop)->GetTileShapeB(), ANTI_QUANT_ALIGN_SIZE_MM_API),
                        shapeInfo);
                }
            }
        }
#endif
    }

    __aicore__ inline void AntiQuantComputePerTensor(const LocalTensor<TransT>& quantOut,
        const LocalTensor<SrcT>& quantIn, const LocalTensor<uint8_t>& sharedLocal, bool isTranspose)
    {
#if __CCE_AICORE__ == 200
        if constexpr (ToMatmulConfig(MM_CFG).hasAntiQuantOffset) {
            if (isTranspose) {
                AscendAntiQuant<SrcT, TransT, true>(
                    quantOut, quantIn, antiQuantOffsetScalar_,
                    antiQuantScaleScalar_, sharedLocal,
                    CeilAlignT<int32_t>(MATMUL_MODULE(KLoop)->GetTileShapeB(), ANTI_QUANT_ALIGN_SIZE_MM_API));
            } else {
                AscendAntiQuant<SrcT, TransT, false>(
                    quantOut, quantIn, antiQuantOffsetScalar_,
                    antiQuantScaleScalar_, sharedLocal,
                    CeilAlignT<int32_t>(MATMUL_MODULE(KLoop)->GetTileShapeB(), ANTI_QUANT_ALIGN_SIZE_MM_API));
            }
        } else {
            if (isTranspose) {
                AscendAntiQuant<SrcT, TransT, true>(
                    quantOut, quantIn, antiQuantScaleScalar_, sharedLocal,
                    CeilAlignT<int32_t>(MATMUL_MODULE(KLoop)->GetTileShapeB(), ANTI_QUANT_ALIGN_SIZE_MM_API));
            } else {
                AscendAntiQuant<SrcT, TransT, false>(
                    quantOut, quantIn, antiQuantScaleScalar_, sharedLocal,
                    CeilAlignT<int32_t>(MATMUL_MODULE(KLoop)->GetTileShapeB(), ANTI_QUANT_ALIGN_SIZE_MM_API));
            }
        }
#endif
    }

    __aicore__ inline LocalTensor<uint8_t> GetSharedLocal(bool isTranspose, int32_t baseWidth)
    {
        LocalTensor<uint8_t> sharedLocal;
        if (isTranspose) {
            int32_t scaleUbSize = baseWidth * 2 + 32 * 2;
            int32_t tmpBuffSize = 16 * baseWidth * 2 * sizeof(TransT);
            ASCENDC_ASSERT((MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength() > (tmpBuffSize + scaleUbSize)), {
                KERNEL_LOG(KERNEL_ERROR, "transLength(%d) must be larger than tmpBuffSize(%d) + scaleUbSize(%d)",
                           MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength(), tmpBuffSize, scaleUbSize);
            });
            int32_t tmpBuffOffset = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength() - tmpBuffSize - scaleUbSize;
            sharedLocal =
                MATMUL_MODULE(LocalWorkspace)->GetWorkspaceWithOffset(tmpBuffOffset).template ReinterpretCast<uint8_t>();
            sharedLocal.SetSize(tmpBuffSize);
        }
        return sharedLocal;
    }

private:
    TransT antiQuantOffsetScalar_;
    TransT antiQuantScaleScalar_;
    LocalTensor<TransT> antiQuantOffsetTensor_;
    LocalTensor<TransT> antiQuantScaleTensor_;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_USING_UB_H