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
* \file data_copy_wrapper_nd.h
* \brief
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_ND_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_ND_H

#include "data_copy_wrapper_intf.h"
#include "data_copy_wrapper_utils.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, const auto& MM_CFG, class INPUT_TYPE>
class DataCopyWrapper<IMPL, MM_CFG, INPUT_TYPE,
                      enable_if_t<!MatmulFeatureTrait<MM_CFG>::IsNeedUB() && INPUT_TYPE::format == CubeFormat::ND>> {
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(LocalWorkspace);

    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    __aicore__ inline DataCopyWrapper() = default;
    __aicore__ inline ~DataCopyWrapper() = default;

    __aicore__ inline void CopyND2NZ(const LocalTensor<TransT>& dst, const GlobalTensor<SrcT>& src,
        const int32_t row, const int32_t col, const int32_t height, const int32_t width, const int32_t gCol,
        const int32_t ndNum = 1, const int32_t srcNdMatrixStride = 0, const int32_t dstNzMatrixStride = 0,
        const bool kAlignToC0Size = false)
    {
        ASCENDC_ASSERT((row >= 0), { KERNEL_LOG(KERNEL_ERROR, "row is %d, which should be no less than 0.", row); });
        ASCENDC_ASSERT((col >= 0), { KERNEL_LOG(KERNEL_ERROR, "col is %d, which should be no less than 0.", col); });
        ASCENDC_ASSERT((height > 0),
            { KERNEL_LOG(KERNEL_ERROR, "height is %d, which should be no less than 0.", height); });
        ASCENDC_ASSERT((width > 0),
            { KERNEL_LOG(KERNEL_ERROR, "width is %d, which should be no less than 0.", width); });
        ASCENDC_ASSERT((gCol >= width), {
            KERNEL_LOG(KERNEL_ERROR,
                "ND2NZ width larger than origin matrix width, gCol is %d, which should be no less than width %d.", gCol,
                width);
        });
        int32_t dstNzC0Stride = 0;
        if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            int32_t tileHeight = GetStaticTileHeight<INPUT_TYPE::isTrans>();
            int32_t tileWidth = GetStaticTileWidth<INPUT_TYPE::isTrans>();
            if (tileHeight != height || tileWidth != width) {
                StaticPadNd2Nz<TransT>(dst, tileHeight, tileWidth, height, width);
                dstNzC0Stride = tileHeight;
            }
        }
        int64_t srcOffset;
        if constexpr (IsSameTypeV<TransT, int4b_t>) {
            srcOffset = ((int64_t)row * (int64_t)gCol * INT4_TWO + (int64_t)col);
        } else {
            srcOffset = ((int64_t)row * (int64_t)gCol  + (int64_t)col);
        }
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = ndNum;
        nd2nzParams.nValue = height;
        nd2nzParams.dValue = width;
        nd2nzParams.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzParams.srcDValue = gCol;

        if (dstNzC0Stride) {
            nd2nzParams.dstNzC0Stride = dstNzC0Stride;
        } else {
            // when k is row(height) axis, int8 type gm->l1 nd2nz should be aligned to 32(c0Size)
            // while float/half type should be aligned to 16
            if (kAlignToC0Size) {
                nd2nzParams.dstNzC0Stride = Ceil(height, c0Size_) * c0Size_;
            } else {
                nd2nzParams.dstNzC0Stride = Ceil(height, BLOCK_CUBE) * BLOCK_CUBE;
            }
        }
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = dstNzMatrixStride;
#if __CCE_AICORE__ == 220
        if constexpr (!ToMatmulConfig(MM_CFG).intrinsicsCheck) {
            DataCopy(dst, src[srcOffset], nd2nzParams);
        } else {
            if (gCol >= UINT16_MAX) {
                nd2nzParams.nValue = 1;
                nd2nzParams.srcDValue = width;
                for (int32_t i = 0; i < height; ++i) {
                    DataCopy(dst[i * c0Size_], src[srcOffset + gCol * i], nd2nzParams);
                }
            } else {
                DataCopy(dst, src[srcOffset], nd2nzParams);
            }
        }
#else
        DataCopy(dst, src[srcOffset], nd2nzParams); // stride scope has increased
#endif
    }

    __aicore__ inline void CopyND2NZ(const LocalTensor<TransT>& dst, const LocalTensor<SrcT>& src,
        const int32_t row, const int32_t col, const int32_t height, const int32_t width, const int32_t gCol)
    {
        ASSERT(gCol >= width && "Copy ND block ub->ub width larger than origin matrix width.");
        int32_t calcWidth = width / c0Size_; // cube block numbers that do not need to be pad zero
        int32_t tail = width % c0Size_;
        int32_t dstOffset = 0;
        int32_t srcOffset = row * gCol + col;
        int32_t calcWidthExr = Ceil(width, c0Size_);
        int32_t calcHeightExr = Ceil(height, BLOCK_CUBE);

        DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = BlockMode::BLOCK_MODE_VECTOR;

        int32_t srcStride = gCol * sizeof(SrcT) / ONE_BLK_SIZE - 1;
        if (gCol % c0Size_ || srcStride >= UINT16_MAX) {
            // each block len is only 32B
            for (int32_t i = 0; i < calcWidth; i++) {
                for (int32_t j = 0; j < height; j++) {
                    DataCopy(dst[dstOffset], src[srcOffset], { 1, 1, 0, 0 }, enhancedParams);
                    dstOffset += c0Size_;
                    srcOffset += gCol;
                }
                srcOffset += c0Size_;
            }
        } else {
            // data copy stride is aligned
            for (int32_t i = 0; i < calcWidth; i++) {
                DataCopy(dst[dstOffset], src[srcOffset],
                    { static_cast<uint16_t>(height), 1, static_cast<uint16_t>(srcStride), 0 }, enhancedParams);
                dstOffset += calcHeightExr * BLOCK_CUBE * c0Size_;
                srcOffset += c0Size_;
            }
        }
    }

private:
    template <bool IS_TRANS = false, typename INPUT_TYPE_ALIAS = INPUT_TYPE>
    __aicore__ constexpr enable_if_t<INPUT_TYPE_ALIAS::TAG == InputTypeTag::A, int32_t> GetStaticTileHeight() const
    {
        if constexpr ((INPUT_TYPE_ALIAS::layout != LayoutMode::NONE) &&
            (ToMatmulConfig(MM_CFG).batchMode != BatchMode::SINGLE_LARGE_THAN_L1)) {
            if constexpr (IS_TRANS) {
                return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
            } else {
                return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreM();
            }
        } else if constexpr (DoMatmulMDL(MM_CFG) || DoMatmulSpecialMDL(MM_CFG)) {
            if constexpr (IS_TRANS) {
                return MATMUL_MODULE(MatmulShapeTiling)
                    ->GetTiling().GetStepKa() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            } else {
                return MATMUL_MODULE(MatmulShapeTiling)
                    ->GetTiling().GetStepM() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
            }
        } else {
            return MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>();
        }
    }

    template <bool IS_TRANS = false, typename INPUT_TYPE_ALIAS = INPUT_TYPE>
    __aicore__ constexpr enable_if_t<INPUT_TYPE_ALIAS::TAG == InputTypeTag::A, int32_t> GetStaticTileWidth() const
    {
        if constexpr ((INPUT_TYPE_ALIAS::layout != LayoutMode::NONE) &&
            (ToMatmulConfig(MM_CFG).batchMode != BatchMode::SINGLE_LARGE_THAN_L1)) {
            if constexpr (IS_TRANS) {
                return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreM();
            } else {
                return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
            }
        } else if constexpr (DoMatmulMDL(MM_CFG) || DoMatmulSpecialMDL(MM_CFG)) {
            if constexpr (IS_TRANS) {
                return MATMUL_MODULE(MatmulShapeTiling)
                    ->GetTiling().GetStepM() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
            } else {
                return MATMUL_MODULE(MatmulShapeTiling)
                    ->GetTiling().GetStepKa() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            }
        } else {
            return MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>();
        }
    }

    template <bool IS_TRANS = false, typename INPUT_TYPE_ALIAS = INPUT_TYPE>
    __aicore__ inline enable_if_t<INPUT_TYPE_ALIAS::TAG == InputTypeTag::B, int32_t> GetStaticTileHeight() const
    {
        if constexpr ((INPUT_TYPE_ALIAS::layout != LayoutMode::NONE) &&
            (ToMatmulConfig(MM_CFG).batchMode != BatchMode::SINGLE_LARGE_THAN_L1)) {
            if constexpr (IS_TRANS) {
                return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreN();
            } else {
                return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
            }
        } else if constexpr (DoMatmulMDL(MM_CFG) || DoMatmulSpecialMDL(MM_CFG)) {
            if constexpr (IS_TRANS) {
                return MATMUL_MODULE(MatmulShapeTiling)
                    ->GetTiling().GetStepN() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
            } else {
                return MATMUL_MODULE(MatmulShapeTiling)
                    ->GetTiling().GetStepKb() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            }
        } else {
            return MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>();
        }
    }

    template <bool IS_TRANS = false, typename INPUT_TYPE_ALIAS = INPUT_TYPE>
    __aicore__ inline enable_if_t<INPUT_TYPE_ALIAS::TAG == InputTypeTag::B, int32_t> GetStaticTileWidth() const
    {
        if constexpr ((INPUT_TYPE_ALIAS::layout != LayoutMode::NONE) &&
            (ToMatmulConfig(MM_CFG).batchMode != BatchMode::SINGLE_LARGE_THAN_L1)) {
            if constexpr (IS_TRANS) {
                return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
            } else {
                return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreN();
            }
        } else if constexpr (DoMatmulMDL(MM_CFG) || DoMatmulSpecialMDL(MM_CFG)) {
            if constexpr (IS_TRANS) {
                return MATMUL_MODULE(MatmulShapeTiling)
                    ->GetTiling().GetStepKb() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            } else {
                return MATMUL_MODULE(MatmulShapeTiling)
                    ->GetTiling().GetStepN() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
            }
        } else {
            return MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>();
        }
    }

    template <typename DataType>
    __aicore__ inline void StaticPadNd2Nz(const LocalTensor<DataType>& dst, const int32_t staticHeight,
        const int32_t staticWidth, const int32_t tileHeight, const int32_t tileWidth)
    {
        if constexpr (DoMatmulNorm(MM_CFG) || DoMatmulBasicBlock(MM_CFG) || DoMatmulSpecialBasicBlock(MM_CFG)) {
            int32_t tileWidthC0 = Ceil(tileWidth, c0Size_);
            int32_t staticWidthC0 = Ceil(staticWidth, c0Size_);
            // pad left bottom area of src.
            if (tileHeight < staticHeight) {
                InitConstValueParams<DataType> initConstValueParams;
                initConstValueParams.repeatTimes = tileWidthC0;
                initConstValueParams.blockNum = staticHeight - tileHeight;
                initConstValueParams.dstGap = tileHeight;
                initConstValueParams.initValue = 0;
                InitConstValue(dst[tileHeight * c0Size_], initConstValueParams);
            }
            // pad right area of src
            if (tileWidthC0 < staticWidthC0) {
                InitConstValueParams<DataType> initConstValueParams;
                initConstValueParams.repeatTimes = 1;
                initConstValueParams.blockNum = (staticWidthC0 - tileWidthC0) * staticHeight;
                initConstValueParams.dstGap = 0;
                initConstValueParams.initValue = 0;
                InitConstValue(dst[tileWidthC0 * staticHeight * c0Size_], initConstValueParams);
            }
        } else if constexpr (DoMatmulMDL(MM_CFG) || DoMatmulSpecialMDL(MM_CFG)) {
            using params = InitConstValueParams<DataType>;
            InitConstValue(dst,
                params{ 1, static_cast<uint16_t>(staticHeight * staticWidth * sizeof(DataType) / ONE_BLK_SIZE), 0, 0 });
        }
    }

private:
    constexpr static int32_t c0Size_ = AuxGetC0Size<SrcT>();
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_ND_H
