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
* \file copy_tile_to_cube_using_ub.h
* \brief
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_COPY_TILE_TO_CUBE_USING_UB_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_COPY_TILE_TO_CUBE_USING_UB_H

#include "copy_tile_to_cube_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, const auto& MM_CFG, class INPUT_TYPE>
class CopyTileToCubeWrapper<IMPL, MM_CFG, INPUT_TYPE, enable_if_t<MatmulFeatureTrait<MM_CFG>::IsNeedUB()>> {
    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(DataCopyWrapper, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);

public:
    __aicore__ inline CopyTileToCubeWrapper() = default;
    __aicore__ inline ~CopyTileToCubeWrapper() = default;

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void CopyTileToCube(const LocalTensor<TransT>& dst, int32_t curRow, int32_t curCol,
        int32_t tileHeight, int32_t tileWidth)
    {
        if (MATMUL_MODULE(CopyCubeInParams)->IsTranspose()) {
            if constexpr (PhyPosIsUB(INPUT_TYPE::pos)) {
                CopyTileToCubeFromUB<true>(dst, curRow, curCol, tileHeight, tileWidth);
            } else {
                CopyTileToCubeFromGM<true>(dst, curRow, curCol, tileHeight, tileWidth);
            }
        } else {
            if constexpr (PhyPosIsUB(INPUT_TYPE::pos)) {
                CopyTileToCubeFromUB<false>(dst, curRow, curCol, tileHeight, tileWidth);
            } else {
                CopyTileToCubeFromGM<false>(dst, curRow, curCol, tileHeight, tileWidth);
            }
        }
    }

private:
    constexpr static int32_t c0Size_ = AuxGetC0Size<SrcT>();

    template <bool IS_TRANS = false, typename INPUT_TYPE_ALIAS = INPUT_TYPE>
    __aicore__ inline enable_if_t<INPUT_TYPE_ALIAS::format == CubeFormat::ND, bool>
    CopyTileToCubeFromGM(const LocalTensor<TransT>& aMatrix, int curRow, int curCol, int tileHeight, int tileWidth)
    {
        GlobalTensor<SrcT> aGlobal;
        aGlobal.SetGlobalBuffer(MATMUL_MODULE(MatmulTensorInfo)->GetGlobalTensor().address_);
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B && IsSameTypeV<TransT, int8_t> &&
                      IsSameTypeV<SrcT, int8_t> && !IS_TRANS) {
            MATMUL_MODULE(DataCopyWrapper)
                ->template CopyND2NZWithTransData<IS_TRANS>(aMatrix, aGlobal, curRow, curCol, tileHeight,
                                                            tileWidth);
        } else {
            if constexpr (ToMatmulConfig(MM_CFG).enVecND2NZ) {
                MATMUL_MODULE(DataCopyWrapper)
                    ->template CopyND2NZWithVecOp<IS_TRANS>(aMatrix, aGlobal,
                        curRow * MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>(),
                        curCol * MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>(),
                        tileHeight, tileWidth,
                        MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<IS_TRANS>());
            } else {
                MATMUL_MODULE(DataCopyWrapper)
                    ->CopyND2NZOnTheFly(
                        aMatrix, aGlobal, curRow * MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>(),
                        curCol * MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>(), tileHeight, tileWidth,
                        MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<IS_TRANS>());
            }
        }

        return true;
    }

    template <bool IS_TRANS = false, typename INPUT_TYPE_ALIAS = INPUT_TYPE>
    __aicore__ inline enable_if_t<INPUT_TYPE_ALIAS::format == CubeFormat::NZ, bool>
    CopyTileToCubeFromGM(const LocalTensor<TransT>& aMatrix, int curRow, int curCol, int tileHeight, int tileWidth)
    {
        GlobalTensor<TransT> aGlobal;
        aGlobal.SetGlobalBuffer(MATMUL_MODULE(MatmulTensorInfo)->GetGlobalTensor().address_);
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B && IsSameTypeV<TransT, int8_t> &&
                      IsSameTypeV<SrcT, int8_t> && !IS_TRANS) {
            MATMUL_MODULE(DataCopyWrapper)
                ->template CopyNZ2NZWithTransData<IS_TRANS>(aMatrix, aGlobal, curRow, curCol, tileHeight,
                                                            tileWidth);
        } else {
            MATMUL_MODULE(DataCopyWrapper)
                ->CopyNZ2NZ(aMatrix, aGlobal,
                            curRow * MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>(),
                            curCol * MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>(), tileHeight,
                            tileWidth, MATMUL_MODULE(CopyCubeInParams)->template GetOrgHeight<IS_TRANS>());
        }

        return true;
    }

    template <bool IS_TRANS = false, typename INPUT_TYPE_ALIAS = INPUT_TYPE>
    __aicore__ inline enable_if_t<INPUT_TYPE_ALIAS::format == CubeFormat::VECTOR, bool>
    CopyTileToCubeFromGM(const LocalTensor<TransT>& aMatrix, int curRow, int curCol, int tileHeight, int tileWidth)
    {
        if (MATMUL_MODULE(CopyCubeInParams)->IsKRowDirec()) {
            return false;
        }
        GlobalTensor<TransT> aGlobal;
        aGlobal.SetGlobalBuffer(MATMUL_MODULE(MatmulTensorInfo)->GetGlobalTensor().address_);
        MATMUL_MODULE(DataCopyWrapper)
            ->CopyVector2A1(aMatrix, aGlobal,
                            curCol * MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>(),
                            CeilT<int32_t>(tileWidth, c0Size_));

        return true;
    }

    template <bool IS_TRANS = false, typename INPUT_TYPE_ALIAS = INPUT_TYPE>
    __aicore__ inline enable_if_t<INPUT_TYPE_ALIAS::format != CubeFormat::ND &&
                                  INPUT_TYPE_ALIAS::format != CubeFormat::NZ &&
                                  INPUT_TYPE_ALIAS::format != CubeFormat::VECTOR, bool>
    CopyTileToCubeFromGM(const LocalTensor<TransT>& aMatrix, int curRow, int curCol, int tileHeight, int tileWidth)
    {
        return false;
    }

    template <bool IS_TRANS = false, typename INPUT_TYPE_ALIAS = INPUT_TYPE>
    __aicore__ inline enable_if_t<INPUT_TYPE_ALIAS::format == CubeFormat::ND, bool>
    CopyTileToCubeFromUB(const LocalTensor<TransT>& aMatrix, int curRow, int curCol, int tileHeight, int tileWidth)
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B && IsSameTypeV<TransT, int8_t> &&
                      IsSameTypeV<SrcT, int8_t> && !IS_TRANS) {
            LocalTensor<SrcT> leftMatrix;
            leftMatrix.SetAddr(MATMUL_MODULE(MatmulTensorInfo)->GetLocalTensor().address_);
            MATMUL_MODULE(DataCopyWrapper)
                ->template CopyND2NZWithTransData<IS_TRANS>(aMatrix, leftMatrix, curRow, curCol, tileHeight,
                                                            tileWidth);
        } else {
            if constexpr (ToMatmulConfig(MM_CFG).enVecND2NZ) {
                return false;
            } else {
                LocalTensor<SrcT> leftMatrix;
                leftMatrix.SetAddr(MATMUL_MODULE(MatmulTensorInfo)->GetLocalTensor().address_);
                MATMUL_MODULE(DataCopyWrapper)
                    ->CopyND2NZOnTheFly(aMatrix, leftMatrix,
                                        curRow * MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>(),
                                        curCol * MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>(), tileHeight,
                                        tileWidth, MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<IS_TRANS>());
            }
        }

        return true;
    }

    template <bool IS_TRANS = false, typename INPUT_TYPE_ALIAS = INPUT_TYPE>
    __aicore__ inline enable_if_t<INPUT_TYPE_ALIAS::format == CubeFormat::NZ, bool>
    CopyTileToCubeFromUB(const LocalTensor<TransT>& aMatrix, int curRow, int curCol, int tileHeight, int tileWidth)
    {
        LocalTensor<SrcT> leftMatrix;
        leftMatrix.SetAddr(MATMUL_MODULE(MatmulTensorInfo)->GetLocalTensor().address_);
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B && IsSameTypeV<TransT, int8_t> &&
                      IsSameTypeV<SrcT, int8_t> && !IS_TRANS) {
            MATMUL_MODULE(DataCopyWrapper)
                ->template CopyNZ2NZWithTransData<IS_TRANS>(aMatrix, leftMatrix, curRow, curCol, tileHeight,
                                                            tileWidth);
        } else {
            MATMUL_MODULE(DataCopyWrapper)
                ->CopyNZ2NZ(aMatrix, leftMatrix,
                            curRow * MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>(),
                            curCol * MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>(), tileHeight,
                            tileWidth, MATMUL_MODULE(CopyCubeInParams)->template GetOrgHeight<IS_TRANS>());
        }
        return true;
    }

    template <bool IS_TRANS = false, typename INPUT_TYPE_ALIAS = INPUT_TYPE>
    __aicore__ inline enable_if_t<INPUT_TYPE_ALIAS::format == CubeFormat::VECTOR, bool>
    CopyTileToCubeFromUB(const LocalTensor<TransT>& aMatrix, int curRow, int curCol, int tileHeight, int tileWidth)
    {
        if (MATMUL_MODULE(CopyCubeInParams)->IsKRowDirec()) {
            return false;
        }
        LocalTensor<SrcT> leftMatrix;
        leftMatrix.SetAddr(MATMUL_MODULE(MatmulTensorInfo)->GetLocalTensor().address_);
        MATMUL_MODULE(DataCopyWrapper)
            ->CopyVector2A1(aMatrix, leftMatrix,
                            curCol * MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>(),
                            CeilT<int32_t>(tileWidth, c0Size_));
        return true;
    }

    template <bool IS_TRANS = false, typename INPUT_TYPE_ALIAS = INPUT_TYPE>
    __aicore__ inline enable_if_t<INPUT_TYPE_ALIAS::format != CubeFormat::ND &&
                                INPUT_TYPE_ALIAS::format != CubeFormat::NZ &&
                                INPUT_TYPE_ALIAS::format != CubeFormat::VECTOR,
                                bool>
    CopyTileToCubeFromUB(const LocalTensor<TransT>& aMatrix, int curRow, int curCol, int tileHeight, int tileWidth)
    {
        return false;
    }
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_COPY_TILE_TO_CUBE_USING_UB_H
