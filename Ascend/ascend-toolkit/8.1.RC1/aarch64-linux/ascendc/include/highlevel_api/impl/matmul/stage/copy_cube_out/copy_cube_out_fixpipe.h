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
 * \file copy_cube_out_fixpipe.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_OUT_COPY_CUBE_OUT_FIXPIPE_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_OUT_COPY_CUBE_OUT_FIXPIPE_H

#include "../../utils/matmul_module.h"
#include "../../utils/matmul_param.h"
#include "../../feature_trait/matmul_feature_trait.h"
#include "quant/quant_processor_utils.h"
#include "copy_cube_out_intf.h"
#include "copy_cube_out_utils.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    CopyCubeOut is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    CopyCubeOut is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, const auto& MM_CFG>
class CopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG, enable_if_t<(!MatmulFeatureTrait<MM_CFG>::IsNeedUB())>>
{
    using DstT = typename C_TYPE::T;
    using SrcT = typename GetDstType<typename A_TYPE::T>::Type;
    using FixpipeAdaptor = FixpipeParamsUtil<A_TYPE, C_TYPE, MM_CFG, MatmulFeatureTrait<MM_CFG>::GetFixpipeParamsType()>;

    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE(MatmulQuantProcessor);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);
    MATMUL_USE_MODULE(MatmulSubBlockInfo);

public:
    __aicore__ inline CopyCubeOut() = default;

    template <bool enSequentialWrite = false, typename ScheduleContext = int>
    __aicore__ inline void Copy(const GlobalTensor<DstT>& gm, const LocalTensor<SrcT>& co1Local, int32_t curRow,
                                   int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                   int32_t baseBlockWidth, const ScheduleContext& context = 0)
    {
#ifdef ASCENDC_CPU_DEBUG
        if (IMPL::CallBack::DataCopyOutPtr == nullptr) {
#else
        if constexpr (IMPL::CallBack::DataCopyOutPtr == nullptr) {
#endif
            if constexpr (ToMatmulConfig(MM_CFG).intraBlockPartSum)  {
                if (!MATMUL_MODULE(MatmulSubBlockInfo)->GetFakeMsg()) {
                    CopyOutImpl<enSequentialWrite, const GlobalTensor<DstT>, true>(gm, co1Local, curRow, curCol, baseHeight,
                        baseWidth, baseBlockHeight, baseBlockWidth);
                    return;
                }
            }
            CopyOutImpl<enSequentialWrite, const GlobalTensor<DstT>, false>(gm, co1Local, curRow, curCol, baseHeight,
                baseWidth, baseBlockHeight, baseBlockWidth);
        } else {
            CopyOutImplCB<enSequentialWrite>(gm, co1Local, curRow, curCol, baseHeight,
                baseWidth, baseBlockHeight, baseBlockWidth);
        }
    }

    template <bool enSequentialWrite = false, typename ScheduleContext = int>
    __aicore__ inline void Copy(const LocalTensor<DstT>& co2Local, const LocalTensor<SrcT>& co1Local, int32_t curRow,
                                   int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                   int32_t baseBlockWidth, const ScheduleContext& context = 0)
    {
        CopyOutImpl<enSequentialWrite>(co2Local, co1Local, curRow, curCol, baseHeight, baseWidth, baseBlockHeight,
                                       baseBlockWidth);
    }

private:
    template <bool enSequentialWrite, class T, bool IS_INTRA_BLOCK = false>
    __aicore__ inline void CopyOutImpl(const T& dst, const LocalTensor<SrcT>& co1Local,
        int32_t curRow, int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        if constexpr(C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
            CopyOutNZ2ND<enSequentialWrite, T, IS_INTRA_BLOCK>(dst, co1Local, curRow, curCol, baseHeight,
            baseWidth, baseBlockHeight, baseBlockWidth);
        } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
            CopyOutNZ2NZ<enSequentialWrite, T, IS_INTRA_BLOCK>(dst, co1Local, curRow, curCol, baseHeight,
            baseWidth, baseBlockHeight, baseBlockWidth);
        } else {
            ASCENDC_ASSERT(false, {KERNEL_LOG(KERNEL_ERROR, "Copy: unsupport Matmul format type.");});
        }
    }

    template <bool enSequentialWrite, class T, bool IS_INTRA_BLOCK = false>
    __aicore__ inline void CopyOutNZ2ND(const T& dst, const LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                        int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                        int32_t baseBlockWidth)
    {
        auto stride = baseWidth;
        int64_t dstOffset  = 0;
        if constexpr (!enSequentialWrite) {
            stride = GetOrgWidth<IS_INTRA_BLOCK>();
            dstOffset = static_cast<int64_t>(static_cast<int64_t>(curRow * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM()) * stride) +
                        static_cast<int64_t>(curCol * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
        }
        FixpipeAdaptor fixpipe(baseWidth,
                        baseHeight,
                        baseBlockWidth,
                        baseBlockHeight,
                        MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM(),
                        stride);

        CopyTensor(dst[dstOffset], co1Local, fixpipe, curCol, baseWidth);
    }

    template <bool enSequentialWrite, class T, bool IS_INTRA_BLOCK = false>
    __aicore__ inline void CopyOutNZ2NZ(const T& dst, const LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                        int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                        int32_t baseBlockWidth)
    {
        int64_t dstOffset = 0;
        uint32_t stride = 0;
        if constexpr (!enSequentialWrite) {
            if constexpr (!ToMatmulConfig(MM_CFG).isEnableChannelSplit) {
                dstOffset = static_cast<int64_t>(curCol * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN()) * GetOrgM<IS_INTRA_BLOCK>() +
                    static_cast<int64_t>(curRow * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM()) * BLOCK_CUBE;
                stride = static_cast<uint32_t>((GetOrgM<IS_INTRA_BLOCK>() - baseHeight) *
                                               BLOCK_CUBE * sizeof(DstT) / ONE_BLK_SIZE);
            } else {
                dstOffset = static_cast<int64_t>(curCol * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN()) * Ceil(GetOrgM<IS_INTRA_BLOCK>(), BLOCK_CUBE) * BLOCK_CUBE +
                    static_cast<int64_t>(curRow * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM()) * B32_C0SIZE;
                stride = static_cast<uint32_t>(Ceil(GetOrgM<IS_INTRA_BLOCK>() , BLOCK_CUBE) * BLOCK_CUBE *
                                               B32_C0SIZE * sizeof(DstT) / ONE_BLK_SIZE);
            }
        } else {
            if constexpr (!ToMatmulConfig(MM_CFG).isEnableChannelSplit) {
                stride = static_cast<uint32_t>((baseBlockHeight * BLOCK_CUBE - baseHeight) * BLOCK_CUBE * sizeof(DstT) / ONE_BLK_SIZE);
            } else {
                stride = static_cast<uint32_t>((baseBlockHeight * BLOCK_CUBE - baseHeight) * B32_C0SIZE * sizeof(DstT) / ONE_BLK_SIZE);
            }
        }
        if constexpr (ToMatmulConfig(MM_CFG).isEnableChannelSplit) {
            baseWidth = CeilAlign(baseWidth, B32_C0SIZE);
            if (MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA()) {
                baseHeight = CeilAlign(baseHeight, B32_C0SIZE);
            } else {
                baseHeight = CeilAlign(baseHeight, BLOCK_CUBE);
            }
        }
        FixpipeAdaptor fixpipe(baseWidth,
                               baseHeight,
                               baseBlockWidth,
                               baseBlockHeight,
                               MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM(),
                               stride);
        CopyTensor(dst[dstOffset], co1Local, fixpipe, curCol, baseWidth);
    }

    template <class T>
    __aicore__ inline void CopyTensor(const T& dst, const LocalTensor<SrcT>& co1Local,
        FixpipeAdaptor& fixpipe, const int32_t curN = 0, const int32_t baseUseN = 0)
    {
        fixpipe.SetCastMode();
        if constexpr (IsQuantSenario<SrcT, DstT>()) {
            fixpipe.SetQuantMode(MATMUL_MODULE(MatmulQuantProcessor)->GetMatmulQuantMode());
            LocalTensor<uint64_t> quantTensor;
            if (MATMUL_MODULE(MatmulQuantProcessor)->IsPerChannelSenario()) {
                MATMUL_MODULE(MatmulQuantProcessor)->CopyQuantTensor(quantTensor, curN, baseUseN);
                fixpipe.template FixpipeOut<T>(dst, co1Local, quantTensor);
                MATMUL_MODULE(MatmulQuantProcessor)->FreeQuantTensor(quantTensor);
            } else {
                fixpipe.SetQuantScalar(MATMUL_MODULE(MatmulQuantProcessor)->GetQuantScalarValue());
                fixpipe.template FixpipeOut<T>(dst, co1Local);
            }
        } else {
            fixpipe.template FixpipeOut<T>(dst, co1Local);
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgWidth()
    {
        uint32_t dimN = GetOrgN<IS_INTRA_BLOCK>();
        if (GetOrgKc<IS_INTRA_BLOCK>() != 0) {
            dimN = GetOrgKc<IS_INTRA_BLOCK>();
        }
        constexpr uint32_t blockCount = ONE_BLK_SIZE / sizeof(DstT);
        if constexpr (C_TYPE::format == CubeFormat::ND_ALIGN) {
            dimN = Ceil(dimN, blockCount) * blockCount;
        }
        return dimN;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgKc()
    {
        if constexpr ((C_TYPE::layout == LayoutMode::SBNGD) || (C_TYPE::layout == LayoutMode::BSNGD)) {
            return 0;
        } else {
            return MATMUL_MODULE(MatmulShapeInfo)->template GetOrgKc<IS_INTRA_BLOCK>();
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgM()
    {
        if constexpr (C_TYPE::layout == LayoutMode::SBNGD) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoB() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoS1();
        } else if constexpr (C_TYPE::layout == LayoutMode::BSNGD) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoS1();
        } else if constexpr (ToMatmulConfig(MM_CFG).isEnableChannelSplit && A_TYPE::format == CubeFormat::ND && C_TYPE::format == CubeFormat::NZ) {
            return Ceil(MATMUL_MODULE(MatmulShapeInfo)->template GetOrgM<IS_INTRA_BLOCK>(), BLOCK_CUBE) * BLOCK_CUBE;
        } else {
            return MATMUL_MODULE(MatmulShapeInfo)->template GetOrgM<IS_INTRA_BLOCK>();
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgN()
    {
        if constexpr (C_TYPE::layout == LayoutMode::SBNGD) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoG() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoS2() *
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoN() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoB();
        } else if constexpr (C_TYPE::layout == LayoutMode::BSNGD) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoG() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoS2() *
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoN();
        } else {
            return MATMUL_MODULE(MatmulShapeInfo)->template GetOrgN<IS_INTRA_BLOCK>();
        }
    }

    template <bool enSequentialWrite>
    __aicore__ inline void CopyOutImplCB(const GlobalTensor<DstT>& dst, const LocalTensor<SrcT>& co1Local,
        int32_t curRow, int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        // Get stride of gm addr and basewith for DataCopyOutParams params
        uint32_t nSize;
        uint32_t dstStrideIn;
        int64_t dstOffset;
        GetCBCopyOutParams<enSequentialWrite>(curRow, curCol, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth,
            nSize, dstStrideIn, dstOffset);

        // DataCopyOut params for callback function
        DataCopyOutParams param(baseBlockWidth, static_cast<uint16_t>(baseHeight * BLOCK_CUBE * sizeof(SrcT) / ONE_BLK_SIZE), 0,
            dstStrideIn, nSize, EnUnitFlag(MM_CFG), curRow, curCol);

        // Update quant params
        LocalTensor<uint64_t> l1TmpForQuant;
        MATMUL_MODULE(MatmulQuantProcessor)->CopyQuantTensor(l1TmpForQuant, curCol, baseWidth);
        param.quantMode = MATMUL_MODULE(MatmulQuantProcessor)->GetMatmulQuantMode();
        param.quantScalar = MATMUL_MODULE(MatmulQuantProcessor)->GetQuantScalarValue();
        param.cbufWorkspaceAddr = reinterpret_cast<uint64_t>(l1TmpForQuant.GetPhyAddr());

        // CallBack with user define
        LocalTensor<int8_t> co1LocalInt8 = co1Local.template ReinterpretCast<int8_t>();
        (IMPL::CallBack::DataCopyOutPtr)(reinterpret_cast<__gm__ void *>(dst[dstOffset].address_),
            co1LocalInt8,
            reinterpret_cast<void *>(&param),
            MATMUL_MODULE(MatmulUserDefineInfo)->GetUserDefineInfo(),
            MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData());

        MATMUL_MODULE(MatmulQuantProcessor)->FreeQuantTensor(l1TmpForQuant);
    }

    template <bool enSequentialWrite>
    __aicore__ inline void GetCBCopyOutParams(int32_t curRow, int32_t curCol, int32_t baseHeight, int32_t baseWidth,
        int32_t baseBlockHeight, int32_t baseBlockWidth, uint32_t &nSize, uint32_t &dstStrideIn, int64_t &dstOffset)
    {
        if constexpr (enSequentialWrite) {
            dstOffset = 0;
            if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
                dstStrideIn = baseWidth;
                nSize = static_cast<uint16_t>(baseWidth);
            } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
                dstStrideIn = static_cast<uint32_t>((baseBlockHeight * BLOCK_CUBE -
                    baseHeight) * BLOCK_CUBE * sizeof(DstT) / ONE_BLK_SIZE);
                nSize = 0;
            }
        } else {
            auto baseM = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
            auto baseN = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
            if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
                dstStrideIn = GetOrgWidth();
                nSize = static_cast<uint16_t>(baseWidth);
                dstOffset = static_cast<int64_t>(static_cast<int64_t>(curRow * baseM) * dstStrideIn) + static_cast<int64_t>(curCol * baseN);
            } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
                dstStrideIn = static_cast<uint32_t>((MATMUL_MODULE(MatmulShapeInfo)->GetOrgM() - baseHeight) *
                    BLOCK_CUBE * sizeof(DstT) / ONE_BLK_SIZE);
                nSize = 0;
                dstOffset = curCol * baseN * MATMUL_MODULE(MatmulShapeInfo)->GetOrgM() + curRow * baseM * BLOCK_CUBE;
            }
        }
    }
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_OUT_COPY_CUBE_OUT_FIXPIPE_H
