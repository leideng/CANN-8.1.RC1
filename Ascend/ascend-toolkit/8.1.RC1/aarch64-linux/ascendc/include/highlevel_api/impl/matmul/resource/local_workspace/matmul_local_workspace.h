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
* \file matmul_local_workspace.h
* \brief matmul local workspace manager
*/
#ifndef IMPL_MATMUL_RESOURCE_LOCAL_WORKSPACE_MATMUL_LOCAL_WORKSPACE_H
#define IMPL_MATMUL_RESOURCE_LOCAL_WORKSPACE_MATMUL_LOCAL_WORKSPACE_H

#include "../../utils/matmul_module.h"

namespace AscendC {
namespace Impl {
namespace Detail {
constexpr int32_t ENVEC_UBUNREUSE_COEFF = 2;
constexpr int32_t ENVEC_NZ2ND_COEFF = 3;
constexpr int32_t ENABLE_TRANS_COEFF = 2;
constexpr int32_t UNABLE_TRANS_COEFF = 4;
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
    const auto& MM_CFG, typename = void>
class MatmulLocalWorkspace {
    using SrcT = typename A_TYPE::T;
    using DstT = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    __aicore__ inline MatmulLocalWorkspace() {}
    __aicore__ inline ~MatmulLocalWorkspace() {}

    /**
     * @description: Init of MatmulLocalWorkspace
     * @param: localBuffer: Local address input through SetLocalWorkspace
     * @return: void
     */
    __aicore__ inline void Init(const LocalTensor<uint8_t>& localBuffer)
    {
#if __CCE_AICORE__ < 220
        SetWorkspace(localBuffer);
#else
        ASCENDC_ASSERT((false),
            { KERNEL_LOG(KERNEL_ERROR, "current vecrsion do not support SetLocalWorkspace interface!"); });
#endif
    }

    /**
     * @description: Get workspace with offset when enVecND2NZ is enable
     * @param: offset: the offset of localworkspace
     * @return: LocalWorkspace
     */
    __aicore__ inline LocalTensor<uint8_t> GetND2NZWorkspace(int32_t offset = 0) const
    {
        ASCENDC_ASSERT(((__ubuf__ uint8_t *)localWorkspace_.GetPhyAddr() != nullptr),
            { KERNEL_LOG(KERNEL_ERROR, "Ub workspace is nullptr, which should be given."); });
        return localWorkspace_[offset];
    }

    /**
     * @description: Get transLength or 3 * transLength workspace when enVecND2NZ is enable
     * @return: LocalWorkspace
     */
    __aicore__ inline LocalTensor<uint8_t> GetNZ2NDWorkspace() const
    {
        ASCENDC_ASSERT(((__ubuf__ uint8_t *)localWorkspace_.GetPhyAddr() != nullptr),
            { KERNEL_LOG(KERNEL_ERROR, "Ub workspace is nullptr, which should be given."); });
        if constexpr (!ToMatmulConfig(MM_CFG).enableUBReuse) {
            return localWorkspace_[MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength() * ENVEC_NZ2ND_COEFF];
        } else {
            return localWorkspace_[MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength()];
        }
    }

    /**
     * @description: Get 0 or 2 * transLength workspace when enVecND2NZ is enable
     * @return: LocalWorkspace
     */
    __aicore__ inline LocalTensor<uint8_t> GetCopy2Co2Workspace() const
    {
        ASCENDC_ASSERT(((__ubuf__ uint8_t *)localWorkspace_.GetPhyAddr() != nullptr),
            { KERNEL_LOG(KERNEL_ERROR, "Ub workspace is nullptr, which should be given."); });
        if constexpr (!ToMatmulConfig(MM_CFG).enableUBReuse) {
            return localWorkspace_[MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength() * ENVEC_UBUNREUSE_COEFF];
        } else {
            return localWorkspace_[0];
        }
    }

    /**
     * @description: Get 0 or 2 * transLength workspace with offset when enVecND2NZ is enable
     * @param: offset: the offset of localworkspace
     * @param: enableUBReuse: the flag of enable UB reuse
     * @return: LocalWorkspace
     */
    template <bool enableUBReuse = true>
    __aicore__ inline LocalTensor<uint8_t> GetWorkspaceWithOffset(int32_t offset = 0) const
    {
        ASCENDC_ASSERT(((__ubuf__ uint8_t *)localWorkspace_.GetPhyAddr() != nullptr),
            { KERNEL_LOG(KERNEL_ERROR, "Ub workspace is nullptr, which should be given."); });
        if constexpr (!enableUBReuse) {
            return localWorkspace_[MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength() * ENVEC_UBUNREUSE_COEFF + offset];
        } else {
            return localWorkspace_[offset];
        }
    }

    /**
     * @description: Get temp workspace
     * @return: LocalWorkspace
     */
    __aicore__ inline LocalTensor<DstT> GetTempWorkspace()
    {
        LocalTensor<DstT> localWorkspace = GetCopy2Co2Workspace().template ReinterpretCast<DstT>();
        localWorkspace.SetSize(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() *
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
        return localWorkspace;
    }

private:
    __aicore__ inline void SetWorkspace(const LocalTensor<uint8_t>& localBuffer)
    {
        localWorkspace_ = localBuffer;
        __ubuf__ uint8_t *addr = (__ubuf__ uint8_t *)localBuffer.GetPhyAddr();
        ASCENDC_ASSERT((addr != nullptr), { KERNEL_LOG(KERNEL_ERROR, "addr can not be nullptr"); });

        int32_t totalTransLen = 0;
        if constexpr (A_TYPE::format == CubeFormat::ND || B_TYPE::format == CubeFormat::ND || !PhyPosIsUB(C_TYPE::pos)) {
            if constexpr (ToMatmulConfig(MM_CFG).enableUBReuse) {
                totalTransLen = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength() * ENABLE_TRANS_COEFF;
            } else {
                totalTransLen = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength() * UNABLE_TRANS_COEFF;
            }
        }

        int32_t biasLen = 0;
        if (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().IsBias() && BIAS_TYPE::pos != TPosition::VECCALC) {
            biasLen = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN() * sizeof(BiasT);
        }

        ASSERT(localBuffer.GetSize() >= (totalTransLen > biasLen ? totalTransLen : biasLen));
    }

private:
    LocalTensor<uint8_t> localWorkspace_;
};

template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
    const auto& MM_CFG>
class MatmulLocalWorkspace<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG,
    enable_if_t<!ToMatmulConfig(MM_CFG).enVecND2NZ>> {
    using SrcT = typename A_TYPE::T;
    using DstT = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);

public:
    __aicore__ inline MatmulLocalWorkspace() {}
    __aicore__ inline ~MatmulLocalWorkspace() {}

    /**
     * @description: Init of MatmulLocalWorkspace
     * @param: localBuffer: Local address input through SetLocalWorkspace
     * @return: void
     */
    __aicore__ inline void Init(const LocalTensor<uint8_t>& localBuffer)
    {
#if __CCE_AICORE__ < 220
        SetWorkspace(localBuffer);
#else
        ASCENDC_ASSERT((false),
            { KERNEL_LOG(KERNEL_ERROR, "current vecrsion do not support SetLocalWorkspace interface!"); });
#endif
    }

    /**
     * @description: Get workspace with nd2nzOffset + offset when enVecND2NZ is unable
     * @param: offset: the offset of localworkspace
     * @param: enableUBReuse: the flag of enable UB reuse
     * @return: LocalWorkspace
     */
    template <bool enableUBReuse = true>
    __aicore__ inline LocalTensor<uint8_t> GetND2NZWorkspace(int32_t offset = 0) const
    {
        ASCENDC_ASSERT(((__ubuf__ uint8_t *)localWorkspace_.GetPhyAddr() != nullptr),
            { KERNEL_LOG(KERNEL_ERROR, "Ub workspace is nullptr, which should be given."); });
        return localWorkspace_[nd2nzOffset_ + offset];
    }

    /**
     * @description: Get workspace with nz2ndOffset when enVecND2NZ is unable
     * @return: LocalWorkspace
     */
    __aicore__ inline LocalTensor<uint8_t> GetNZ2NDWorkspace() const
    {
        ASCENDC_ASSERT(((__ubuf__ uint8_t *)localWorkspace_.GetPhyAddr() != nullptr),
            { KERNEL_LOG(KERNEL_ERROR, "Ub workspace is nullptr, which should be given."); });
        return localWorkspace_[nz2ndOffset_];
    }

    /**
     * @description: Get workspace with co2Offset when enVecND2NZ is unable
     * @return: LocalWorkspace
     */
    __aicore__ inline LocalTensor<uint8_t> GetCopy2Co2Workspace() const
    {
        ASCENDC_ASSERT(((__ubuf__ uint8_t *)localWorkspace_.GetPhyAddr() != nullptr),
            { KERNEL_LOG(KERNEL_ERROR, "Ub workspace is nullptr, which should be given."); });
        return localWorkspace_[co2Offset_];
    }

    /**
     * @description: Get workspace with offset when enVecND2NZ is unable
     * @param: offset: the offset of localworkspace
     * @param: enableUBReuse: the flag of enable UB reuse
     * @return: LocalWorkspace
     */
    template <bool enableUBReuse = true>
    __aicore__ inline LocalTensor<uint8_t> GetWorkspaceWithOffset(int32_t offset = 0) const
    {
        ASCENDC_ASSERT(((__ubuf__ uint8_t *)localWorkspace_.GetPhyAddr() != nullptr),
            { KERNEL_LOG(KERNEL_ERROR, "Ub workspace is nullptr, which should be given."); });
        return localWorkspace_[offset];
    }

    /**
     * @description: Get temp workspace
     * @return: LocalWorkspace
     */
    __aicore__ inline LocalTensor<DstT> GetTempWorkspace()
    {
        LocalTensor<DstT> localWorkspace = GetCopy2Co2Workspace().template ReinterpretCast<DstT>();
        localWorkspace.SetSize(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() *
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
        return localWorkspace;
    }

private:
    __aicore__ inline int32_t GetAUsedSpace()
    {
        constexpr int32_t c0Size = AuxGetC0Size<SrcT>();
        if constexpr (A_TYPE::pos != TPosition::TSCM) {
            if (!MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA() && (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK() % c0Size != 0)) {
                return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() * ONE_BLK_SIZE;
            } else if (MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA() && (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreM() % c0Size != 0)) {
                return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK() * ONE_BLK_SIZE;
            }
        }
        return 0;
    }

    __aicore__ inline int32_t GetBUsedSpace()
    {
        constexpr int32_t c0Size = AuxGetC0Size<SrcT>();
        constexpr int32_t bUsedSize = 2;
        if constexpr (B_TYPE::pos != TPosition::TSCM) {
            if (IsSameType<typename A_TYPE::T, int8_t>::value && IsSameType<typename B_TYPE::T, int8_t>::value &&
                !B_TYPE::isTrans && B_TYPE::format == CubeFormat::ND) {
                if constexpr (DoMatmulNorm(MM_CFG) || DoMatmulBasicBlock(MM_CFG)) {
                    return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN() * bUsedSize;
                } else if constexpr (DoMatmulMDL(MM_CFG) || DoMatmulSpecialMDL(MM_CFG)) {
                    return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa() *
                        MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN() * bUsedSize;
                }
            } else {
                if (!MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB() && (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreN() % c0Size != 0)) {
                    return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK() * ONE_BLK_SIZE;
                } else if (MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB() && (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK() % c0Size != 0)) {
                    return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN() * ONE_BLK_SIZE;
                }
            }
        }
        return 0;
    }

    __aicore__ inline void SetWorkspace(const LocalTensor<uint8_t>& localBuffer)
    {
        localWorkspace_ = localBuffer;
        __ubuf__ uint8_t *addr = (__ubuf__ uint8_t *)localBuffer.GetPhyAddr();
        ASCENDC_ASSERT((addr != nullptr), { KERNEL_LOG(KERNEL_ERROR, "addr can not be nullptr"); });

        int32_t len = 0;

        if (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().IsBias() && BIAS_TYPE::pos != TPosition::VECCALC) {
            len += MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN() * sizeof(BiasT);
        }

        if constexpr (C_TYPE::pos == TPosition::GM ||
                      (C_TYPE::pos == TPosition::VECCALC && C_TYPE::format != CubeFormat::NZ)) {
            co2Offset_ = len;
            len += MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN() * sizeof(DstT);
        }
        if constexpr (C_TYPE::pos == TPosition::GM) {
            constexpr int32_t blockCount = ONE_BLK_SIZE / sizeof(DstT);
            if (C_TYPE::format == CubeFormat::ND && MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreN() % blockCount != 0) {
                nz2ndOffset_ = len;
                len += ONE_BLK_SIZE;
            }
        }

        if constexpr (A_TYPE::format == CubeFormat::ND || B_TYPE::format == CubeFormat::ND) {
            nd2nzOffset_ = len;
            int32_t aTmp = GetAUsedSpace();
            int32_t bTmp = GetBUsedSpace();
            len += (aTmp >= bTmp) ? aTmp : bTmp;
        }

        ASSERT(localBuffer.GetSize() >= len);
    }

private:
    LocalTensor<uint8_t> localWorkspace_;
    int32_t nd2nzOffset_ = 0;
    int32_t nz2ndOffset_ = 0;
    int32_t co2Offset_ = 0;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _MATMUL_LOCAL_WORKSPACE_H_