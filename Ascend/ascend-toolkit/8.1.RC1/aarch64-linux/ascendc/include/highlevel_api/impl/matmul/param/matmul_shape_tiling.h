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
 * \file matmul_shape_tiling.h
 * \brief matmul variable manager
 */

#ifndef IMPL_MATMUL_PARAM_MATMUL_SHAPE_TILING_H
#define IMPL_MATMUL_PARAM_MATMUL_SHAPE_TILING_H

#include "../utils/matmul_module.h"
#include "../tiling/matmul_constant_tiling_struct.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, const auto &MM_CFG>
class MatmulShapeTiling {
public:
    __aicore__ inline void SetTiling(const TCubeTiling* __restrict tiling)
    {
        tiling_.SetTiling(tiling);
    }

    __aicore__ inline const MatmulTiling<MM_CFG>& GetTiling() const
    {
        return tiling_;
    }

    template <typename SrcT, typename L0cT>
    __aicore__ inline void CheckTiling()
    {
#ifdef ASCENDC_CPU_DEBUG
        NumericalValidCheck();
        ShareInfoCheck();
        ShapeVaildCheck<SrcT, L0cT>();
        DepthCheck();
        ConfigCommonCheck();
        ConfigSpecificCheck();
#endif
    }

private:
#ifdef ASCENDC_CPU_DEBUG
    __aicore__ inline void NumericalValidCheck()
    {
        ASCENDC_ASSERT((tiling_.GetDepthA1() > 0), {
            KERNEL_LOG(KERNEL_ERROR, "tiling_.GetDepthA1() is %d , which should be larger than 0", tiling_.GetDepthA1());
        });
        ASCENDC_ASSERT((tiling_.GetDepthB1() > 0), {
            KERNEL_LOG(KERNEL_ERROR, "tiling_.GetDepthB1() is %d , which should be larger than 0", tiling_.GetDepthB1());
        });
        ASCENDC_ASSERT((tiling_.GetStepM() > 0), {
            KERNEL_LOG(KERNEL_ERROR, "tiling_.GetStepM() is %d , which should be larger than 0", tiling_.GetStepM());
        });
        ASCENDC_ASSERT((tiling_.GetStepN() > 0), {
            KERNEL_LOG(KERNEL_ERROR, "tiling_.GetStepN() is %d , which should be larger than 0", tiling_.GetStepN());
        });
        ASCENDC_ASSERT((tiling_.IsBias() >= 0), {
            KERNEL_LOG(KERNEL_ERROR, "tiling_.IsBias() is %d , which should be not less than 0", tiling_.IsBias());
        });

#if __CCE_AICORE__ < 220
        ASCENDC_ASSERT((tiling_.GetTransLength() > 0), {
            KERNEL_LOG(KERNEL_ERROR, "tiling_.GetTransLength() is %d , which should be larger than 0",
                tiling_.GetTransLength());
        });
        if constexpr (!ToMatmulConfig(MM_CFG).enableUBReuse) {
            ASCENDC_ASSERT(tiling_.GetTransLength() * 4 <= UBSize_, { KERNEL_LOG(KERNEL_ERROR,
                "When enableUBReuse is false, tiling_.GetTransLength() * 4 should be less than UB size");});
        }
#endif
        ASCENDC_ASSERT((tiling_.GetIterateOrder() >= 0), {
                KERNEL_LOG(KERNEL_ERROR, "tiling_.GetIterateOrder() is %d , which should be not less than 0",
                    tiling_.GetIterateOrder());
        });
    }

    __aicore__ inline void ShareInfoCheck()
    {
        ASCENDC_ASSERT((tiling_.GetShareMode() >= 0), {
            KERNEL_LOG(KERNEL_ERROR, "tiling_.GetShareMode() is %d , which should be not less than 0",
                tiling_.GetShareMode());
        });
        ASCENDC_ASSERT((tiling_.GetShareL1Size() >= 0), {
            KERNEL_LOG(KERNEL_ERROR, "tiling_.GetShareL1Size() is %d , which should be not less than 0",
                tiling_.GetShareL1Size());
        });
        ASCENDC_ASSERT((tiling_.GetShareL0CSize() >= 0), {
            KERNEL_LOG(KERNEL_ERROR, "tiling_.GetShareL0CSize() is %d , which should be not less than 0",
                tiling_.GetShareL0CSize());
        });
        ASCENDC_ASSERT((tiling_.GetShareUbSize() >= 0), {
            KERNEL_LOG(KERNEL_ERROR, "tiling_.GetShareUbSize() is %d , which should be not less than 0",
                tiling_.GetShareUbSize());
        });
    }

    template <typename SrcT, typename L0cT>
    __aicore__ inline void ShapeVaildCheck()
    {
        ASCENDC_ASSERT((tiling_.GetBaseM() * tiling_.GetBaseK() * sizeof(SrcT) <= L0ASize_), {
            KERNEL_LOG(KERNEL_ERROR, "baseM * baseK is %d , which should be not larger than L0ASize_ %d",
                tiling_.GetBaseM() * tiling_.GetBaseK() * sizeof(SrcT), L0ASize_);
        });
        ASCENDC_ASSERT((tiling_.GetBaseN() * tiling_.GetBaseK() * sizeof(SrcT) <= L0BSize_), {
            KERNEL_LOG(KERNEL_ERROR, "baseN * baseK is %d , which should be not larger than L0BSize_ %d",
                tiling_.GetBaseN() * tiling_.GetBaseK() * sizeof(SrcT), L0BSize_);
        });
        ASCENDC_ASSERT((tiling_.GetBaseM() * tiling_.GetBaseN() * sizeof(L0cT) <= L0CSize_), {
            KERNEL_LOG(KERNEL_ERROR, "baseM * baseN is %d , which should be not larger than L0CSize_ %d",
                tiling_.GetBaseM() * tiling_.GetBaseN() * sizeof(L0cT), L0CSize_);
        });
#if __CCE_AICORE__ == 220
        if constexpr ((DoMatmulNorm(MM_CFG) || DoMatmulMDL(MM_CFG)) && ToMatmulConfig(MM_CFG).isA2B2Shared) {
            ASCENDC_ASSERT((tiling_.GetBaseM() * tiling_.GetBaseK() * sizeof(SrcT) <= L0ASize_ / Impl::DB_FACTOR), {
                KERNEL_LOG(KERNEL_ERROR, "baseM * baseK is %d , which should be not larger than A2 Size / 2 when isA2B2Shared is enable %d",
                    tiling_.GetBaseM() * tiling_.GetBaseK() * sizeof(SrcT), L0ASize_ / Impl::DB_FACTOR);
            });
            ASCENDC_ASSERT((tiling_.GetBaseN() * tiling_.GetBaseK() * sizeof(SrcT) <= L0BSize_ / Impl::DB_FACTOR), {
                KERNEL_LOG(KERNEL_ERROR, "baseN * baseK is %d , which should be not larger than B2 Size / 2 when isA2B2Shared is enable %d",
                    tiling_.GetBaseN() * tiling_.GetBaseK() * sizeof(SrcT), L0BSize_ / Impl::DB_FACTOR);
            });
        }
#endif
        if (tiling_.GetShareMode() == 1) {
            ASCENDC_ASSERT((tiling_.GetBaseM() * tiling_.GetBaseK() * sizeof(SrcT) <= L0ASize_ / HALF_FACTOR), {
                KERNEL_LOG(KERNEL_ERROR,
                    "baseM is %d , baseK is %d, baseM * baseK should be less than half l0a when in mode 1",
                    tiling_.GetBaseM(), tiling_.GetBaseK());
            });
            ASCENDC_ASSERT((tiling_.GetBaseN() * tiling_.GetBaseK() * sizeof(SrcT) <= L0BSize_ / HALF_FACTOR), {
                KERNEL_LOG(KERNEL_ERROR,
                    "baseN is %d , baseK is %d, baseN * baseK should be less than half l0b when in mode 1",
                    tiling_.GetBaseN(), tiling_.GetBaseK());
            });
            ASCENDC_ASSERT((tiling_.GetBaseM() * tiling_.GetBaseN() * sizeof(L0cT) <= L0CSize_ / HALF_FACTOR), {
                KERNEL_LOG(KERNEL_ERROR,
                    "baseM is %d , baseN is %d, baseM * baseN should be less than half l0c when in mode 1",
                    tiling_.GetBaseM(), tiling_.GetBaseN());
            });
        }
    }

    __aicore__ inline void DepthCheck()
    {
#if __CCE_AICORE__ >= 220
        if constexpr (DoMatmulMDL(MM_CFG) || DoMatmulSpecialMDL(MM_CFG)) {
            ASCENDC_ASSERT((tiling_.GetDepthA1() % (tiling_.GetStepM() * tiling_.GetStepKa()) == 0), {
                KERNEL_LOG(KERNEL_ERROR, "depthA1 is %d , which should be divided exactly by stepM * stepKa(%d * %d)",
                    tiling_.GetDepthA1(), tiling_.GetStepM(), tiling_.GetStepKa());
            });
            ASCENDC_ASSERT((tiling_.GetDepthB1() % (tiling_.GetStepN() * tiling_.GetStepKb()) == 0), {
                KERNEL_LOG(KERNEL_ERROR, "depthB1 is %d , which should be divided exactly by stepN * stepKb(%d * %d)",
                    tiling_.GetDepthB1(), tiling_.GetStepN(), tiling_.GetStepKb());
            });
            ASCENDC_ASSERT((tiling_.GetDepthA1() / (tiling_.GetStepM() * tiling_.GetStepKa()) <= 2), {
                KERNEL_LOG(KERNEL_ERROR, "depthA1 is %d , stepM %d, stepKa %d, depthA1 <= 2 * (stepM * stepKa)",
                    tiling_.GetDepthA1(), tiling_.GetStepM(), tiling_.GetStepKa());
            });
            ASCENDC_ASSERT((tiling_.GetDepthB1() / (tiling_.GetStepN() * tiling_.GetStepKb()) <= 2), {
                KERNEL_LOG(KERNEL_ERROR, "depthB1 is %d , stepN %d, stepKb %d, depthB1 <= 2 * (stepN * stepKb)",
                    tiling_.GetDepthB1(), tiling_.GetStepN(), tiling_.GetStepKb());
            });
        }
#endif
    }

    template <typename IMPL_ALIAS = IMPL, const auto& MM_CFG_ALIAS = MM_CFG,
              enable_if_t<NormInitScene<MM_CFG_ALIAS>, bool> = false>
    __aicore__ inline void ConfigSpecificCheck()
    {
#if __CCE_AICORE__ < 220
        // when output is int8 and ND format, do not support on the fly trans nd2nz
        if constexpr (IMPL::CType::format == CubeFormat::ND && !ToMatmulConfig(MM_CFG).enVecND2NZ &&
                      (IsSameType<typename IMPL::CType::T, int8_t>::value ||
                       IsSameType<typename IMPL::CType::T, uint8_t>::value)) {
            ASCENDC_ASSERT(false, {
                KERNEL_LOG(KERNEL_ERROR,
                           "Norm Scene, When output's data format is ND and data type is int8_t or uint8_t,"
                           " the parameter enVecND2NZ of MM_CFG should be true");
            });
        }
#endif
    }

    template <typename IMPL_ALIAS = IMPL, const auto& MM_CFG_ALIAS = MM_CFG,
              enable_if_t<MdlInitScene<MM_CFG_ALIAS>, bool> = false>
    __aicore__ inline void ConfigSpecificCheck()
    {
#if __CCE_AICORE__ < 200
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "MatmulVersion MULTI_DATA_LOAD is valid only in v220."); });
#endif
#if __CCE_AICORE__ < 220
        // when output is int8 and ND format, do not support on the fly trans nd2nz
        if constexpr (IMPL::CType::format == CubeFormat::ND && !ToMatmulConfig(MM_CFG).enVecND2NZ &&
                      (IsSameType<typename IMPL::CType::T, int8_t>::value ||
                       IsSameType<typename IMPL::CType::T, uint8_t>::value)) {
            ASCENDC_ASSERT(false, {
                KERNEL_LOG(KERNEL_ERROR,
                           "MDL Scene, When output's data format is ND and data type is int8_t or uint8_t,"
                           " the parameter enVecND2NZ of MM_CFG should be true");
            });
        }
#endif
#if __CCE_AICORE__ != 220
        if constexpr (ToMatmulConfig(MM_CFG).scheduleType == ScheduleType::OUTER_PRODUCT) {
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ScheduleType is OUTER_PRODUCT only supported on A2"); });
        }
#endif
#if __CCE_AICORE__ == 220
        if constexpr (ToMatmulConfig(MM_CFG).scheduleType == ScheduleType::OUTER_PRODUCT) {
            ASCENDC_ASSERT(tiling_.GetSingleCoreK() <= tiling_.GetBaseK(), {
                KERNEL_LOG(KERNEL_ERROR, "When singleCoreK is larger than baseK, the parameter scheduleType of "
                                         "MM_CFG should not be OUTER_PRODUCT");
            });
            ASCENDC_ASSERT((ToMatmulConfig(MM_CFG).iterateOrder != IterateOrder::UNDEF), {
                KERNEL_LOG(KERNEL_ERROR,
                           "When scheduleType is OUTER_PRODUCT, iterateOrder of MM_CFG should not be UNDEF.");
            });
            if constexpr (ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_M) {
                ASCENDC_ASSERT((tiling_.GetStepN() > 1), {
                    KERNEL_LOG(KERNEL_ERROR, "When scheduleType is OUTER_PRODUCT and iterateOrder is ORDER_M, "
                                             "stepN should be larger than 1");
                });
            }

            if constexpr (ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_N) {
                ASCENDC_ASSERT((tiling_.GetStepM() > 1), {
                    KERNEL_LOG(KERNEL_ERROR, "When scheduleType is OUTER_PRODUCT and iterateOrder is ORDER_N, "
                                             "stepM should be larger than 1");
                });
            }
        }
#endif
    }

    template <typename IMPL_ALIAS = IMPL, const auto& MM_CFG_ALIAS = MM_CFG,
              enable_if_t<DoMatmulIBShareNorm(MM_CFG_ALIAS), bool> = false>
    __aicore__ inline void ConfigSpecificCheck()
    {
        if constexpr (IMPL::AType::ibShare) {
            ASCENDC_ASSERT((IMPL::BType::ibShare == false),
                           { KERNEL_LOG(KERNEL_ERROR, "When A is ibShare, B should not be ibShare"); });
            ASCENDC_ASSERT((!PhyPosIsL1(IMPL::AType::pos)),
                           { KERNEL_LOG(KERNEL_ERROR, "When A is ibShare, A pos should be GM"); });
        } else {
            ASCENDC_ASSERT((IMPL::BType::ibShare == true),
                           { KERNEL_LOG(KERNEL_ERROR, "When A is not ibShare, B should be ibShare"); });
            ASCENDC_ASSERT((!PhyPosIsL1(IMPL::BType::pos)),
                           { KERNEL_LOG(KERNEL_ERROR, "When B is ibShare, B pos should be GM"); });
        }
    }

    template <typename IMPL_ALIAS = IMPL, const auto& MM_CFG_ALIAS = MM_CFG,
              enable_if_t<!NormInitScene<MM_CFG_ALIAS> && !MdlInitScene<MM_CFG_ALIAS> && !DoMatmulIBShareNorm(MM_CFG),
                          bool> = false>
    __aicore__ inline void ConfigSpecificCheck()
    {
        if constexpr (IMPL::AType::layout != LayoutMode::NONE) {
            ASCENDC_ASSERT(!DoMatmulMDL(MM_CFG), { KERNEL_LOG(KERNEL_ERROR, "BatchMatmul unsupport MDL."); });
            if constexpr (ToMatmulConfig(MM_CFG).batchMode == BatchMode::SINGLE_LARGE_THAN_L1 &&
                          !ToMatmulConfig(MM_CFG).isBiasBatch) {
                ASCENDC_ASSERT(false, {
                    KERNEL_LOG(KERNEL_ERROR, "Bias reuse does not supported BatchMode::SINGLE_LARGE_THAN_L1");
                });
            }

#if __CCE_AICORE__ == 220
            if constexpr (ToMatmulConfig(MM_CFG).scheduleType == ScheduleType::OUTER_PRODUCT) {
                ASCENDC_ASSERT(tiling_.GetSingleCoreK() <= tiling_.GetBaseK(), {
                    KERNEL_LOG(KERNEL_ERROR, "When singleCoreK is larger than baseK, the parameter scheduleType of "
                                             "MM_CFG should not be OUTER_PRODUCT");
                });
            }
#endif
        }
    }

    __aicore__ inline void ConfigCommonCheck()
    {
#if __CCE_AICORE__ == 200
        if (IMPL::CType::format == CubeFormat::ND && (tiling_.GetN() * sizeof(typename IMPL::CType::T) % ONE_BLK_SIZE != 0)) {
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "N dims need to be aligined to 32B when ND format output in v200."); });
        }
#endif
#if __CCE_AICORE__ == 220
        if constexpr (ToMatmulConfig(MM_CFG).isEnableChannelSplit) {
            ASCENDC_ASSERT(
                ((IMPL::CType::format == CubeFormat::NZ) && IsSameType<typename IMPL::CType::T, float>::value), { KERNEL_LOG(KERNEL_ERROR,
                    "ChannelSplit supports only NZ format and float data type output in v220."); });
        }
#endif
        if constexpr (IMPL::AType::layout == LayoutMode::NONE && !ToMatmulConfig(MM_CFG).isBiasBatch) {
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "Bias reuse is only valid in BMM."); });
        }
    }

#endif // #ifdef ASCENDC_CPU_DEBUG

private:
    MatmulTiling<MM_CFG> tiling_;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_PARAM_MATMUL_SHAPE_TILING_H
