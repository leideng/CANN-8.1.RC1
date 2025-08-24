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
 * \file copy_cube_out_datacopy.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_OUT_COPY_CUBE_OUT_DATACOPY_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_OUT_COPY_CUBE_OUT_DATACOPY_H

#include "../../utils/matmul_module.h"
#include "../../utils/matmul_param.h"
#include "copy_cube_out_intf.h"
#include "copy_cube_out_datacopy_wrapper.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    CopyCubeOut is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    CopyCubeOut is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, const auto& MM_CFG>
class CopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG, enable_if_t<(MatmulFeatureTrait<MM_CFG>::IsNeedUB())>>
{
    using SrcT = typename A_TYPE::T;
    using DstT = typename C_TYPE::T;
    using L0cT = typename GetDstType<typename A_TYPE::T>::Type;

    MATMUL_USE_MODULE(MatmulQuantProcessor);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(LocalWorkspace);
    MATMUL_USE_MODULE(CopyCubeOutUtils);

public:
    template <bool enSequentialWrite = false, typename ScheduleContext = int>
    __aicore__ inline void Copy(const GlobalTensor<DstT>& gm, const LocalTensor<L0cT>& co1Local, int32_t curRow,
                                   int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                   int32_t baseBlockWidth, const ScheduleContext& context = 0)
    {
        CopyOutImpl<enSequentialWrite>(gm, co1Local, curRow, curCol, baseHeight, baseWidth, baseBlockHeight,
                                       baseBlockWidth);
    }

    template <bool enSequentialWrite = false, typename ScheduleContext = int>
    __aicore__ inline void Copy(const LocalTensor<DstT>& co2Local, const LocalTensor<L0cT>& co1Local, int32_t curRow,
                                   int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                   int32_t baseBlockWidth, const ScheduleContext& context = 0)
    {
        CopyOutImpl<enSequentialWrite>(co2Local, co1Local, curRow, curCol, baseHeight, baseWidth, baseBlockHeight,
                                       baseBlockWidth);
    }

    template <bool enSequentialWrite = false, typename ScheduleContext = int>
    __aicore__ inline void Copy(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& co2Local,
                                   const LocalTensor<L0cT>& co1Local, int32_t curRow, int32_t curCol, int32_t baseHeight,
                                   int32_t baseWidth, int32_t baseBlockHeight, int32_t baseBlockWidth,
                                   const ScheduleContext& context = 0)
    {
        CopyOutImpl<enSequentialWrite>(gm, co2Local, co1Local, curRow, curCol, baseHeight, baseWidth, baseBlockHeight,
                                       baseBlockWidth);
    }

private:
    template <bool enSequentialWrite, class T>
    __aicore__ inline void CopyOutImpl(const T& dst, const LocalTensor<L0cT>& co1Local, int32_t curRow, int32_t curCol,
                                       int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                       int32_t baseBlockWidth)
    {
        if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
            CopyOutNZ2ND<enSequentialWrite>(dst, co1Local, curRow, curCol, baseHeight, baseWidth, baseBlockHeight,
                                            baseBlockWidth);
        } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
            CopyOutNZ2NZ<enSequentialWrite>(dst, co1Local, curRow, curCol, baseHeight, baseWidth, baseBlockHeight,
                                            baseBlockWidth);
        } else {
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Copy: unsupport Matmul format type."); });
        }
    }

    template <bool enSequentialWrite, class T>
    __aicore__ inline void CopyOutImpl(const T& dst, const LocalTensor<DstT>& co2Local,
        const LocalTensor<L0cT>& co1Local, int32_t curRow, int32_t curCol, int32_t baseHeight,
        int32_t baseWidth, int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        if constexpr(C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
            CopyOutNZ2ND<enSequentialWrite>(dst, co2Local, co1Local, curRow, curCol, baseHeight, baseWidth,
                                            baseBlockHeight, baseBlockWidth);
        } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
            CopyOutNZ2NZ<enSequentialWrite>(dst, co2Local, co1Local, curRow, curCol, baseHeight, baseWidth,
                                            baseBlockHeight, baseBlockWidth);
        } else {
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Copy: unsupport Matmul format type."); });
        }
    }

    template <bool enSequentialWrite>
    __aicore__ inline void CopyOutNZ2NZ(const LocalTensor<DstT>& co2Local, const LocalTensor<L0cT>& co1Local,
                                        int32_t curRow, int32_t curCol, int32_t baseHeight, int32_t baseWidth,
                                        int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        ASCENDC_ASSERT((MATMUL_MODULE(MatmulShapeInfo)->GetOrgM() >= MATMUL_MODULE(MatmulShapeTiling)->
            GetTiling().GetBaseM()), { KERNEL_LOG(KERNEL_ERROR, "M_ is %d , which should be not less than baseM %d",
            MATMUL_MODULE(MatmulShapeInfo)->GetOrgM(), MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM());
        });

        DataCopyParams dataCopyInfo;
        dataCopyInfo.blockCount = baseBlockWidth;
        dataCopyInfo.blockLen = baseBlockHeight;
        dataCopyInfo.srcStride = 0;
        DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        if constexpr (enSequentialWrite) {
            dataCopyInfo.dstStride = 0;
            CopyCo12Co2WithQuant(co2Local, co1Local, curCol, baseBlockHeight, baseBlockWidth, dataCopyInfo,
                enhancedParams);
        } else {
            dataCopyInfo.dstStride = (CeilAlign(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM(), BLOCK_CUBE) -
                baseBlockHeight * BLOCK_CUBE) * BLOCK_CUBE * sizeof(DstT) / ONE_BLK_SIZE;
            int32_t dstOffset = curRow * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() * BLOCK_CUBE +
                curCol * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN() *
                MATMUL_MODULE(MatmulShapeInfo)->GetOrgM();
            CopyCo12Co2WithQuant(co2Local[dstOffset], co1Local, curCol, baseBlockHeight, baseBlockWidth,
                dataCopyInfo, enhancedParams);
        }
    }

    template <bool enSequentialWrite>
    __aicore__ inline void CopyOutNZ2NZ(const GlobalTensor<DstT>& gm, const LocalTensor<L0cT>& co1Local, int32_t curRow,
                                        int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                        int32_t baseBlockWidth)
    {
        CopyOutEnQue();

        LocalTensor<DstT> localBuf = MATMUL_MODULE(LocalWorkspace)->GetTempWorkspace();
        CopyCo12Local(localBuf, co1Local, curCol, baseBlockHeight, baseBlockWidth);

        CopyOutDeQue();

        CopyLocal2GMNZ2NZ<enSequentialWrite>(gm, localBuf, curRow, curCol, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
    }

    template <bool enSequentialWrite>
    __aicore__ inline void CopyOutNZ2NZ(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& co2Local,
        const LocalTensor<L0cT>& co1Local, int32_t curRow, int32_t curCol, int32_t baseHeight, int32_t baseWidth,
        int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        CopyOutNZ2NZ<enSequentialWrite>(co2Local, co1Local, curRow, curCol, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
        CopyLocal2GMNZ2NZ<enSequentialWrite>(gm, co2Local, curRow, curCol, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
    }

    template <bool enSequentialWrite>
    __aicore__ inline void CopyOutNZ2ND(const LocalTensor<DstT>& co2Local, const LocalTensor<L0cT>& co1Local,
                                        int32_t curRow, int32_t curCol, int32_t baseHeight, int32_t baseWidth,
                                        int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        if constexpr (A_TYPE::format == CubeFormat::VECTOR) {
            ASCENDC_ASSERT((MATMUL_MODULE(MatmulShapeInfo)->GetOrgM() == 1), { KERNEL_LOG(KERNEL_ERROR,
                "M_ is %d, which should be equal with 1.", MATMUL_MODULE(MatmulShapeInfo)->GetOrgM()); });

            DataCopyParams dataCopyInfo;
            dataCopyInfo.blockCount = 1;
            dataCopyInfo.blockLen = baseBlockHeight * baseBlockWidth;
            DataCopyEnhancedParams enhancedParams;
            enhancedParams.blockMode = BlockMode::BLOCK_MODE_VECTOR;

            if constexpr (enSequentialWrite) {
                DataCopy(co2Local, co1Local, dataCopyInfo, enhancedParams);
            } else {
                int32_t dstOffset = curCol * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
                DataCopy(co2Local[dstOffset], co1Local, dataCopyInfo, enhancedParams);
            }
        } else {
            ASCENDC_ASSERT((!IsSameType<DstT, int8_t>::value && !IsSameType<DstT, uint8_t>::value),
                { KERNEL_LOG(KERNEL_ERROR, "Data format should be NZ if GetTensorC to UB when output is int8_t."); });

            LocalTensor<DstT> trans = MATMUL_MODULE(LocalWorkspace)->GetTempWorkspace();

            CopyCo12Co2WithoutQuant(trans, co1Local, curCol, baseBlockHeight, baseBlockWidth);

            if constexpr(enSequentialWrite) {
                TransNZ2NDForDstUB(co2Local, trans, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN(),
                    baseHeight, baseBlockWidth, baseBlockHeight);
            } else {
                uint32_t dimN = (MATMUL_MODULE(MatmulShapeInfo)->GetOrgKc() != 0) ?
                    MATMUL_MODULE(MatmulShapeInfo)->GetOrgKc() : MATMUL_MODULE(MatmulShapeInfo)->GetOrgN();
                int32_t dstOffset = curRow * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() * dimN +
                    curCol * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
                TransNZ2NDForDstUB(co2Local[dstOffset], trans, CeilAlign(dimN,
                    MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount()), baseHeight, baseBlockWidth, baseBlockHeight);
            }
        }
    }

    template <bool enSequentialWrite>
    __aicore__ inline void CopyOutNZ2ND(const GlobalTensor<DstT>& gm, const LocalTensor<L0cT>& co1Local, int32_t curRow,
                                        int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                        int32_t baseBlockWidth)
    {
        CopyOutEnQue();

        LocalTensor<DstT> localBuf = MATMUL_MODULE(LocalWorkspace)->GetTempWorkspace();
        CopyCo12Local(localBuf, co1Local, curCol, baseBlockHeight, baseBlockWidth);

        CopyOutDeQue();

        if constexpr (A_TYPE::format == CubeFormat::VECTOR) {
            CopyLocal2GMNZ2NZ<enSequentialWrite>(gm, localBuf, curRow, curCol, baseHeight, baseWidth, baseBlockHeight,
                                                 baseBlockWidth);
        } else if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
            if constexpr (!ToMatmulConfig(MM_CFG).enVecND2NZ || IsSameType<typename A_TYPE::T, half>::value &&
                                                                    IsSameType<typename B_TYPE::T, int8_t>::value) {
                CopyLocal2GMNZ2NDOnTheFly<enSequentialWrite>(gm, localBuf, curRow, curCol, baseHeight, baseWidth,
                                                             baseBlockHeight, baseBlockWidth);
            } else {
                CopyLocal2GMNZ2NDByVec<enSequentialWrite>(gm, localBuf, curRow, curCol, baseHeight, baseWidth,
                                                          baseBlockHeight, baseBlockWidth);
            }
        }
    }

    template <bool enSequentialWrite>
    __aicore__ inline void CopyOutNZ2ND(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& co2Local,
        const LocalTensor<L0cT>& co1Local, int32_t curRow, int32_t curCol, int32_t baseHeight, int32_t baseWidth,
        int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        CopyOutNZ2ND<enSequentialWrite>(co2Local, co1Local, curRow, curCol, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
        if constexpr (A_TYPE::format == CubeFormat::VECTOR) {
            CopyLocal2GMNZ2NZ<enSequentialWrite>(gm, co2Local, curRow, curCol, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
        } else if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
            CopyOutNZ2ND<enSequentialWrite>(gm, co1Local, curRow, curCol, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
        }
    }

    __aicore__ inline void CopyCo12Co2WithQuant(const LocalTensor<DstT>& dst, const LocalTensor<L0cT>& src,
        int32_t curCol, int32_t baseBlockHeight, int32_t baseBlockWidth, DataCopyParams& dataCopyInfo, DataCopyEnhancedParams& enhancedParams)
    {
        if constexpr (IsSameType<SrcT, int8_t>::value) {
            MATMUL_MODULE(MatmulQuantProcessor)->UpdateDataCopyParamForQuant(enhancedParams, curCol);
            uint64_t alignedHeight = baseBlockHeight * BLOCK_CUBE;
            int32_t blockOffset = BLOCK_CUBE * alignedHeight;
            if (MATMUL_MODULE(MatmulQuantProcessor)->GetMatmulQuantMode() == QuantMode_t::VREQ8 ||
                MATMUL_MODULE(MatmulQuantProcessor)->GetMatmulQuantMode() == QuantMode_t::REQ8) {
                dataCopyInfo.blockLen = baseBlockHeight;
                uint64_t addr = enhancedParams.deqTensorAddr;
                int32_t offset = ONE_BLK_SIZE * alignedHeight;
                int32_t dstOffset = 0;
                constexpr int32_t WIDTH_SIZE = ONE_BLK_SIZE * ONE_BYTE_BIT_SIZE;
                constexpr int32_t STORE_SIZE = BLOCK_CUBE * ONE_BYTE_BIT_SIZE;
                for (int32_t i = 0; i < Ceil(baseBlockWidth, TWO_TIMES); ++i) {
                    for (int32_t storeMode = 0; storeMode < TWO_TIMES; ++storeMode) {
                        if (baseBlockWidth % TWO_TIMES != 0 &&
                            i == Ceil(baseBlockWidth, TWO_TIMES) - 1 &&
                            storeMode == 1) {
                            continue;
                        }
                        if (MATMUL_MODULE(MatmulQuantProcessor)->GetMatmulQuantMode() == QuantMode_t::VREQ8) {
                            enhancedParams.deqTensorAddr = addr + i * WIDTH_SIZE + storeMode * STORE_SIZE;
                        }
                        enhancedParams.sidStoreMode = (uint8_t)storeMode;
                        DataCopy(dst[dstOffset], src[dstOffset + storeMode * blockOffset],
                            dataCopyInfo, enhancedParams);
                    }
                    dstOffset += offset;
                }
            } else if (MATMUL_MODULE(MatmulQuantProcessor)->GetMatmulQuantMode() == QuantMode_t::VDEQF16) {
                dataCopyInfo.blockCount = 1;
                dataCopyInfo.blockLen = baseBlockHeight;
                dataCopyInfo.dstStride = 0;
                uint64_t addr = enhancedParams.deqTensorAddr;
                int32_t offset = 0;
                constexpr int32_t DEQ_OFFSET = 128;
                for (int32_t i = 0; i < baseBlockWidth; ++i) {
                    enhancedParams.deqTensorAddr = addr + i * DEQ_OFFSET;
                    DataCopy(dst[offset], src[offset], dataCopyInfo, enhancedParams);
                    offset += blockOffset;
                }
            } else {
                DataCopy(dst, src, dataCopyInfo, enhancedParams);
            }
        } else {
            DataCopy(dst, src, dataCopyInfo, enhancedParams);
        }
    }

    __aicore__ inline void CopyCo12Co2WithoutQuant(const LocalTensor<DstT>& dst, const LocalTensor<L0cT>& src, int32_t curCol,
        int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        DataCopyParams dataCopyInfo;
        dataCopyInfo.blockCount = baseBlockWidth;
        dataCopyInfo.blockLen = baseBlockHeight;
        dataCopyInfo.srcStride = 0;
        dataCopyInfo.dstStride = 0;
        DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        if constexpr (IsSameType<SrcT, int8_t>::value) {
            MATMUL_MODULE(MatmulQuantProcessor)->UpdateDataCopyParamForQuant(enhancedParams, curCol);
        }
        DataCopy(dst, src, dataCopyInfo, enhancedParams);
    }

    __aicore__ inline void CopyCo12Local(const LocalTensor<DstT>& localBuf, const LocalTensor<L0cT>& co1Local, int32_t curCol, int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        DataCopyParams dataCopyInfo;
        dataCopyInfo.blockCount = 1;
        dataCopyInfo.blockLen = baseBlockHeight * baseBlockWidth;
        DataCopyEnhancedParams enhancedParams;
        if constexpr (A_TYPE::format == CubeFormat::VECTOR) {
            enhancedParams.blockMode = BlockMode::BLOCK_MODE_VECTOR;
        } else {
            enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
            ASCENDC_ASSERT((localBuf.GetSize() >= dataCopyInfo.blockLen * CUBE_MAX_SIZE), {
                KERNEL_LOG(KERNEL_ERROR, "copy len is %d, which should be less than dst size %d",
                    dataCopyInfo.blockLen * CUBE_MAX_SIZE, localBuf.GetSize());
            });
        }
        CopyCo12Co2WithQuant(localBuf, co1Local, curCol, baseBlockHeight, baseBlockWidth, dataCopyInfo, enhancedParams);
    }

    __aicore__ inline void CopyLocal2GMNZ2NZNotSeq(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& localBuf,
        int32_t curRow, int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
        int32_t baseBlockWidth)
    {
        int64_t alignM;
        int alignBaseUseM;
        if constexpr (C_TYPE::format == CubeFormat::NZ) { // nz2nz
            alignM = Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetOrgM(), BLOCK_CUBE) * BLOCK_CUBE;
            alignBaseUseM = Ceil(baseHeight, BLOCK_CUBE) * BLOCK_CUBE;
        } else { // nz2nd A is vector
            alignM = MATMUL_MODULE(MatmulShapeInfo)->GetOrgM();
            alignBaseUseM = baseHeight;
        }

        int64_t dstOffset;
        int64_t dstStride;
        int blockLen;
        int blockCount;
        if constexpr (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value) {
            dstOffset = curCol * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN() * alignM +
                curRow * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() * ONE_BLK_SIZE;
            dstStride = (alignM - alignBaseUseM) * sizeof(DstT);
            blockLen = baseBlockHeight * BLOCK_CUBE * sizeof(DstT);
            blockCount = Ceil(baseBlockWidth, TWO_TIMES);
        } else {
            dstOffset = curCol * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN() * alignM +
                curRow * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() * BLOCK_CUBE;
            dstStride = (alignM - alignBaseUseM) * sizeof(DstT) * BLOCK_CUBE / ONE_BLK_SIZE;
            blockLen = baseBlockHeight * BLOCK_CUBE * sizeof(DstT) *
                BLOCK_CUBE / ONE_BLK_SIZE;
            blockCount = baseBlockWidth;
        }

        if (dstStride >= UINT16_MAX) {
            int32_t srcOffset = 0;
            int32_t srcStride;
            if constexpr (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value) {
                dstStride = alignM * ONE_BLK_SIZE;
                srcStride = baseHeight * ONE_BLK_SIZE;
            } else {
                dstStride = alignM * BLOCK_CUBE;
                srcStride = baseHeight * BLOCK_CUBE;
            }
            for (int32_t i = 0; i < blockCount; ++i) {
                DataCopy(gm[dstOffset], localBuf[srcOffset], { 1, static_cast<uint16_t>(blockLen), 0, 0 });
                dstOffset += dstStride;
                srcOffset += srcStride;
            }
        } else {
            DataCopy(gm[dstOffset], localBuf, { static_cast<uint16_t>(blockCount), static_cast<uint16_t>(blockLen), 0,
                static_cast<uint16_t>(dstStride) });
        }
    }

    __aicore__ inline void CopyLocal2GMNZ2NZSeq(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& localBuf, int32_t baseHeight, int32_t baseBlockWidth)
    {
        int32_t blockLen = baseHeight * BLOCK_CUBE * sizeof(DstT) / ONE_BLK_SIZE;
        DataCopy(gm, localBuf, { static_cast<uint16_t>(baseBlockWidth),
            static_cast<uint16_t>(blockLen), 0, 0 });
    }

    template <bool enSequentialWrite>
    __aicore__ inline void CopyLocal2GMNZ2NZ(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& localBuf,
        int32_t curRow, int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        if constexpr (enSequentialWrite) {
            CopyLocal2GMNZ2NZSeq(gm, localBuf, baseHeight, baseBlockWidth);
        } else {
            ASCENDC_ASSERT((MATMUL_MODULE(MatmulShapeInfo)->GetOrgM() >= baseHeight), {
                KERNEL_LOG(KERNEL_ERROR, "M_ is %d, baseHeight is %d, M_ should be no less than baseHeight",
                    MATMUL_MODULE(MatmulShapeInfo)->GetOrgM(), baseHeight);
            });
            CopyLocal2GMNZ2NZNotSeq(gm, localBuf, curRow, curCol, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
        }
    }

    __aicore__ inline void TransNZ2NDForDstUB(const LocalTensor<DstT>& co2Local, const LocalTensor<DstT>& trans,
        int32_t dstStride, int32_t baseHeight, int32_t baseBlockWidth, int32_t baseBlockHeight)
    {
        DataCopyParams dataCopyInfo {
            static_cast<uint16_t>(baseBlockWidth),
            static_cast<uint16_t>(MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount() * sizeof(DstT) / ONE_BLK_SIZE),
            static_cast<uint16_t>((baseBlockHeight * BLOCK_CUBE * MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount() -
                MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount()) * sizeof(DstT) / ONE_BLK_SIZE),
            0
        };
        int32_t dstOffset = 0;
        int32_t srcOffset = 0;
        for (int32_t i = 0; i < baseHeight; i++) {
            DataCopy(co2Local[dstOffset], trans[srcOffset], dataCopyInfo);
            dstOffset += dstStride;
            srcOffset += MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount();
        }
    }

    template <bool enSequentialWrite>
    __aicore__ inline void CopyLocal2GMNZ2NDByVec(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& localBuf,
        int32_t curRow, int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        uint32_t dimN = (MATMUL_MODULE(MatmulShapeInfo)->GetOrgKc() != 0) ?
            MATMUL_MODULE(MatmulShapeInfo)->GetOrgKc() : MATMUL_MODULE(MatmulShapeInfo)->GetOrgN();

        LocalTensor<DstT> trans = MATMUL_MODULE(LocalWorkspace)->template
            GetWorkspaceWithOffset<ToMatmulConfig(MM_CFG).enableUBReuse>(
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength())
            .template ReinterpretCast<DstT>();
        int32_t transSize = localBuf.GetSize();
        if constexpr (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value) {
            if (baseBlockWidth % TWO_TIMES != 0) {
                transSize += baseBlockHeight * CUBE_MAX_SIZE;
            }
        }
        trans.SetSize(transSize);

        int32_t dstOffset;
        int32_t dstStride;
        int32_t offset;
        bool isGmAligned;
        if constexpr (enSequentialWrite) {
            dstOffset = 0;
            dstStride = 0;
            offset = baseWidth;
            isGmAligned = ((baseWidth % MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount()) == 0);
        } else {
            int32_t width = baseBlockWidth * MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount();
            if constexpr (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value) {
                width = width / TWO_TIMES;
            }
            ASCENDC_ASSERT((dimN >= width),
                { KERNEL_LOG(KERNEL_ERROR, "dimN is %d, width is %d, dimN should be no less than width", dimN, width); });
            if constexpr (C_TYPE::format == CubeFormat::ND_ALIGN) {
                isGmAligned = 1;
            } else {
                isGmAligned = ((dimN % MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount()) == 0 &&
                    (MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN() %
                    MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount()) == 0);
            }

            dstOffset = curRow * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() * dimN +
                curCol * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
            dstStride = (dimN - width) * sizeof(DstT) / ONE_BLK_SIZE;
            offset = dimN;
        }
        bool isTargetAligned = (baseWidth % MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount()) == 0;
        const bool isComputeLineByLine = (!isGmAligned || dstStride >= UINT16_MAX);

        // 1 if target is not aligned, must copy the unalign data to trans UB
        if (MATMUL_MODULE(CopyCubeOutUtils)
                ->IsNeedPadUnalignedToTrans(baseWidth, dimN, isComputeLineByLine, isTargetAligned)) {
            MATMUL_MODULE(CopyCubeOutUtils)
                ->template PadUnalignedToTrans<enSequentialWrite>(
                    trans, gm, dstOffset, isComputeLineByLine, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
        }

        // 2. trans nz buffer to nd buffer
        MATMUL_MODULE(CopyCubeOutUtils)
            ->TransNZ2NDByVec(trans, localBuf, baseBlockHeight, baseBlockWidth, baseHeight, baseWidth, baseBlockWidth);

        // 3. copy trans buffer to gm
        MATMUL_MODULE(CopyCubeOutUtils)->template CopyTrans2GM<enSequentialWrite>(gm, trans, curRow, curCol, baseHeight,
            baseWidth, baseBlockHeight, baseBlockWidth, dstOffset, offset, dstStride, isComputeLineByLine, isTargetAligned);
    }

    template <bool enSequentialWrite>
    __aicore__ inline void CopyLocal2GMNZ2NDOnTheFly(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& localBuf,
                                                     int32_t curRow, int32_t curCol, int32_t baseHeight, int32_t baseWidth,
                                                     int32_t baseBlockHeight, int32_t baseBlockWidth)
    {
        uint32_t dimN = (MATMUL_MODULE(MatmulShapeInfo)->GetOrgKc() != 0) ?
            MATMUL_MODULE(MatmulShapeInfo)->GetOrgKc() : MATMUL_MODULE(MatmulShapeInfo)->GetOrgN();
        int32_t calcWidth = baseWidth / MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount();
        int32_t dstOffset = curRow * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() * dimN +
            curCol * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        int32_t blockLen = MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount() * sizeof(DstT) / ONE_BLK_SIZE;
        int32_t srcRepeatGap = (baseBlockHeight * BLOCK_CUBE * MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount() -
            MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount()) * sizeof(DstT) / ONE_BLK_SIZE;
        int32_t tail = baseWidth % MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount();

        int32_t offset = dimN;
        if constexpr (enSequentialWrite) {
            dstOffset = 0;
            offset = baseWidth;
        }

        if constexpr (C_TYPE::format == CubeFormat::ND_ALIGN) {
            offset = CeilAlign(offset, MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount());
            calcWidth = baseBlockWidth;
            tail = 0;
        }

        // Allocate MTE2_MTE3 eventId: eventIDMte3ToMte2
        event_t eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        int32_t srcOffset = 0;
        for (int32_t i = 0; i < baseHeight; i++) {
            if (calcWidth > 0) {
                DataCopy(gm[dstOffset], localBuf[srcOffset], { static_cast<uint16_t>(calcWidth),
                    static_cast<uint16_t>(blockLen), static_cast<uint16_t>(srcRepeatGap), 0 });
                if constexpr (IsSameType<typename A_TYPE::T, half>::value &&
                    IsSameType<typename B_TYPE::T, int8_t>::value) {
                    PipeBarrier<PIPE_MTE3>();
                }
            }

            if (tail != 0) {
                MATMUL_MODULE(CopyCubeOutUtils)->CopyLocal2GMNZ2NDOnTheFlyTail(
                    gm[dstOffset], localBuf, baseHeight, baseWidth, i, calcWidth, eventIDMte3ToMte2);
            }
            dstOffset += offset;
            srcOffset += MATMUL_MODULE(CopyCubeOutUtils)->GetBlockCount();
        }
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_V>());
        SetFlag<HardEvent::MTE3_V>(eventID);
        WaitFlag<HardEvent::MTE3_V>(eventID);
        // Release MTE2_MTE3 eventId: eventIDMte3ToMte2
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
    }
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_OUT_COPY_CUBE_OUT_DATACOPY_H