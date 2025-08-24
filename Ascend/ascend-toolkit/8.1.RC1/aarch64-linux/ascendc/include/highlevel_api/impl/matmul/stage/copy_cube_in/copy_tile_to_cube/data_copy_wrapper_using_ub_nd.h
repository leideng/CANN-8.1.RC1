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
* \file data_copy_wrapper_using_ub_nd.h
* \brief
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_USING_UB_ND_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_USING_UB_ND_H

#include "data_copy_wrapper_intf.h"
#include "data_copy_wrapper_utils.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, const auto& MM_CFG, class INPUT_TYPE>
class DataCopyWrapper <IMPL, MM_CFG, INPUT_TYPE,
                      enable_if_t<MatmulFeatureTrait<MM_CFG>::IsNeedUB() && INPUT_TYPE::format == CubeFormat::ND>> {
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(LocalWorkspace);
    MATMUL_USE_MODULE(MatmulAntiQuantProcessor);

    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    __aicore__ inline DataCopyWrapper() = default;
    __aicore__ inline ~DataCopyWrapper() = default;

    template <bool IS_TRANS = false>
    __aicore__ inline void CopyND2NZWithVecOp(const LocalTensor<TransT>& dst, const GlobalTensor<SrcT>& src, const int row,
                                     const int col, const int height, const int width, const int gCol)
    {
        if constexpr (ToMatmulConfig(MM_CFG).enableL1CacheUB) {
            static_assert(DoMatmulMDL(MM_CFG), "Only MDL version support L1CacheUB.");
        }
        int calcWidth = CeilT(width, c0Size_);
        bool isBankConflict =
            calcWidth * EACH_BLOCK_BYTES_MM_API % CACHE_LINE_SIZE_MM_API == 0 && calcWidth < EACH_BLOCK_BYTES_MM_API
                ? true
                : false;
        constexpr int c0Size = IsSameTypeV<TransT, half> && IsSameTypeV<SrcT, int8_t> ? 32 : c0Size_;
        int padWidth = isBankConflict ? Ceil(width, c0Size) + 1 : Ceil(width, c0Size);
        int size = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength();
        if constexpr (IsSameTypeV<TransT, half> && IsSameTypeV<SrcT, int8_t>) {
            size = Ceil(height, c0Size) * padWidth * c0Size * c0Size / AuxGetFactor<TransT>();
        }

        auto transTensor = MATMUL_MODULE(LocalWorkspace)->GetWorkspaceWithOffset(0).template ReinterpretCast<SrcT>();
        transTensor.SetSize(size);

        auto trans = MATMUL_MODULE(LocalWorkspace)
                         ->GetWorkspaceWithOffset(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength())
                         .template ReinterpretCast<TransT>();
        trans.SetSize(size);

        int64_t srcOffset = static_cast<int64_t>(row) * static_cast<int64_t>(gCol) + static_cast<int64_t>(col);

        int calcHigh = CeilT<int32_t>(height, BLOCK_CUBE);
        auto enQueEvtID = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
        if constexpr (!ToMatmulConfig(MM_CFG).enableL1CacheUB) {
            SetFlag<HardEvent::V_MTE2>(enQueEvtID);
            WaitFlag<HardEvent::V_MTE2>(enQueEvtID);
        }
        if constexpr (ToMatmulConfig(MM_CFG).enableL1CacheUB) {
            if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
                calcWidth = GetNDBlockFromGM<InputTypeTag::A, IS_TRANS>(transTensor, src, row, col, height, width, gCol,
                                                                        isBankConflict);
            } else {
                calcWidth = GetNDBlockFromGM<InputTypeTag::B, IS_TRANS>(transTensor, src, row, col, height, width, gCol,
                                                                        isBankConflict);
            }
        } else {
            calcWidth = CopyNDBlock<SrcT, TransT>(transTensor, src, srcOffset, height, width, gCol, isBankConflict);
        }

        if constexpr (IsSameTypeV<TransT, half> && IsSameTypeV<SrcT, int8_t>) {
            static_assert(DoMatmulMDL(MM_CFG), "Only MDL version support AntiQuant.");
            if (!MATMUL_MODULE(CopyCubeInParams)->IsTranspose()) {
                enQueEvtID = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
                SetFlag<HardEvent::MTE2_S>(enQueEvtID);
                WaitFlag<HardEvent::MTE2_S>(enQueEvtID);
            }
            MATMUL_MODULE(MatmulAntiQuantProcessor)->AntiQuantCompute(trans, transTensor, isBankConflict,
                MATMUL_MODULE(CopyCubeInParams)->IsTranspose(),
                MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth());
            PipeBarrier<PIPE_V>();
            constexpr int32_t padBlock = 2;
            int32_t padWidth = isBankConflict ? calcWidth + padBlock : calcWidth;
            // update fp16 padwidth
            (const_cast<LocalTensor<TransT>&>(dst)).SetSize(size);
            SetMaskNorm();
            NDPadZeros(trans, height, padWidth, gCol, width, isBankConflict);
            LocalTensor<TransT> nzTensor;
            nzTensor = MATMUL_MODULE(LocalWorkspace)->GetWorkspaceWithOffset(0).template ReinterpretCast<TransT>();
            nzTensor.SetSize(size);
            PipeBarrier<PIPE_V>();
            NDTrans2NZ<SrcT>(nzTensor, trans, calcHigh, calcWidth, isBankConflict);
            enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(enQueEvtID);
            WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
            DataCopy(dst, nzTensor, size);
            enQueEvtID = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(enQueEvtID);
            WaitFlag<HardEvent::MTE3_MTE2>(enQueEvtID);
        } else {
            int padWidth = isBankConflict ? calcWidth + 1 : calcWidth;
            int size = calcHigh * padWidth * BLOCK_CUBE * c0Size_ / AuxGetFactor<TransT>();
            transTensor.SetSize(size);
            trans.SetSize(size);
            (const_cast<LocalTensor<TransT>&>(dst)).SetSize(size);
            NDPadZeros(transTensor, height, padWidth, gCol, width, isBankConflict);
            NDTrans2NZ<SrcT>(trans, transTensor, calcHigh, calcWidth, isBankConflict);
            enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(enQueEvtID);
            WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
            DataCopy(dst, trans, size);
            enQueEvtID = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(enQueEvtID);
            WaitFlag<HardEvent::MTE3_V>(enQueEvtID);
        }
    };

    template <typename SrcTensor, typename DstTensor>
    __aicore__ inline void CopyND2NZOnTheFly(const DstTensor& dst, const SrcTensor& src, int row, int col, int height,
                                             int width, int gCol)
    {
        if constexpr (IsGlobalTensorV<SrcTensor>) {
            ASSERT(gCol >= width && "Copy ND block gm->ub width larger than origin matrix width.");
        } else {
            ASSERT(gCol >= width && "Copy ND block ub->ub width larger than origin matrix width.");
        }

        int calcWidth = width / c0Size_; // cube block numbers that do not need to be pad zero
        int tail = width % c0Size_;
        int dstOffset = 0;
        int64_t srcOffset = (static_cast<int64_t>(row) * static_cast<int64_t>(gCol) + static_cast<int64_t>(col));
        int calcWidthExr = CeilT<int32_t>(width, c0Size_);
        int calcHeightExr = CeilT<int32_t>(height, BLOCK_CUBE);

#if __CCE_AICORE__ == 200
        // set2d, pad tail zero
        if (height % BLOCK_CUBE != 0) {
            int64_t repeat = calcWidthExr * calcHeightExr;
            InitConstValueParams<TransT> initConstValueParams;
            initConstValueParams.repeatTimes = (uint16_t)repeat;
            initConstValueParams.initValue = 0;
            InitConstValue(dst, initConstValueParams);
            if constexpr (IsGlobalTensorV<SrcTensor>) {
                PipeBarrier<PIPE_MTE2>();
            } else {
                event_t eventIDMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                SetFlag<HardEvent::MTE2_MTE3>(eventIDMte2ToMte3);
                WaitFlag<HardEvent::MTE2_MTE3>(eventIDMte2ToMte3);
            }
        }
#endif

        DataCopyEnhancedParams enhancedParams {};
        if constexpr (!IsGlobalTensorV<SrcTensor>) {
            enhancedParams.blockMode = BlockMode::BLOCK_MODE_VECTOR;
        }

        // gCol unaligned, can not use dma copy repeat stride
        if (tail) {
            // tail elements that need to be pad zero
            int blockLen = calcWidthExr * (c0Size_ * sizeof(TransT) / DEFAULT_C0_SIZE);

            // gm->l1
            int srcGap = gCol * sizeof(TransT) / ONE_BLK_SIZE - 1;
            if (gCol % c0Size_ || srcGap >= UINT16_MAX) {
                // each block len is only 32B
                for (int i = 0; i < calcWidth; i++) {
                    for (int j = 0; j < height; j++) {
                        DataCopy(dst[dstOffset + i * calcHeightExr * BLOCK_CUBE * c0Size_ + j * c0Size_],
                                 src[srcOffset + j * gCol + i * c0Size_], { 1, 1, 0, 0 }, enhancedParams);
                    }
                }
            } else {
                // data copy stride is aligned
                for (int i = 0; i < calcWidth; i++) {
                    DataCopy(dst[dstOffset], src[srcOffset],
                             { static_cast<uint16_t>(height), 1, static_cast<uint16_t>(srcGap), 0 }, enhancedParams);
                    dstOffset += calcHeightExr * BLOCK_CUBE * c0Size_;
                    srcOffset += c0Size_;
                }
            }

            // tail gm->ub pad zero, and then ub->l1

            auto trans = MATMUL_MODULE(LocalWorkspace)->GetND2NZWorkspace(0).template ReinterpretCast<TransT>();
            auto tranSize = width * EACH_BLOCK_BYTES_MM_API /
                            sizeof(typename AscendC::Conditional<IsGlobalTensorV<SrcTensor>, TransT, SrcT>::type);
            trans.SetSize(tranSize);

            int64_t tailSrcoffset = static_cast<int64_t>(row) * static_cast<int64_t>(gCol) + static_cast<int64_t>(col) +
                                    static_cast<int64_t>(calcWidth) * static_cast<int64_t>(c0Size_);

            // gm->ub
            for (int i = 0; i < height; i++) {
                DataCopy(trans[i * c0Size_], src[tailSrcoffset], { 1, 1, 0, 0 }, enhancedParams);
                tailSrcoffset += gCol;
            }

            event_t eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);

            // tail pad zero
            uint64_t mask[2];
            if constexpr (IsGlobalTensorV<SrcTensor>) {
                constexpr int32_t DIV_TWO = 2;
                if constexpr (IsSameTypeV<TransT, int8_t>) {
                    tail = CeilT<int32_t>(tail, DIV_TWO);
                }
            }
            uint16_t mask_tail_16bit = ~((1 << tail) - 1);
            uint64_t mask_tail_64bit = mask_tail_16bit;
            mask[0] = mask_tail_64bit + (mask_tail_64bit << FIRST_16BIT_OFFSET_MM_API) +
                      (mask_tail_64bit << SECOND_16BIT_OFFSET_MM_API) + (mask_tail_64bit << THIRD_16BIT_OFFSET_MM_API);
            mask[1] = mask[0];
            constexpr int32_t DUP_CEIL_NUM = 8;
            if constexpr (IsSameTypeV<TransT, int8_t>) {
                LocalTensor<int16_t> tmpTrans = trans.template ReinterpretCast<int16_t>();
                Duplicate(tmpTrans, (int16_t)0, mask, CeilT<int32_t>(height, DUP_CEIL_NUM), 1, DUP_CEIL_NUM);
            } else {
                Duplicate(trans, (TransT)0, mask, CeilT<int32_t>(height, DUP_CEIL_NUM), 1, DUP_CEIL_NUM);
            }

            event_t eventIDVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMte3);

            // ub->l1
            int heightAlignBlock = CeilT<int32_t>(height, BLOCK_CUBE);
            int tailDstOffset = heightAlignBlock * BLOCK_CUBE * c0Size_ * calcWidth;
            DataCopy(dst[tailDstOffset], trans, { static_cast<uint16_t>(height), 1, 0, 0 }, enhancedParams);
            return;
        }

        int srcGap = gCol * sizeof(TransT) / ONE_BLK_SIZE - 1;
        if (gCol % c0Size_ != 0 || srcGap >= UINT16_MAX) {
            int64_t oriSrcOffset = srcOffset;
            int oriDstOffset = dstOffset;
            // each block len is only 32B
            for (int i = 0; i < calcWidth; i++) {
                for (int j = 0; j < height; j++) {
                    DataCopy(dst[dstOffset], src[srcOffset], { 1, 1, 0, 0 }, enhancedParams);
                    dstOffset += c0Size_;
                    srcOffset += gCol;
                }
                srcOffset = oriSrcOffset + (i + 1) * c0Size_;
                dstOffset = oriDstOffset + (i + 1) * calcHeightExr * BLOCK_CUBE * c0Size_;
            }
        } else {
            // data copy stride is aligned
            for (int i = 0; i < calcWidth; i++) {
                DataCopy(dst[dstOffset], src[srcOffset],
                         { static_cast<uint16_t>(height), 1, static_cast<uint16_t>(srcGap), 0 }, enhancedParams);
                dstOffset += calcHeightExr * BLOCK_CUBE * c0Size_;
                srcOffset += c0Size_;
            }
        }
        if constexpr (IsGlobalTensorV<SrcTensor>) {
            event_t eventIDMte2ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
            SetFlag<HardEvent::MTE2_MTE1>(eventIDMte2ToMte1);
            WaitFlag<HardEvent::MTE2_MTE1>(eventIDMte2ToMte1);
        }
    }

    template <bool IS_TRANS, typename SrcTensor, typename DstTensor>
    __aicore__ inline void CopyND2NZWithTransData(const DstTensor& dst, SrcTensor& src, const int row, const int col,
                                                  const int tileHeight, const int tileWidth)
    {
        int calcWidth = CeilT(tileWidth, c0Size_) * c0Size_;
        int calcHigh = CeilT(tileHeight, c0Size_) * c0Size_;
        int64_t size = calcHigh * calcWidth;
        LocalTensor<TransT> rightMatrix =
            MATMUL_MODULE(LocalWorkspace)->GetND2NZWorkspace(0).template ReinterpretCast<TransT>();
        rightMatrix.SetSize(size);
        int srcOffset = row * MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>() *
                            MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<IS_TRANS>() +
                        col * MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>();
        int dstMatrixOffset = 0;
        for (int i = 0; i < tileHeight; i++) {
            DataCopy(rightMatrix[dstMatrixOffset], src[srcOffset], calcWidth);
            srcOffset += MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<IS_TRANS>();
            dstMatrixOffset += calcWidth;
        }
        LocalTensor<TransT> trans =
            MATMUL_MODULE(LocalWorkspace)->GetND2NZWorkspace(size).template ReinterpretCast<TransT>();
        trans.SetSize(size);

        if constexpr (IsGlobalTensorV<SrcTensor>) {
            event_t eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
        } else {
            event_t eventIDMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMte3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMte3ToV);
        }
        TransDataNDBMatrix<SrcT, TransT>(trans, rightMatrix, tileHeight, tileWidth);
        event_t eventIDVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMte3);
        CopyNZ2NZImpl(dst, trans, 0, 0, calcWidth, calcHigh, calcWidth);
    }

private:
    template <InputTypeTag TAG, bool IS_TRANS = false, enable_if_t<TAG == InputTypeTag::A, bool> = false>
    __aicore__ void CopyNDBlockToUB(const GlobalTensor<SrcT>& src, int64_t srcOffset, uint32_t cacheL1Size, int height,
                                     int width, int gCol, bool isBankConflict, int& calcWidth)
    {
        if (MATMUL_MODULE(CopyCubeInParams)->IsL1KFullLoad()) {
            for (int i = 0; i < MATMUL_MODULE(CopyCubeInParams)->GetDepthL1CacheUB(); ++i) {
                if (MATMUL_MODULE(CopyCubeInParams)->IsBufferPosEnd(i)) {
                    break;
                }
                int copyHeight = MATMUL_MODULE(CopyCubeInParams)->GetCopyHeight(i);
                auto a1CacheUb = cacheHead2UB_[i * cacheL1Size];
                calcWidth = CopyNDBlock<SrcT, TransT>(a1CacheUb, src, srcOffset, copyHeight, width, gCol, isBankConflict);
                if (MATMUL_MODULE(CopyCubeInParams)->IsTranspose()) {
                    srcOffset += MATMUL_MODULE(CopyCubeInParams)->GetStepCol() *
                                 MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight();
                } else {
                    srcOffset += MATMUL_MODULE(CopyCubeInParams)->GetStepCol() *
                                 MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight() * static_cast<int64_t>(gCol);
                }
            }
        } else {
            for (int i = 0; i < MATMUL_MODULE(CopyCubeInParams)->GetDepthL1CacheUB(); ++i) {
                if (MATMUL_MODULE(CopyCubeInParams)->IsBufferKPosEnd(i)) {
                    break;
                }
                int copyWidth =
                    MATMUL_MODULE(CopyCubeInParams)
                        ->GetCopyWidth(i, MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>());
                auto a1CacheUb = cacheHead2UB_[i * cacheL1Size];
                calcWidth = CopyNDBlock(a1CacheUb, src, srcOffset, height, copyWidth, gCol, isBankConflict);
                if (MATMUL_MODULE(CopyCubeInParams)->IsTranspose()) {
                    srcOffset += MATMUL_MODULE(CopyCubeInParams)->GetStepRow() *
                                 MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>() *
                                 static_cast<int64_t>(gCol);
                } else {
                    srcOffset += MATMUL_MODULE(CopyCubeInParams)->GetStepRow() *
                                 MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>();
                }
            }
        }
    }

    template <InputTypeTag TAG, bool IS_TRANS = false, enable_if_t<TAG == InputTypeTag::B, bool> = false>
    __aicore__ void CopyNDBlockToUB(const GlobalTensor<SrcT>& src, int64_t srcOffset, uint32_t cacheL1Size, int height,
                                     int width, int gCol, bool isBankConflict, int& calcWidth)
    {
        if (MATMUL_MODULE(CopyCubeInParams)->IsL1KFullLoad()) {
            for (int i = 0; i < MATMUL_MODULE(CopyCubeInParams)->GetDepthL1CacheUB(); ++i) {
                if (MATMUL_MODULE(CopyCubeInParams)->IsBufferPosEnd(i)) {
                    break;
                }
                int copyWidth =
                    MATMUL_MODULE(CopyCubeInParams)
                        ->GetCopyWidth(i, MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>());
                auto b1CacheUb = cacheHead2UB_[i * cacheL1Size];
                calcWidth = CopyNDBlock<SrcT, TransT>(b1CacheUb, src, srcOffset, height, copyWidth, gCol, isBankConflict);
                srcOffset += MATMUL_MODULE(CopyCubeInParams)->GetStepCol() *
                             MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>() *
                             static_cast<int64_t>(gCol);
            }
        } else {
            for (int i = 0; i < MATMUL_MODULE(CopyCubeInParams)->GetDepthL1CacheUB(); ++i) {
                if (MATMUL_MODULE(CopyCubeInParams)->IsBufferKPosEnd(i)) {
                    break;
                }
                int copyHeight =
                    MATMUL_MODULE(CopyCubeInParams)
                        ->GetCopyWidth(i, MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>());
                auto b1CacheUb = cacheHead2UB_[i * cacheL1Size];
                calcWidth = CopyNDBlock<SrcT, TransT>(b1CacheUb, src, srcOffset, copyHeight, width, gCol, isBankConflict);
                srcOffset += MATMUL_MODULE(CopyCubeInParams)->GetStepRow() *
                             MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>();
            }
        }
    }

    template <InputTypeTag TAG, bool IS_TRANS = false>
    __aicore__ inline int32_t GetNDBlockFromGM(const LocalTensor<SrcT>& transTensor, const GlobalTensor<SrcT>& src,
                                               int row, int col, int height, int width, int gCol, bool isBankConflict)
    {
        auto enQueEvtID = GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2);
        SetFlag<HardEvent::MTE1_MTE2>(enQueEvtID);
        WaitFlag<HardEvent::MTE1_MTE2>(enQueEvtID);
        int64_t srcOffset = static_cast<int64_t>(row) * static_cast<int64_t>(gCol) + static_cast<int64_t>(col);

        uint32_t cacheL1Size = MATMUL_MODULE(CopyCubeInParams)->GetStepCol() *
                               MATMUL_MODULE(CopyCubeInParams)->GetStepRow() *
                               MATMUL_MODULE(CopyCubeInParams)->GetBufferSize();
        int calcWidth = CeilT(width, c0Size_);
        if (cache2UBProc_ == 0 || cache2UBProc_ >= MATMUL_MODULE(CopyCubeInParams)->GetDepthL1CacheUB()) {
            using allocType = typename AscendC::Conditional<TAG == InputTypeTag::A, TransT, SrcT>::type;
            if (cache2UBProc_ == 0) {
                cacheHead2UB_ = qidUBCache_.template AllocTensor<allocType>();
            } else {
                qidUBCache_.FreeTensor(cacheHead2UB_);
                cacheHead2UB_ = qidUBCache_.template AllocTensor<allocType>(); // To use que to insert events
            }

            CopyNDBlockToUB<TAG>(src, srcOffset, cacheL1Size, height, width, gCol, isBankConflict, calcWidth);

            cache2UBProc_ = 0;
            auto mte2ToMte1EvtID = GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1);
            SetFlag<HardEvent::MTE2_MTE1>(mte2ToMte1EvtID);
            WaitFlag<HardEvent::MTE2_MTE1>(mte2ToMte1EvtID);
        }

        // fetch data from Cache
        uint16_t blockLen = cacheL1Size * sizeof(SrcT) / ONE_BLK_SIZE;

        auto vToMte1EvtID = GetTPipePtr()->FetchEventID(HardEvent::V_MTE1);
        SetFlag<HardEvent::V_MTE1>(vToMte1EvtID);
        WaitFlag<HardEvent::V_MTE1>(vToMte1EvtID);
        DataCopy(transTensor, cacheHead2UB_[cache2UBProc_ * cacheL1Size], { 1, static_cast<uint16_t>(blockLen), 0, 0 });
        auto mte1ToVEvtID = GetTPipePtr()->FetchEventID(HardEvent::MTE1_V);
        SetFlag<HardEvent::MTE1_V>((event_t)mte1ToVEvtID);
        WaitFlag<HardEvent::MTE1_V>((event_t)mte1ToVEvtID);
        ++cache2UBProc_;
        if (MATMUL_MODULE(CopyCubeInParams)->IsL1KFullLoad()) {
            if (MATMUL_MODULE(CopyCubeInParams)->IsBufferPosEnd()) {
                cache2UBProc_ = 0;
                qidUBCache_.FreeTensor(cacheHead2UB_);
            }
        } else {
            if (MATMUL_MODULE(CopyCubeInParams)->IsBufferKPosEnd()) {
                cache2UBProc_ = 0;
                qidUBCache_.FreeTensor(cacheHead2UB_);
            }
        }

        return calcWidth;
    }

private:
    constexpr static int32_t c0Size_ = AuxGetC0Size<TransT>();
    typename CubeInQueType<INPUT_TYPE>::QUE qidUBCache_;
    LocalTensor<SrcT> cacheHead2UB_; // Allocate and release using qidUBCache_
    int32_t cache2UBProc_ = 0;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_DATA_COPY_WRAPPER_USING_UB_ND_H
