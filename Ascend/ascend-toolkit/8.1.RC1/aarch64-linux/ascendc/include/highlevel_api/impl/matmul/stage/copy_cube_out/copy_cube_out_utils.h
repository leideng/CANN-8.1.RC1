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
 * \file copy_cube_out_utils.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_OUT_COPY_CUBE_OUT_UTILS_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_OUT_COPY_CUBE_OUT_UTILS_H

namespace AscendC {
namespace Impl {
namespace Detail {

#if __CCE_AICORE__ >= 220
const static uint8_t FIX_PIPE_UNIT_FLAG = 3;

template <class A_TYPE, class C_TYPE, const auto& MM_CFG, FixpipeParamsType version>
struct FixpipeParamsUtil {
    using DstT = typename C_TYPE::T;
    using SrcT = typename GetDstType<typename A_TYPE::T>::Type;
    using TYPE = FixpipeParamsV220;

public:
    __aicore__ inline ~FixpipeParamsUtil() = default;

    __aicore__ inline FixpipeParamsUtil(int32_t nSize, int32_t mSize,
        int32_t nSizeBlock, int32_t mSizeBlock, int32_t baseHeight, int32_t dstStride)
    {}

    __aicore__ inline void SetQuantMode(QuantMode_t quantMode) {}

    __aicore__ inline void SetQuantScalar(uint64_t scalar) {}

    template <typename T>
    __aicore__ inline void FixpipeOut(const T& dst, const LocalTensor<SrcT>& colLocal,
        const LocalTensor<uint64_t>& quantTensor) {}

    template <typename T>
    __aicore__ inline void FixpipeOut(const T& dst, const LocalTensor<SrcT>& colLocal) {}

public:
    TYPE params_;
};


template <class A_TYPE, class C_TYPE, const auto& MM_CFG>
struct FixpipeParamsUtil <A_TYPE, C_TYPE, MM_CFG, FixpipeParamsType::V220>
{
    using DstT = typename C_TYPE::T;
    using SrcT = typename GetDstType<typename A_TYPE::T>::Type;
    using TYPE = FixpipeParamsV220;

public:
    __aicore__ inline ~FixpipeParamsUtil() = default;

    __aicore__ inline FixpipeParamsUtil(int32_t nSize, int32_t mSize,
        int32_t nSizeBlock, int32_t mSizeBlock, int32_t baseHeight, int32_t dstStride)
    {
        if constexpr(C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
            params_.nSize = static_cast<uint16_t>(nSize);
        } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
            if constexpr (!ToMatmulConfig(MM_CFG).isEnableChannelSplit) {
                params_.nSize = static_cast<uint16_t>(nSizeBlock * BLOCK_CUBE);
                dstStride = dstStride + static_cast<uint32_t>(mSize * BLOCK_CUBE * sizeof(SrcT) / ONE_BLK_SIZE) *
                    sizeof(DstT) / sizeof(SrcT);
            } else {
                params_.nSize = static_cast<uint16_t>(nSize);
                params_.isChannelSplit = true;
            }
        }
        params_.mSize = static_cast<uint16_t>(mSize);
        params_.srcStride = CeilAlign((IsStaticPaddingEnable(MM_CFG) ? baseHeight : mSize), BLOCK_CUBE);
        params_.dstStride = dstStride;
        if constexpr(EnUnitFlag(MM_CFG)) {
            params_.unitFlag = FIX_PIPE_UNIT_FLAG;
        }
    }

    __aicore__ inline void SetQuantMode(QuantMode_t quantMode)
    {
        params_.quantPre = quantMode;
    }

    __aicore__ inline void SetQuantScalar(uint64_t scalar)
    {
        params_.deqScalar = scalar;
    }

    template <typename T>
    __aicore__ inline void FixpipeOut(const T& dst, const LocalTensor<SrcT>& colLocal, const LocalTensor<uint64_t>& quantTensor)
    {
        if constexpr (C_TYPE::format == CubeFormat::NZ) {
            Fixpipe<DstT, SrcT, CFG_NZ>(dst, colLocal, quantTensor, params_);
        } else {
            Fixpipe<DstT, SrcT, CFG_ROW_MAJOR>(dst, colLocal, quantTensor, params_);
        }
    }

    template <typename T>
    __aicore__ inline void FixpipeOut(const T& dst, const LocalTensor<SrcT>& colLocal)
    {
        if constexpr (C_TYPE::format == CubeFormat::NZ) {
            Fixpipe<DstT, SrcT, CFG_NZ>(dst, colLocal, params_);
        } else {
            Fixpipe<DstT, SrcT, CFG_ROW_MAJOR>(dst, colLocal, params_);
        }
    }

    __aicore__ inline constexpr void SetCastMode()
    {
        if constexpr (IsSameType<DstT, half>::value && IsSameType<SrcT, float>::value) {
            params_.quantPre = QuantMode_t::F322F16;
        } else if constexpr (IsSameType<DstT, bfloat16_t>::value && IsSameType<SrcT, float>::value) {
            params_.quantPre = QuantMode_t::F322BF16;
        }
    }

public:
    TYPE params_;
};
#endif

__aicore__ inline void CopyOutEnQue()
{
    event_t eventIDMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIDMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIDMte3ToV);
}

__aicore__ inline void CopyOutDeQue()
{
    event_t eventIDVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMte3);
}

__aicore__ inline void CopyLocal2GMNZ2NDEnQue()
{
    event_t eventIDSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMte3);
}

__aicore__ inline void CopyLocal2GMNZ2NDDeQue()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventID);
    WaitFlag<HardEvent::MTE3_S>(eventID);
}

__aicore__ inline void CopyTrans2GMEnQue()
{
    auto eventIDVToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_OUT_COPY_CUBE_OUT_UTILS_H