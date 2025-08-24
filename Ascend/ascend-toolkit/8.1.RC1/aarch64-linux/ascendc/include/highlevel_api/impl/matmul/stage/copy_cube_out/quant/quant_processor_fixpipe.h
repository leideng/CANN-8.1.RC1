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
 * \file quant_processor_fixpipe.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_FIXPIPE_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_FIXPIPE_H

#include "../../../utils/matmul_module.h"
#include "../../../utils/matmul_param.h"
#include "quant_processor_intf.h"
#include "quant_processor_utils.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, class A_TYPE, class C_TYPE, const auto& MM_CFG>
class MatmulQuantProcessor<IMPL, A_TYPE, C_TYPE, MM_CFG, enable_if_t<(IsQuantSenario<typename GetDstType<typename A_TYPE::T>::Type,  typename C_TYPE::T>()
                                                                && !MatmulFeatureTrait<MM_CFG>::IsNeedUB())>>
{
    using SrcT = typename A_TYPE::T;
    using DstT = typename C_TYPE::T;
    using L0cT = typename GetDstType<typename A_TYPE::T>::Type;

public:
    __aicore__ inline MatmulQuantProcessor() {}
    __aicore__ inline ~MatmulQuantProcessor() {}

    __aicore__ inline void Init(const int32_t baseN)
    {
        baseN_ = baseN;
        isPerChannel_ = false;
        isPerTensor_ = false;
        GetTPipePtr()->InitBuffer(qidFixPipe_, 1, baseN_ * sizeof(int64_t));
    }

    __aicore__ inline QuantMode_t GetMatmulQuantMode()
    {
        return quantMode_;
    }

    __aicore__ inline void SetQuantVector(const GlobalTensor<uint64_t>& quantTensor)
    {
        if constexpr (IsSameType<L0cT, int32_t>::value && IsSameType<DstT, half>::value) {
            quantMode_ = QuantMode_t::VDEQF16;
            isPerChannel_ = true;
            quantTensor_ = quantTensor;
        } else if constexpr (IsSameType<L0cT, int32_t>::value &&
            (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value)) {
            quantMode_ = QuantMode_t::VREQ8;
            isPerChannel_ = true;
            quantTensor_ = quantTensor;
        } else if constexpr (IsSameType<L0cT, float>::value &&
            (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value)) {
            quantMode_ = QuantMode_t::VQF322B8_PRE;
            isPerChannel_ = true;
            quantTensor_ = quantTensor;
        }
    }

    __aicore__ inline void SetQuantScalar(const uint64_t quantScalar)
    {
        if constexpr (IsSameType<L0cT, int32_t>::value && IsSameType<DstT, half>::value) {
            quantMode_ = QuantMode_t::DEQF16;
            isPerTensor_ = true;
            quantScalar_ = quantScalar;
        } else if constexpr (IsSameType<L0cT, int32_t>::value &&
            (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value)) {
            quantMode_ = QuantMode_t::REQ8;
            isPerTensor_ = true;
            quantScalar_ = quantScalar;
        } else if constexpr (IsSameType<L0cT, float>::value &&
            (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value)) {
            quantMode_ = QuantMode_t::QF322B8_PRE;
            isPerTensor_ = true;
            quantScalar_ = quantScalar;
        }
    }

    __aicore__ inline void CopyQuantTensor(LocalTensor<uint64_t>& quantTensor,
        const int32_t curN, const int32_t baseUseN)
    {
        if (isPerChannel_) {
            quantTensor = qidFixPipe_.template AllocTensor<uint64_t>();
            if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
                CopyDeqTensorToL1(quantTensor, quantTensor_[curN * baseN_], baseUseN);
            } else {
                CopyDeqTensorToL1(quantTensor, quantTensor_[curN * baseN_], DivCeil(baseUseN, BLOCK_CUBE) * BLOCK_CUBE);
            }
            qidFixPipe_.EnQue(quantTensor);
            qidFixPipe_.DeQue();
        }
    }

    __aicore__ inline uint64_t GetQuantScalarValue()
    {
        return quantScalar_;
    }

    __aicore__ inline void UpdateQuantTensor(int32_t idx)
    {
        quantTensor_ = quantTensor_[idx];
    }

    __aicore__ inline bool IsPerChannelSenario()
    {
        return isPerChannel_;
    }

    __aicore__ inline void FreeQuantTensor(LocalTensor<uint64_t>& quantTensor)
    {
        if (isPerChannel_) {
            qidFixPipe_.FreeTensor(quantTensor);
        }
    }

    __aicore__ inline void Destroy()
    {
        if constexpr (((IsSameType<SrcT, int8_t>::value || IsSameType<SrcT, int4b_t>::value) &&
                       IsSameType<DstT, half>::value) ||
                      (IsSameType<SrcT, int8_t>::value &&
                       (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value))) {
            qidFixPipe_.FreeAllEvent();
        }
    }

private:
    __aicore__ inline void CopyDeqTensorToL1(const LocalTensor<uint64_t>& dst, const GlobalTensor<uint64_t>& src,
        int32_t calNSize)
    {
        event_t eventIDFixToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventIDFixToMte2);
        WaitFlag<HardEvent::FIX_MTE2>(eventIDFixToMte2);
        constexpr int DEQ_SIZE = 128;
        uint16_t deqDataSize = DivCeil(calNSize * sizeof(uint64_t), DEQ_SIZE) * DEQ_SIZE;
        // GM -> L1
        if (calNSize % BLOCK_CUBE) {
            // nd2nz pad to 32Bytes align
            uint16_t dValue = calNSize * FLOAT_FACTOR;
            Nd2NzParams intriParams{ 1, 1, dValue, 0, dValue, 1, 1, 0 };
            GlobalTensor<uint32_t> srcTmp;
            srcTmp.SetGlobalBuffer((__gm__ uint32_t *)src.GetPhyAddr(), src.GetSize());
            DataCopy(dst.ReinterpretCast<uint32_t>(), srcTmp, intriParams);
        } else {
            DataCopyParams intriParams{ 1, static_cast<uint16_t>(deqDataSize / ONE_BLK_SIZE), 0, 0 };
            DataCopy(dst, src, intriParams);
        }
        event_t eventIDMte2ToFix = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_FIX));
        SetFlag<HardEvent::MTE2_FIX>(eventIDMte2ToFix);
        WaitFlag<HardEvent::MTE2_FIX>(eventIDMte2ToFix);
    }

private:
    bool isPerTensor_ = false;
    bool isPerChannel_ = false;
    QuantMode_t quantMode_ = QuantMode_t::NoQuant;
    TQue<TPosition::C1, QUEUE_DEPTH> qidFixPipe_;
    GlobalTensor<uint64_t> quantTensor_;
    uint64_t quantScalar_ = 0;
    int32_t baseN_ = 0;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_FIXPIPE_H
