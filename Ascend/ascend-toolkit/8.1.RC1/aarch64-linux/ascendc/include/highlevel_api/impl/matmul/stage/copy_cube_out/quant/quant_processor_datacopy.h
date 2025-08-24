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
 * \file quant_processor_datacopy.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_DATACOPY_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_DATACOPY_H

#include "../../../utils/matmul_module.h"
#include "../../../utils/matmul_param.h"
#include "quant_processor_intf.h"
#include "quant_processor_utils.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, class A_TYPE, class C_TYPE, const auto& MM_CFG>
class MatmulQuantProcessor<IMPL, A_TYPE, C_TYPE, MM_CFG, enable_if_t<(IsQuantSenario<typename GetDstType<typename A_TYPE::T>::Type,  typename C_TYPE::T>() &&
    MatmulFeatureTrait<MM_CFG>::IsNeedUB())>>
{
    using SrcT = typename A_TYPE::T;
    using DstT = typename C_TYPE::T;

    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(LocalWorkspace);

public:
    __aicore__ inline MatmulQuantProcessor() {}
    __aicore__ inline ~MatmulQuantProcessor() {}

    __aicore__ inline void Init(const int32_t baseN) {}

    __aicore__ inline QuantMode_t GetMatmulQuantMode()
    {
        return quantMode_;
    }

    __aicore__ inline void SetQuantVector(const GlobalTensor<uint64_t>& quantTensor)
    {
        if constexpr (IsSameType<SrcT, int8_t>::value && IsSameType<DstT, half>::value) {
            quantTensor_ = quantTensor;
            quantMode_ = QuantMode_t::VDEQF16;
        } else if constexpr (IsSameType<SrcT, half>::value && IsSameType<DstT, int8_t>::value) {
            quantTensor_ = quantTensor;
            quantMode_ = QuantMode_t::VQF322B8_PRE;
        } else if constexpr (IsSameType<SrcT, int8_t>::value &&
            (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value)) {
            quantTensor_ = quantTensor;
            quantMode_ = QuantMode_t::VREQ8;
        } else {
            ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Unsupported quantMode"); });
        }
    }

    __aicore__ inline void SetQuantScalar(const uint64_t quantScalar)
    {
        if constexpr (IsSameType<SrcT, int8_t>::value && IsSameType<DstT, half>::value) {
            quantScalar_ = quantScalar;
            quantMode_ = QuantMode_t::DEQF16;
        } else if constexpr (IsSameType<SrcT, half>::value && IsSameType<DstT, int8_t>::value) {
            quantScalar_ = quantScalar;
            quantMode_ = QuantMode_t::QF322B8_PRE;
        } else if constexpr (IsSameType<SrcT, int8_t>::value &&
            (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value)) {
            quantScalar_ = quantScalar;
            quantMode_ = QuantMode_t::REQ8;
        } else {
            ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Unsupported quantMode."); });
        }
    }

    __aicore__ inline void CopyQuantTensor(LocalTensor<uint64_t>& quantTensor,
        const int32_t curN, const int32_t baseUseN)
    {
        CopyQuantTensorImpl(quantTensor, curN, baseUseN);
    }

    __aicore__ inline uint64_t GetQuantScalarValue()
    {
        return quantScalar_;
    }

    __aicore__ inline void UpdateQuantTensor(int32_t idx)
    {
        quantTensor_ = quantTensor_[idx];
    }

    __aicore__ inline bool IsPerChannelSenario() {}

    __aicore__ inline bool IsPerTensorSenario() {}

    __aicore__ inline void FreeQuantTensor(LocalTensor<uint64_t>& tmpQuantTensor) {}

    __aicore__ inline void Destroy() {}

    __aicore__ inline void UpdateDataCopyParamForQuant(DataCopyEnhancedParams& enhancedParams, int curCol)
    {
        if constexpr (IsSameType<DstT, half>::value) {
            if (quantMode_ == QuantMode_t::DEQF16) {
                enhancedParams.deqScale = DeqScale::DEQ16;
                enhancedParams.deqValue = quantScalar_;
            } else if (quantMode_ == QuantMode_t::VDEQF16) {
                enhancedParams.deqScale = DeqScale::VDEQ16;
                LocalTensor<uint64_t> quantLocalTensor;
                CopyQuantTensor(quantLocalTensor, curCol, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
                enhancedParams.deqTensorAddr = reinterpret_cast<uint64_t>(quantLocalTensor.GetPhyAddr());
            }
        } else if constexpr (IsSameType<DstT, int8_t>::value || IsSameType<DstT, uint8_t>::value) {
            constexpr uint8_t SID_MODE = 2;
            enhancedParams.sidStoreMode = SID_MODE;
            if (quantMode_ == QuantMode_t::QF322B8_PRE ||
                quantMode_ == QuantMode_t::REQ8) {
                enhancedParams.deqScale = DeqScale::DEQ8;
                enhancedParams.deqValue = quantScalar_;
            } else if (quantMode_ == QuantMode_t::VQF322B8_PRE ||
                quantMode_ == QuantMode_t::VREQ8) {
                enhancedParams.deqScale = DeqScale::VDEQ8;
                LocalTensor<uint64_t> quantLocalTensor;
                CopyQuantTensor(quantLocalTensor, curCol, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
                enhancedParams.deqTensorAddr = reinterpret_cast<uint64_t>(quantLocalTensor.GetPhyAddr());
            }
        }
    }

private:
    __aicore__ inline void CopyQuantTensorImpl(LocalTensor<uint64_t>& quantTensor,
        const int32_t curN, const int32_t baseUseN)
    {
        quantTensor = MATMUL_MODULE(LocalWorkspace)->template
            GetWorkspaceWithOffset<ToMatmulConfig(MM_CFG).enableUBReuse>(
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength())
            .template ReinterpretCast<uint64_t>();
        quantTensor.SetSize(baseUseN);
        auto enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(enQueEvtID);
        WaitFlag<HardEvent::MTE3_MTE2>(enQueEvtID);
        DataCopy(quantTensor, quantTensor_[curN * baseUseN], baseUseN);
        auto eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
    }

private:
    QuantMode_t quantMode_ = QuantMode_t::NoQuant;
    GlobalTensor<uint64_t> quantTensor_;
    uint64_t quantScalar_ = 0;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_DATACOPY_H
