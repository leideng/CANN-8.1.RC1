/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file kernel_operator_symbol_override_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_SYM_OVERRIDE_H
#define ASCENDC_MODULE_OPERATOR_SYM_OVERRIDE_H
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include "kernel_check.h"
#endif

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_cmpsel_impl.h"
#include "dav_c100/kernel_operator_vec_binary_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_cmpsel_impl.h"
#include "dav_m200/kernel_operator_vec_binary_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_cmp_impl.h"
#include "dav_c220/kernel_operator_vec_binary_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_binary_continuous_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_binary_continuous_impl.h"
#endif
#pragma begin_pipe(V)
namespace AscendC {
template <typename T> class LocalTensor;

// Addition symbol overload
template <typename T> class SymbolOverrideAdd {
public:
    __aicore__ inline SymbolOverrideAdd(const LocalTensor<T> &src0Tensor, const LocalTensor<T> &src1Tensor)
        : src0Tensor_(src0Tensor), src1Tensor_(src1Tensor)
    {}

    __aicore__ inline void Process(const LocalTensor<T> &dstTensor) const
    {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT((CheckFuncVecBinary(dstTensor, this->src0Tensor_, this->src1Tensor_, dstTensor.GetSize(),
        "Add operator")), { ASCENDC_REPORT_CHECK_ERROR("Add operator", KernelFuncType::NONE_MODE);});
#endif
        AddImpl((__ubuf__ PrimT<T>*)dstTensor.GetPhyAddr(), (__ubuf__ PrimT<T>*)this->src0Tensor_.GetPhyAddr(),
            (__ubuf__ PrimT<T>*)this->src1Tensor_.GetPhyAddr(), dstTensor.GetSize());
    }

private:
    const LocalTensor<T> &src0Tensor_;
    const LocalTensor<T> &src1Tensor_;
};
// Subtract Symbol Overload
template <typename T> class SymbolOverrideSub {
public:
    __aicore__ inline SymbolOverrideSub(const LocalTensor<T> &src0Tensor, const LocalTensor<T> &src1Tensor)
        : src0Tensor_(src0Tensor), src1Tensor_(src1Tensor)
    {}

    __aicore__ inline void Process(const LocalTensor<T> &dstTensor) const
    {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT((CheckFuncVecBinary(dstTensor, this->src0Tensor_, this->src1Tensor_, dstTensor.GetSize(),
            "Sub operator")), { ASCENDC_REPORT_CHECK_ERROR("Sub operator", KernelFuncType::NONE_MODE);});
#endif
        SubImpl((__ubuf__ PrimT<T>*)dstTensor.GetPhyAddr(), (__ubuf__ PrimT<T>*)this->src0Tensor_.GetPhyAddr(),
            (__ubuf__ PrimT<T>*)this->src1Tensor_.GetPhyAddr(), dstTensor.GetSize());
    }

private:
    const LocalTensor<T> &src0Tensor_;
    const LocalTensor<T> &src1Tensor_;
};
// Multiplication symbol overload
template <typename T> class SymbolOverrideMul {
public:
    __aicore__ inline SymbolOverrideMul(const LocalTensor<T> &src0Tensor, const LocalTensor<T> &src1Tensor)
        : src0Tensor_(src0Tensor), src1Tensor_(src1Tensor)
    {}

    __aicore__ inline void Process(const LocalTensor<T> &dstTensor) const
    {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT((CheckFuncVecBinary(dstTensor, this->src0Tensor_, this->src1Tensor_, dstTensor.GetSize(),
            "Mul operator")), { ASCENDC_REPORT_CHECK_ERROR("Mul operator", KernelFuncType::NONE_MODE);});
#endif
        MulImpl((__ubuf__ PrimT<T>*)dstTensor.GetPhyAddr(), (__ubuf__ PrimT<T>*)this->src0Tensor_.GetPhyAddr(),
            (__ubuf__ PrimT<T>*)this->src1Tensor_.GetPhyAddr(), dstTensor.GetSize());
    }

private:
    const LocalTensor<T> &src0Tensor_;
    const LocalTensor<T> &src1Tensor_;
};
// Division symbol overload
template <typename T> class SymbolOverrideDiv {
public:
    __aicore__ inline SymbolOverrideDiv(const LocalTensor<T> &src0Tensor, const LocalTensor<T> &src1Tensor)
        : src0Tensor_(src0Tensor), src1Tensor_(src1Tensor)
    {}

    __aicore__ inline void Process(const LocalTensor<T> &dstTensor) const
    {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT((CheckFuncVecBinary(dstTensor, this->src0Tensor_, this->src1Tensor_, dstTensor.GetSize(),
            "Div operator")), { ASCENDC_REPORT_CHECK_ERROR("Div operator", KernelFuncType::NONE_MODE);});
#endif
        DivImpl((__ubuf__ PrimT<T>*)dstTensor.GetPhyAddr(), (__ubuf__ PrimT<T>*)this->src0Tensor_.GetPhyAddr(),
            (__ubuf__ PrimT<T>*)this->src1Tensor_.GetPhyAddr(), dstTensor.GetSize());
    }

private:
    const LocalTensor<T> &src0Tensor_;
    const LocalTensor<T> &src1Tensor_;
};

// bitwise and symbol overloads
template <typename T> class SymbolOverrideAnd {
public:
    __aicore__ inline SymbolOverrideAnd(const LocalTensor<T> &src0Tensor, const LocalTensor<T> &src1Tensor)
        : src0Tensor_(src0Tensor), src1Tensor_(src1Tensor)
    {}

    __aicore__ inline void Process(const LocalTensor<T> &dstTensor) const
    {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT((CheckFuncVecBinary(dstTensor, this->src0Tensor_, this->src1Tensor_, dstTensor.GetSize(),
            "And operator")), { ASCENDC_REPORT_CHECK_ERROR("And operator", KernelFuncType::NONE_MODE);});
#endif
        if constexpr(SupportType<T, int32_t, uint32_t>()) {
            AndImpl((__ubuf__ PrimT<T>*)dstTensor.GetPhyAddr(), (__ubuf__ PrimT<T>*)this->src0Tensor_.GetPhyAddr(),
                (__ubuf__ PrimT<T>*)this->src1Tensor_.GetPhyAddr(), dstTensor.GetSize() * 2);
        } else {  // mainly for uint16_t  + int16_t
            AndImpl((__ubuf__ PrimT<T>*)dstTensor.GetPhyAddr(), (__ubuf__ PrimT<T>*)this->src0Tensor_.GetPhyAddr(),
                (__ubuf__ PrimT<T>*)this->src1Tensor_.GetPhyAddr(), dstTensor.GetSize());
        }
    }

private:
    const LocalTensor<T> &src0Tensor_;
    const LocalTensor<T> &src1Tensor_;
};
// bitwise or symbol overloads
template <typename T> class SymbolOverrideOr {
public:
    __aicore__ inline SymbolOverrideOr(const LocalTensor<T> &src0Tensor, const LocalTensor<T> &src1Tensor)
        : src0Tensor_(src0Tensor), src1Tensor_(src1Tensor)
    {}

    __aicore__ inline void Process(const LocalTensor<T> &dstTensor) const
    {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT((CheckFuncVecBinary(dstTensor, this->src0Tensor_, this->src1Tensor_, dstTensor.GetSize(),
            "Or operator")), { ASCENDC_REPORT_CHECK_ERROR("Or operator", KernelFuncType::NONE_MODE);});
#endif
        if constexpr(SupportType<T, int32_t, uint32_t>()) {
            OrImpl((__ubuf__ PrimT<T>*)dstTensor.GetPhyAddr(), (__ubuf__ PrimT<T>*)this->src0Tensor_.GetPhyAddr(),
                (__ubuf__ PrimT<T>*)this->src1Tensor_.GetPhyAddr(), dstTensor.GetSize() * 2);
        } else {  // mainly for uint16_t  + int16_t
            OrImpl((__ubuf__ PrimT<T>*)dstTensor.GetPhyAddr(), (__ubuf__ PrimT<T>*)this->src0Tensor_.GetPhyAddr(),
                (__ubuf__ PrimT<T>*)this->src1Tensor_.GetPhyAddr(), dstTensor.GetSize());
        }
    }

private:
    const LocalTensor<T> &src0Tensor_;
    const LocalTensor<T> &src1Tensor_;
};

// Compare symbol overloads
template <typename T> class SymbolOverrideCompare {
public:
    __aicore__ inline SymbolOverrideCompare(const LocalTensor<T> &src0Tensor, const LocalTensor<T> &src1Tensor,
        CMPMODE cmpMode)
        : src0Tensor_(src0Tensor), src1Tensor_(src1Tensor), cmpMode_(cmpMode)
    {}

    template <typename U> __aicore__ inline void Process(const LocalTensor<U> &dstTensor) const
    {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT((CheckFuncVecBinaryCmp(dstTensor, this->src0Tensor_, this->src1Tensor_, dstTensor.GetSize(),
            "Compare operator")), { ASCENDC_REPORT_CHECK_ERROR("Compare operator", KernelFuncType::NONE_MODE);});
#endif
        VcmpvImpl((__ubuf__ U *)dstTensor.GetPhyAddr(), (__ubuf__ T *)this->src0Tensor_.GetPhyAddr(),
            (__ubuf__ T *)this->src1Tensor_.GetPhyAddr(), cmpMode_, this->src0Tensor_.GetSize());
    }

private:
    const LocalTensor<T> src0Tensor_;
    const LocalTensor<T> src1Tensor_;
    CMPMODE cmpMode_;
};
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_SYM_OVERRIDE_H
