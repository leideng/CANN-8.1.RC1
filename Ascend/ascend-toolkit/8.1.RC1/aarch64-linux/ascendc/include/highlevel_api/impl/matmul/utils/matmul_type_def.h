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
* \file matmul_type_def.h
* \brief
*/
#ifndef IMPL_MATMUL_UTILS_MATMUL_TYPE_DEF_H
#define IMPL_MATMUL_UTILS_MATMUL_TYPE_DEF_H

#include "lib/matmul/tiling.h"

namespace AscendC {

enum class InputTypeTag : uint8_t {
    A = 0,
    B = 1,
    C = 2,
};

template <TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
          LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false>
struct MatmulType {
    constexpr static TPosition pos = POSITION;
    constexpr static CubeFormat format = FORMAT;
    using T = TYPE;
    constexpr static bool isTrans = ISTRANS;
    constexpr static LayoutMode layout = LAYOUT;
    constexpr static bool ibShare = IBSHARE;
};

template <class INPUT_TYPE, typename TRANS_TYPE = typename INPUT_TYPE::T>
struct MatmulInputAType : INPUT_TYPE {
    using TRANS_T = TRANS_TYPE;
    constexpr static InputTypeTag TAG = InputTypeTag::A;
};

template <class INPUT_TYPE, typename TRANS_TYPE = typename INPUT_TYPE::T>
struct MatmulInputBType : INPUT_TYPE {
    using TRANS_T = TRANS_TYPE;
    constexpr static InputTypeTag TAG = InputTypeTag::B;
};

template <class INPUT_TYPE, typename TRANS_TYPE = typename INPUT_TYPE::T>
struct MatmulInputCType : INPUT_TYPE {
    using TRANS_T = TRANS_TYPE;
    constexpr static InputTypeTag TAG = InputTypeTag::C;
};

template <typename Type, Type valueIn>
struct ConstantType
{
    static constexpr Type value = valueIn;
    typedef Type value_type;
    typedef ConstantType<Type, valueIn> type;
    constexpr __aicore__ inline operator value_type() const noexcept {return value;}
};


typedef ConstantType<bool, false> falseType;
typedef ConstantType<bool, true> trueType;

template <typename...> using voidT = void;

template <typename INPUT_TYPE, typename = void>
struct HasSparseIndex : falseType {};

template <typename INPUT_TYPE>
struct HasSparseIndex<INPUT_TYPE, voidT<decltype(&INPUT_TYPE::indexPosition)>> : trueType {};

template <TPosition POSITION, TPosition INDEX_POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
          LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false>
struct SparseMatmulType: public MatmulType <POSITION, FORMAT, TYPE, ISTRANS, LAYOUT, IBSHARE> {
    constexpr static TPosition indexPosition = INDEX_POSITION;
};

}  // namespace AscendC
#endif // _MATMUL_TYPE_DEF_H_