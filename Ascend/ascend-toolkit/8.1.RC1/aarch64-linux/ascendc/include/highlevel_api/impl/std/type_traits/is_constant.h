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
 * \file is_constant.h
 * \brief
 */
 #ifndef IMPL_STD_ASCENDC_STD_IS_CONSTANT_IMPL__H
 #define IMPL_STD_ASCENDC_STD_IS_CONSTANT_IMPL__H
 
 #include "integral_constant.h"
 
 namespace AscendC {
 namespace Std {
 
 template <auto n, typename T>
 struct is_constant : false_type {};
 
 template <auto n, typename T>
 struct is_constant<n, T const> : is_constant<n,T> {};
 
 template <auto n, typename T>
 struct is_constant<n, T const&> : is_constant<n,T> {};
 
 template <auto n, typename T>
 struct is_constant<n, T &> : is_constant<n,T> {};
 
 template <auto n, typename T>
 struct is_constant<n, T &&> : is_constant<n,T> {};
 
 template <auto n, typename T, T v>
 struct is_constant<n, integral_constant<T,v>> : bool_constant<v == n> {};
 
 }
 }
 
 #endif // IMPL_STD_ASCENDC_STD_IS_CONSTANT_IMPL__H
