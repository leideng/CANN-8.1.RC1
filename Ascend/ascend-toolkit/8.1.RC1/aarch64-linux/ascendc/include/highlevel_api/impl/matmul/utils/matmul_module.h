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
* \file matmul_module.h
* \brief
*/
#ifndef IMPL_MATMUL_UTILS_MATMUL_MODULE_H
#define IMPL_MATMUL_UTILS_MATMUL_MODULE_H

#include "../dfx/dfx_registry.h"
#include "../dfx/dfx_config.h"

/*                                        MatmulModuleBase                   */
namespace AscendC {
namespace Impl {
namespace Detail {
template <class... _Types>
using void_t = void;

// if user define self-implement module, but inherited from base module implemented in matmul,
// child module shoud declare : using BASE_MODULE = AscendC::XXXModuleName<xxx, xxx>;
struct MatmulNullBase {};

template <typename MODULE, typename = void>
struct MatmulModuleBase {
    using type = MatmulNullBase;
};

template <typename MODULE>
struct MatmulModuleBase<MODULE, void_t<typename MODULE::BASE_MODULE>> {
    using type = typename MODULE::BASE_MODULE;
};

template <typename MODULE, typename = void>
struct MatmulModuleBaseBase {
    using type = MatmulNullBase;
};

template <typename MODULE>
struct MatmulModuleBaseBase<MODULE, void_t<typename MODULE::BASE_MODULE::BASE_MODULE>> {
    using type = typename MODULE::BASE_MODULE::BASE_MODULE;
};

template <typename MODULE, typename = void>
struct MatmulModuleRoot {
    using type = MatmulNullBase;
};

template <typename MODULE>
struct MatmulModuleRoot<MODULE, void_t<typename MODULE::BASE_MODULE::BASE_MODULE::BASE_MODULE>> {
    using type = typename MODULE::BASE_MODULE::BASE_MODULE::BASE_MODULE;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
/*                                        MatmulImplBase                            */
#define MATMUL_IMPL__ IMPL
#define MATMUL_POLICY__ POLICY

#define MATMUL_CAST_TO_IMPL() static_cast<MATMUL_IMPL__*>(this)
#define MATMUL_CAST_TO_CONST_IMPL() static_cast<const MATMUL_IMPL__*>(this)

#define MATMUL_CAST_TO_IMPL_OF(...)        \
(static_cast<typename MATMUL_IMPL__::__VA_ARGS__*>(MATMUL_CAST_TO_IMPL()))

#define MATMUL_CAST_TO_CONST_IMPL_OF(...)        \
(static_cast<const typename MATMUL_IMPL__::__VA_ARGS__*>(MATMUL_CAST_TO_CONST_IMPL()))

#define MATMUL_CAST_TO_PROXY_OF(NAME)                           \
typename AscendC::Impl::Detail::DfxProxy<MATMUL_IMPL__, typename MATMUL_IMPL__::NAME> (*MATMUL_CAST_TO_IMPL_OF(NAME))

#define MATMUL_CAST_TO_CONST_PROXY_OF(NAME)                     \
typename AscendC::Impl::Detail::DfxProxy<const MATMUL_IMPL__, typename MATMUL_IMPL__::NAME> (*MATMUL_CAST_TO_CONST_IMPL_OF(NAME))

#define MATMUL_MODULE(NAME)  cast_to_##NAME()

#define MATMUL_USE_MODULE(NAME)                                         \
__aicore__ inline constexpr decltype(auto) MATMUL_MODULE(NAME) {        \
    if constexpr (AscendC::Impl::Detail::DfxConfig::ENABLE) {                                  \
        return MATMUL_CAST_TO_PROXY_OF(NAME);                           \
    } else {                                                            \
        return MATMUL_CAST_TO_IMPL_OF(NAME);                            \
    }                                                                   \
}                                                                       \
__aicore__ inline constexpr decltype(auto) MATMUL_MODULE(NAME) const {  \
    if constexpr (AscendC::Impl::Detail::DfxConfig::ENABLE) {                                  \
        return MATMUL_CAST_TO_CONST_PROXY_OF(NAME);                     \
    } else {                                                            \
        return MATMUL_CAST_TO_CONST_IMPL_OF(NAME);                      \
    }                                                                   \
}

#define MATMUL_USE_MODULE_ON(NAME, ...)                                  \
__aicore__ inline constexpr decltype(auto) MATMUL_MODULE(NAME) {         \
    if constexpr (AscendC::Impl::Detail::DfxConfig::ENABLE) {                                   \
        return MATMUL_CAST_TO_PROXY_OF(template NAME<__VA_ARGS__>);      \
    } else {                                                             \
        return MATMUL_CAST_TO_IMPL_OF(template NAME<__VA_ARGS__>);       \
    }                                                                    \
}                                                                        \
__aicore__ inline constexpr decltype(auto) MATMUL_MODULE(NAME) const {   \
    if constexpr (AscendC::Impl::Detail::DfxConfig::ENABLE) {                                   \
        return MATMUL_CAST_TO_CONST_PROXY_OF(template NAME<__VA_ARGS__>);\
    } else {                                                             \
        return MATMUL_CAST_TO_CONST_IMPL_OF(template NAME<__VA_ARGS__>); \
    }                                                                    \
}

/*                                         MatmulPolicy                                   */
#define MATMUL_POLICY_TEMPLATE MATMUL_POLICY

#define MATMUL_POLICY_DEFAULT_OF(DEFAULT)      \
template <const auto& = MM_CFG, typename ...>  \
        class MATMUL_POLICY = AscendC::Impl::Detail::DEFAULT

#define MATMUL_POLICY_TEMPLATE_OF(NAME)        \
template <const auto& = MM_CFG, typename ...> class NAME

#define MATMUL_IMPL_TYPE                       \
MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY_TEMPLATE>

#define MATMUL_MODULE_IN_POLICY(...)           \
MATMUL_POLICY_TEMPLATE<MM_CFG, MATMUL_IMPL_TYPE, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::__VA_ARGS__

#define MATMUL_IMPORT_MODULE(...) protected MATMUL_MODULE_IN_POLICY(__VA_ARGS__)

#define MATMUL_ALLOW_USING(NAME)                         \
using NAME = typename MATMUL_MODULE_IN_POLICY(NAME);     \
friend typename AscendC::Impl::Detail::MatmulModuleBase<NAME>::type;        \
friend typename AscendC::Impl::Detail::MatmulModuleBaseBase<NAME>::type;        \
friend typename AscendC::Impl::Detail::MatmulModuleRoot<NAME>::type;        \
friend NAME

#define MATMUL_ALLOW_USING_TEMPLATE(NAME, ...)                \
using NAME = typename MATMUL_MODULE_IN_POLICY(template NAME<__VA_ARGS__>)

/*                                        Matmul Private Module                           */
#define MATMUL_PRIVATE_TEMPLATE AscendC::Impl::Detail::MatmulPrivateModules
#define MATMUL_MODULE_IN_PRIVATE(...)                 \
MATMUL_PRIVATE_TEMPLATE<MM_CFG, MATMUL_IMPL_TYPE, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::__VA_ARGS__

#define MATMUL_IMPORT_MODULE_PRIVATE(...) protected MATMUL_MODULE_IN_PRIVATE(__VA_ARGS__)

#define MATMUL_ALLOW_USING_PRIVATE(NAME)                                \
using NAME = typename MATMUL_MODULE_IN_PRIVATE(NAME);                   \
friend typename AscendC::Impl::Detail::MatmulModuleBase<NAME>::type;    \
friend NAME

#define MATMUL_ALLOW_USING_TEMPLATE_PRIVATE(NAME, ...)                \
using NAME = typename MATMUL_MODULE_IN_PRIVATE(template NAME<__VA_ARGS__>)

/*                                       Matmul Var & Context                         */
#define MATMUL_CONTEXT()              MATMUL_CAST_TO_IMPL()->var
#define MATMUL_CONST_CONTEXT()        MATMUL_CAST_TO_CONST_IMPL()->var

#define MATMUL_PARAM_VAR           MATMUL_CONTEXT()
#define MATMUL_CONST_PARAM_VAR     MATMUL_CONST_CONTEXT()

#endif // _MATMUL_MODULE_H_
