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
 * \file template_argument.h
 * \brief
 */

 #ifndef TEMPLATE_ARGUMENT_H
 #define TEMPLATE_ARGUMENT_H

#include <cstdint>
#ifndef ASCENDC_TPL_KERNEL
#include <iostream>
#include <vector>
#include <cstring>
#endif

// bitWidth
#define ASCENDC_TPL_1_BW 1
#define ASCENDC_TPL_2_BW 2
#define ASCENDC_TPL_4_BW 4
#define ASCENDC_TPL_8_BW 8
// Declare
#define ASCENDC_TPL_DTYPE 0
#define ASCENDC_TPL_FORMAT 1
#define ASCENDC_TPL_UINT 2
#define ASCENDC_TPL_BOOL 3
// int type
#define ASCENDC_TPL_UI_RANGE 0
#define ASCENDC_TPL_UI_LIST 1
#define ASCENDC_TPL_UI_MIX 2

#if defined(ASCENDC_TPL_PRE)
#define ASCENDC_TPL_DTYPE_DECL(x, ...) @@ASCENDC_TPL_DTYPE_DECL_##x@@ = {__VA_ARGS__}
#define ASCENDC_TPL_FORMAT_DECL(x, ...) @@ASCENDC_TPL_FORMAT_DECL_##x@@ = {__VA_ARGS__}
#define ASCENDC_TPL_UINT_DECL(x, ...) @@ASCENDC_TPL_UINT_DECL_##x@@ = {__VA_ARGS__}
#define ASCENDC_TPL_BOOL_DECL(x, ...) @@ASCENDC_TPL_BOOL_DECL_##x@@ = {__VA_ARGS__}

#define ASCENDC_TPL_DTYPE_SEL(x, ...) @@ASCENDC_TPL_DTYPE_SEL_##x@@ = {__VA_ARGS__}
#define ASCENDC_TPL_FORMAT_SEL(x, ...) @@ASCENDC_TPL_FORMAT_SEL_##x@@ = {__VA_ARGS__}
#define ASCENDC_TPL_UINT_SEL(x, ...) @@ASCENDC_TPL_UINT_SEL_##x@@ = {__VA_ARGS__}
#define ASCENDC_TPL_BOOL_SEL(x, ...) @@ASCENDC_TPL_BOOL_SEL_##x@@ = {__VA_ARGS__}
#define ASCENDC_TPL_TILING_STRUCT_SEL(x, ...) @@ASCENDC_TPL_TILING_STRUCT_SEL_##x@@ = {__VA_ARGS__}

#define ASCENDC_TPL_ARGS_DECL(x, ...) @@ASCENDC_TPL_ARGS_DECL_##x@@ = {__VA_ARGS__}
#define ASCENDC_TPL_ARGS_SEL(...) @@{__VA_ARGS__}
#define ASCENDC_TPL_SEL(...) @@ASCENDC_TPL_LISTS@@ = {__VA_ARGS__}

#elif defined(ASCENDC_TPL_KERNEL)
#define ASCENDC_TPL_DTYPE_DECL(...)
#define ASCENDC_TPL_FORMAT_DECL(...)
#define ASCENDC_TPL_UINT_DECL(...)
#define ASCENDC_TPL_BOOL_DECL(...)

#define ASCENDC_TPL_DTYPE_SEL(...)
#define ASCENDC_TPL_FORMAT_SEL(...)
#define ASCENDC_TPL_UINT_SEL(...)
#define ASCENDC_TPL_BOOL_SEL(...)
#define ASCENDC_TPL_TILING_STRUCT_SEL(...)

#define ASCENDC_TPL_ARGS_DECL(...)
#define ASCENDC_TPL_ARGS_SEL(...)
#define ASCENDC_TPL_SEL(...)
#else
struct ParamStruct {
    const char* name;
    uint32_t paramType;
    uint8_t bitWidth;
    std::vector<uint64_t> vals;
    const char* macroType;
    ParamStruct(const char* inName, uint32_t inParamType, uint8_t inBitWidth, std::vector<uint64_t> inVals,
        const char* inMacroType):
        name(inName), paramType(inParamType), bitWidth(inBitWidth), vals(std::move(inVals)),
        macroType(inMacroType) {}
};
using TilingDeclareParams = std::vector<ParamStruct>;
using TilingSelectParams = std::vector<std::vector<ParamStruct>>;

#define ASCENDC_TPL_DTYPE_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_DTYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "DECL"}
#define ASCENDC_TPL_FORMAT_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_FORMAT, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "DECL"}
#define ASCENDC_TPL_UINT_DECL(x, bw, ...) ParamStruct{#x, ASCENDC_TPL_UINT, bw, {__VA_ARGS__}, "DECL"}
#define ASCENDC_TPL_BOOL_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_BOOL, ASCENDC_TPL_1_BW, {__VA_ARGS__}, "DECL"}

#define ASCENDC_TPL_DTYPE_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_DTYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "SEL"}
#define ASCENDC_TPL_FORMAT_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_FORMAT, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "SEL"}
#define ASCENDC_TPL_UINT_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_UINT, 0, {__VA_ARGS__}, "SEL"}
#define ASCENDC_TPL_BOOL_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_BOOL, ASCENDC_TPL_1_BW, {__VA_ARGS__}, "SEL"}
#define ASCENDC_TPL_TILING_STRUCT_SEL(x, ...)

#define ASCENDC_TPL_ARGS_DECL(x, ...) static TilingDeclareParams g_tilingDeclareParams{ __VA_ARGS__ }
#define ASCENDC_TPL_ARGS_SEL(...) { __VA_ARGS__}
#define ASCENDC_TPL_SEL(...) static TilingSelectParams g_tilingSelectParams{ __VA_ARGS__ }

namespace AscendC {
    uint64_t EncodeTilingKey(TilingDeclareParams declareParams,
                             TilingSelectParams selectParamsVec,
                             std::vector<uint64_t> tilingParams);
}

#define GET_TPL_TILING_KEY(...) \
    AscendC::EncodeTilingKey(g_tilingDeclareParams, g_tilingSelectParams, {__VA_ARGS__})

#define ASCENDC_TPL_SEL_PARAM(context, ...)           \
do {                                                  \
    uint64_t key = GET_TPL_TILING_KEY({__VA_ARGS__}); \
    context->SetTilingKey(key);                       \
} while(0)

#endif // TEMPLATEDE_TILING

#endif