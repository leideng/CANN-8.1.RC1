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
 * \file kfc_register_obj.h
 * \brief
 */
#ifndef LIB_KFC_REGISTER_OBJ_H
#define LIB_KFC_REGISTER_OBJ_H
#include "kernel_operator.h"

namespace AscendC {
constexpr int8_t CUBEOBJ_MIX_MODE = 2; // 0 no, 1 all, 2 mix
template <class... Args> struct CubeObjs {};

template <class... Args> __aicore__ inline CubeObjs<Args...> GetObjType(Args...); // noexception

template <class... Args> struct GetCubeObjConfig;
template <class T, class... Args> struct GetCubeObjConfig<CubeObjs<T, Args...>> {
    static constexpr int8_t headMixDualMasterValue = GetCubeObjConfig<CubeObjs<T>>::enableMixDualMasterValue;
    static constexpr int8_t headABShareValue = GetCubeObjConfig<CubeObjs<T>>::enableABShareValue;
    static constexpr int8_t tailMixDualMasterValue = GetCubeObjConfig<CubeObjs<Args...>>::enableMixDualMasterValue;
    static constexpr int8_t tailABShareValue = GetCubeObjConfig<CubeObjs<Args...>>::enableABShareValue;

    static constexpr int8_t enableMixDualMasterValue = (headMixDualMasterValue == -1) ?
        tailMixDualMasterValue :
        (tailMixDualMasterValue == -1) ?
        headMixDualMasterValue :
        (headMixDualMasterValue == tailMixDualMasterValue) ? headMixDualMasterValue : CUBEOBJ_MIX_MODE;
    static constexpr int8_t enableABShareValue = (headABShareValue == -1) ?
        tailABShareValue :
        (tailABShareValue == -1) ? headABShareValue : (headABShareValue == tailABShareValue) ? headABShareValue : CUBEOBJ_MIX_MODE;
};
template <class T> struct GetCubeObjConfig<CubeObjs<T>> {
    static constexpr int8_t enableMixDualMasterValue = T::enableMixDualMaster;
    static constexpr int8_t enableABShareValue = T::enableABShare;
};

template <class T> struct GetCubeObjConfig<CubeObjs<T *>> {
    static constexpr int8_t enableMixDualMasterValue = -1;
    static constexpr int8_t enableABShareValue = -1;
};

template <> struct GetCubeObjConfig<CubeObjs<>> {
    static constexpr int8_t enableMixDualMasterValue = -1;
    static constexpr int8_t enableABShareValue = -1;
};

template <class T, class... Args> __aicore__ static T* GetCurTiling(T* t, Args&&... b)
{
    return t;
}

template <class T, class... Args>
__aicore__ inline void InitCurObjSkip(AscendC::TPipe* tpipe, T* a,
    Args&&... b)
{
    InitCurObj(tpipe, b...);
}

template <class T, class... Args>
__aicore__ inline void InitCurObj(AscendC::TPipe* tpipe, T& a, Args&&... b)
{
    ASSERT(tpipe != nullptr && "tpipe cannot be nullptr");
    if constexpr (sizeof...(b) == 0) {
        SetTPipe(a, tpipe);
    } else {
        auto tiling = GetCurTiling(b...);
        a.SetSubBlockIdx(0);
        a.Init(tiling, tpipe);
        if constexpr (sizeof...(b) > 1) {
            InitCurObjSkip(tpipe, b...);
        }
    }
}
}

#ifdef ASCENDC_CPU_DEBUG
#if __CCE_AICORE__ == 220
#ifdef ASCENDC_CUBE_ONLY
#define REGIST_CUBE_OBJ(tpipe, workspace, ...) \
    AscendC::InitCurObj(tpipe, __VA_ARGS__)

#else
#define REGIST_CUBE_OBJ(tpipe, workspace, ...)                                                         \
    using ASCubeObjConfig = AscendC::GetCubeObjConfig<decltype(AscendC::GetObjType(__VA_ARGS__))>;     \
    static_assert(ASCubeObjConfig::enableABShareValue != AscendC::CUBEOBJ_MIX_MODE,                    \
        "If both aType ibshare and bType ibshare are set to true, the values must "                    \
        "be the same for all cube objects.");                                                          \
    constexpr int8_t asEnableMixDualMaster = ASCubeObjConfig::enableMixDualMasterValue;                \
    static_assert(asEnableMixDualMaster != AscendC::CUBEOBJ_MIX_MODE,                                  \
        "enableMixDualMaster must be consistent for all cube objects.");                               \
    if ASCEND_IS_AIC {                                                                                 \
        AscendC::KfcServer server;                                                                     \
        server.Init(workspace);                                                                        \
        server.InitObj(tpipe, __VA_ARGS__);                                                            \
        if constexpr (!asEnableMixDualMaster) {                                                        \
            while (server.isRun()) {                                                                   \
                server.Run(__VA_ARGS__);                                                               \
            };                                                                                         \
            server.Quit();                                                                             \
            return;                                                                                    \
        }                                                                                              \
    }                                                                                                  \
    AscendC::KfcCommClient __kfcClient__(workspace, AscendC::GetSubBlockIdx(), asEnableMixDualMaster); \
    if ASCEND_IS_AIV {                                                                                 \
        if constexpr (!asEnableMixDualMaster) {                                                        \
            AscendC::g_kfcClient = &__kfcClient__;                                                     \
        } else {                                                                                       \
            AscendC::g_kfcClient = nullptr;                                                            \
        }                                                                                              \
        AscendC::SetMatrixKfc(tpipe, &__kfcClient__, 0, workspace, __VA_ARGS__);                       \
    }
#endif

#else

#define REGIST_CUBE_OBJ(tpipe, workspace, ...) \
    AscendC::InitCurObj(tpipe, __VA_ARGS__)
#endif

#else

#ifdef __DAV_C220_CUBE__
#ifdef ASCENDC_CUBE_ONLY
#define REGIST_CUBE_OBJ(tpipe, workspace, ...) \
    AscendC::InitCurObj(tpipe, __VA_ARGS__);   \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_SERVER_OBJ))

#define REGIST_CUBE_OBJ_REMOTE(tpipe, workspace, ...)
#else
#define REGIST_CUBE_OBJ(tpipe, workspace, ...)                                                             \
    using ASCubeObjConfig = AscendC::GetCubeObjConfig<decltype(AscendC::GetObjType(__VA_ARGS__))>;         \
    static_assert(ASCubeObjConfig::enableABShareValue != AscendC::CUBEOBJ_MIX_MODE,                        \
        "If both aType ibshare and bType ibshare are set to true, the values must "                        \
        "be the same for all cube objects.");                                                              \
    constexpr int8_t asEnableMixDualMaster = ASCubeObjConfig::enableMixDualMasterValue;                    \
    static_assert(asEnableMixDualMaster != AscendC::CUBEOBJ_MIX_MODE,                                      \
        "enableMixDualMaster must be consistent for all cube objects.");                                   \
    AscendC::KfcServer server;                                                                             \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_SERVER));      \
    server.Init(workspace);                                                                                \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_SERVER_INIT)); \
    server.InitObj(tpipe, __VA_ARGS__);                                                                    \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_SERVER_OBJ));  \
    if constexpr (!asEnableMixDualMaster) {                                                                \
        while (server.isRun()) {                                                                           \
            server.Run(__VA_ARGS__);                                                                       \
        };                                                                                                 \
        server.Quit();                                                                                     \
        return;                                                                                            \
    }
#endif

#elif defined(__DAV_C220_VEC__)
#ifdef ASCENDC_CUBE_ONLY
#define REGIST_CUBE_OBJ(tpipe, workspace, ...) \
    return
#else
#define REGIST_CUBE_OBJ(tpipe, workspace, ...)                                                            \
    using ASCubeObjConfig = AscendC::GetCubeObjConfig<decltype(AscendC::GetObjType(__VA_ARGS__))>;        \
    static_assert(ASCubeObjConfig::enableABShareValue != AscendC::CUBEOBJ_MIX_MODE,                       \
        "If both aType ibshare and bType ibshare are set to true, the values must "                       \
        "be the same for all cube objects.");                                                             \
    constexpr int8_t asEnableMixDualMaster = ASCubeObjConfig::enableMixDualMasterValue;                   \
    static_assert(asEnableMixDualMaster != AscendC::CUBEOBJ_MIX_MODE,                                     \
        "enableMixDualMaster must be consistent for all cube objects.");                                  \
    AscendC::KfcCommClient __kfcClient__(workspace, AscendC::GetSubBlockIdx(), asEnableMixDualMaster);    \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_CLIENT_KFC)); \
    if constexpr (!asEnableMixDualMaster) {                                                               \
        AscendC::g_kfcClient = &__kfcClient__;                                                            \
    }                                                                                                     \
    AscendC::SetMatrixKfc(tpipe, &__kfcClient__, 0, workspace, __VA_ARGS__);                              \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_MATRIX_KFC)); \
    if constexpr (!asEnableMixDualMaster) {                                                               \
        AscendC::WaitEvent(AscendC::WORKSPACE_SYNC_ID);                                                   \
    }                                                                                                     \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_WAIT_EVE))
#endif
#else

#define REGIST_CUBE_OBJ(tpipe, workspace, ...) \
    AscendC::InitCurObj(tpipe, __VA_ARGS__);   \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_OBJ))
#endif
#endif

#endif