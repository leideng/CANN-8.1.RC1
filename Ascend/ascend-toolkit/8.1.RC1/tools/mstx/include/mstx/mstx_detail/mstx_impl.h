/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 */
#ifndef MSTX_IMPL_H
#define MSTX_IMPL_H
#ifndef MSTX_IMPL_GUARD
#error Do not include this file directly, please include ms_tools_ext.h instead(except when MSTX_NO_IMPL is defined).
#endif

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sched.h>
#include <dlfcn.h>

#include "mstx_types.h"

typedef enum mstxInitState {
    MSTX_INIT_STATE_UNINITED        = 0,
    MSTX_INIT_STATE_INITIALIZING    = 1,
    MSTX_INIT_STATE_COMPLETE        = 2,
    MSTX_INIT_STATE_FAILED          = 3,
} mstxInitState;

#if defined(__cplusplus)
    #define MSTX_INNER_FUNC_DEFINE extern "C" __attribute__((visibility("hidden"))) inline
#else
    #define MSTX_INNER_FUNC_DEFINE __attribute__((visibility("hidden")))
#endif

#ifdef __cplusplus
extern "C" {
#endif // _cplusplus

MSTX_INNER_FUNC_DEFINE int mstxGetModuleFuncTable(mstxFuncModule module,
                                                  mstxFuncTable *outTable, unsigned int *outSize);

MSTX_INNER_FUNC_DEFINE void mstxInitOnce(void);

typedef struct MstxContext_t {
    volatile unsigned int initState;
    mstxGetModuleFuncTableFunc getModuleFuncTable;

    mstxMarkAFunc mstxMarkAPtr;
    mstxRangeStartAFunc mstxRangeStartAPtr;
    mstxRangeEndFunc mstxRangeEndPtr;

    mstxDomainCreateAFunc mstxDomainCreateAPtr;
    mstxDomainDestroyFunc mstxDomainDestroyPtr;
    mstxDomainMarkAFunc mstxDomainMarkAPtr;
    mstxDomainRangeStartAFunc mstxDomainRangeStartAPtr;
    mstxDomainRangeEndFunc mstxDomainRangeEndPtr;

    mstxMemHeapRegisterFunc mstxMemHeapRegisterPtr;
    mstxMemHeapUnregisterFunc mstxMemHeapUnregisterPtr;
    mstxMemRegionsRegisterFunc mstxMemRegionsRegisterPtr;
    mstxMemRegionsUnregisterFunc mstxMemRegionsUnregisterPtr;

    mstxFuncPointer* funcTableCore[MSTX_API_CORE_SIZE + 1];
    mstxFuncPointer* funcTableCore2[MSTX_API_CORE2_SIZE + 1];
    mstxFuncPointer* funcTableMemCore[MSTX_API_CORE_MEM_SIZE + 1];
} MstxContext_t;

#include "mstx_init_defs.h"

__attribute__((visibility("hidden"))) __attribute__ ((weak)) MstxContext_t g_mstxContext = {
    MSTX_INIT_STATE_UNINITED,
    mstxGetModuleFuncTable,
    // function implement
    mstxMarkAInit,
    mstxRangeStartAInit,
    mstxRangeEndInit,

    mstxDomainCreateAInit,
    mstxDomainDestroyInit,
    mstxDomainMarkAInit,
    mstxDomainRangeStartAInit,
    mstxDomainRangeEndInit,

    mstxMemHeapRegisterInit,
    mstxMemHeapUnregisterInit,
    mstxMemRegionsRegisterInit,
    mstxMemRegionsUnregisterInit,

    // function pointers
    {
        0,
        (mstxFuncPointer*)&g_mstxContext.mstxMarkAPtr,
        (mstxFuncPointer*)&g_mstxContext.mstxRangeStartAPtr,
        (mstxFuncPointer*)&g_mstxContext.mstxRangeEndPtr,
        0
    },

    {
        0,
        (mstxFuncPointer*)&g_mstxContext.mstxDomainCreateAPtr,
        (mstxFuncPointer*)&g_mstxContext.mstxDomainDestroyPtr,
        (mstxFuncPointer*)&g_mstxContext.mstxDomainMarkAPtr,
        (mstxFuncPointer*)&g_mstxContext.mstxDomainRangeStartAPtr,
        (mstxFuncPointer*)&g_mstxContext.mstxDomainRangeEndPtr,
        0
    },

    {
        0,
        (mstxFuncPointer*)&g_mstxContext.mstxMemHeapRegisterPtr,
        (mstxFuncPointer*)&g_mstxContext.mstxMemHeapUnregisterPtr,
        (mstxFuncPointer*)&g_mstxContext.mstxMemRegionsRegisterPtr,
        (mstxFuncPointer*)&g_mstxContext.mstxMemRegionsUnregisterPtr,
        0
    },
};

#include "mstx_impl_core.h"

MSTX_INNER_FUNC_DEFINE int mstxGetModuleFuncTable(
    mstxFuncModule module, mstxFuncTable* outTable, unsigned int* outSize)
{
    size_t bytes = 0;
    mstxFuncTable table = (mstxFuncTable)0;

    switch (module) {
        case MSTX_API_MODULE_CORE:
            table = g_mstxContext.funcTableCore;
            bytes = sizeof(g_mstxContext.funcTableCore);
            break;
        case MSTX_API_MODULE_CORE_DOMAIN:
            table = g_mstxContext.funcTableCore2;
            bytes = sizeof(g_mstxContext.funcTableCore2);
            break;
        case MSTX_API_MODULE_CORE_MEM:
            table = g_mstxContext.funcTableMemCore;
            bytes = sizeof(g_mstxContext.funcTableMemCore);
            break;
        default:
            return MSTX_FAIL;
    }

    if (outSize) {
        *outSize = (unsigned int)(bytes / sizeof(mstxFuncPointer*)) - 1;
    } else {
        return MSTX_FAIL;
    }

    if (outTable) {
        *outTable = table;
    } else {
        return MSTX_FAIL;
    }
    return MSTX_SUCCESS;
}

const char *const MSTX_INJECTION_PATH = "MSTX_INJECTION_PATH";

MSTX_INNER_FUNC_DEFINE int mstxInitWithInjectionLib(void)
{
    const char *const initFuncName = "InitInjectionMstx";
    void *handle = (void *)0;
    char *injectionPath = getenv(MSTX_INJECTION_PATH);
    if (injectionPath) {
        handle = dlopen(injectionPath, RTLD_LAZY | RTLD_LOCAL);
    }
    mstxInitInjectionFunc initFunc = (mstxInitInjectionFunc)0;
    initFunc = (mstxInitInjectionFunc)dlsym(handle, initFuncName);
    if (initFunc) {
        return initFunc(g_mstxContext.getModuleFuncTable);
    }
    return MSTX_FAIL;
}

MSTX_INNER_FUNC_DEFINE void refreshUninitMstxCoreFuncPtr(int forceNull)
{
    if ((g_mstxContext.mstxMarkAPtr == mstxMarkAInit) || forceNull == 1) {
        g_mstxContext.mstxMarkAPtr = (mstxMarkAFunc)0;
    }
    if ((g_mstxContext.mstxRangeStartAPtr == mstxRangeStartAInit) || forceNull == 1) {
        g_mstxContext.mstxRangeStartAPtr = (mstxRangeStartAFunc)0;
    }
    if ((g_mstxContext.mstxRangeEndPtr == mstxRangeEndInit) || forceNull == 1) {
        g_mstxContext.mstxRangeEndPtr = (mstxRangeEndFunc)0;
    }
}

MSTX_INNER_FUNC_DEFINE void refreshUninitMstxCoreDomainFuncPtr(int forceNull)
{
    if ((g_mstxContext.mstxDomainCreateAPtr == mstxDomainCreateAInit) || forceNull == 1) {
        g_mstxContext.mstxDomainCreateAPtr = (mstxDomainCreateAFunc)0;
    }
    if ((g_mstxContext.mstxDomainDestroyPtr == mstxDomainDestroyInit) || forceNull == 1) {
        g_mstxContext.mstxDomainDestroyPtr = (mstxDomainDestroyFunc)0;
    }
    if ((g_mstxContext.mstxDomainMarkAPtr == mstxDomainMarkAInit) || forceNull == 1) {
        g_mstxContext.mstxDomainMarkAPtr = (mstxDomainMarkAFunc)0;
    }
    if ((g_mstxContext.mstxDomainRangeStartAPtr == mstxDomainRangeStartAInit) || forceNull == 1) {
        g_mstxContext.mstxDomainRangeStartAPtr = (mstxDomainRangeStartAFunc)0;
    }
    if ((g_mstxContext.mstxDomainRangeEndPtr == mstxDomainRangeEndInit) || forceNull == 1) {
        g_mstxContext.mstxDomainRangeEndPtr = (mstxDomainRangeEndFunc)0;
    }
}

MSTX_INNER_FUNC_DEFINE void refreshUninitMstxCoreMemFuncPtr(int forceNull)
{
    if ((g_mstxContext.mstxMemHeapRegisterPtr == mstxMemHeapRegisterInit) || forceNull == 1) {
        g_mstxContext.mstxMemHeapRegisterPtr = (mstxMemHeapRegisterFunc)0;
    }
    if ((g_mstxContext.mstxMemHeapUnregisterPtr == mstxMemHeapUnregisterInit) || forceNull == 1) {
        g_mstxContext.mstxMemHeapUnregisterPtr = (mstxMemHeapUnregisterFunc)0;
    }
    if ((g_mstxContext.mstxMemRegionsRegisterPtr == mstxMemRegionsRegisterInit) || forceNull == 1) {
        g_mstxContext.mstxMemRegionsRegisterPtr = (mstxMemRegionsRegisterFunc)0;
    }
    if ((g_mstxContext.mstxMemRegionsUnregisterPtr == mstxMemRegionsUnregisterInit) || forceNull == 1) {
        g_mstxContext.mstxMemRegionsUnregisterPtr = (mstxMemRegionsUnregisterFunc)0;
    }
}

MSTX_INNER_FUNC_DEFINE void refreshUninitMstxContextFuncPtr(int forceNull)
{
    refreshUninitMstxCoreFuncPtr(forceNull);
    refreshUninitMstxCoreDomainFuncPtr(forceNull);
    refreshUninitMstxCoreMemFuncPtr(forceNull);
}

MSTX_INNER_FUNC_DEFINE void mstxInitOnce()
{
    if (g_mstxContext.initState == MSTX_INIT_STATE_COMPLETE) {
        return;
    }
    __sync_synchronize();
    int state = __sync_val_compare_and_swap(&g_mstxContext.initState,
                                            MSTX_INIT_STATE_UNINITED, MSTX_INIT_STATE_INITIALIZING);
    if (state == MSTX_INIT_STATE_UNINITED) {
        int result = mstxInitWithInjectionLib();
        refreshUninitMstxContextFuncPtr(result != MSTX_SUCCESS);
        unsigned int initState = result == MSTX_SUCCESS ? MSTX_INIT_STATE_COMPLETE : MSTX_INIT_STATE_FAILED;
        __sync_lock_test_and_set(&g_mstxContext.initState, initState);
    } else {
        __sync_synchronize();
        while ((g_mstxContext.initState != MSTX_INIT_STATE_COMPLETE) &&
                (g_mstxContext.initState != MSTX_INIT_STATE_FAILED)) {
            sched_yield();
            __sync_synchronize();
        }
    }
}
#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
#endif