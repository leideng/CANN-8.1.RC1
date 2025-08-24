/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 */
#ifndef MSTX_IMPL_CORE_H
#define MSTX_IMPL_CORE_H
#ifndef MSTX_IMPL_GUARD
#error Do not include this file directly, please include msToolsExt.h instead(except when MSTX_NO_IMPL is defined).
#endif

MSTX_DECLSPEC void mstxMarkA(const char *message, aclrtStream stream)
{
#ifndef MSTX_DISABLE
    mstxMarkAFunc local = g_mstxContext.mstxMarkAPtr;
    if (local != 0) {
        (*local)(message, stream);
    }
#endif // MSTX_DISABLE
}

MSTX_DECLSPEC mstxRangeId mstxRangeStartA(const char *message, aclrtStream stream)
{
#ifndef MSTX_DISABLE
    mstxRangeStartAFunc local = g_mstxContext.mstxRangeStartAPtr;
    if (local != 0) {
        return (*local)(message, stream);
    } else {
        return (mstxRangeId)0;
    }
#else
    return (mstxRangeId)0;
#endif // MSTX_DISABLE
}

MSTX_DECLSPEC void mstxRangeEnd(mstxRangeId id)
{
#ifndef MSTX_DISABLE
    mstxRangeEndFunc local = g_mstxContext.mstxRangeEndPtr;
    if (local != 0) {
        (*local)(id);
    }
#endif // MSTX_DISABLE
}

MSTX_DECLSPEC mstxMemHeapHandle_t mstxMemHeapRegister(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc)
{
#ifndef MSTX_DISABLE
    mstxMemHeapRegisterFunc local = g_mstxContext.mstxMemHeapRegisterPtr;
    if (local != NULL) {
        return (*local)(domain, desc);
    } else {
        return NULL;
    }
#else
    return NULL;
#endif // MSTX_DISABLE
}

MSTX_DECLSPEC void mstxMemHeapUnregister(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap)
{
#ifndef MSTX_DISABLE
    mstxMemHeapUnregisterFunc local = g_mstxContext.mstxMemHeapUnregisterPtr;
    if (local != NULL) {
        (*local)(domain, heap);
    }
#endif // MSTX_DISABLE
}

MSTX_DECLSPEC void mstxMemRegionsRegister(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc)
{
#ifndef MSTX_DISABLE
    mstxMemRegionsRegisterFunc local = g_mstxContext.mstxMemRegionsRegisterPtr;
    if (local != NULL) {
        (*local)(domain, desc);
    }
#endif // MSTX_DISABLE
}

MSTX_DECLSPEC void mstxMemRegionsUnregister(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc)
{
#ifndef MSTX_DISABLE
    mstxMemRegionsUnregisterFunc local = g_mstxContext.mstxMemRegionsUnregisterPtr;
    if (local != NULL) {
        (*local)(domain, desc);
    }
#endif // MSTX_DISABLE
}

MSTX_DECLSPEC mstxDomainHandle_t mstxDomainCreateA(const char *name)
{
#ifndef MSTX_DISABLE
    mstxDomainCreateAFunc local = g_mstxContext.mstxDomainCreateAPtr;
    if (local != 0) {
        return (*local)(name);
    } else {
        return (mstxDomainHandle_t)0;
    }
#else
    return (mstxDomainHandle_t)0;
#endif // MSTX_DISABLE
}

MSTX_DECLSPEC void mstxDomainDestroy(mstxDomainHandle_t domain)
{
#ifndef MSTX_DISABLE
    mstxDomainDestroyFunc local = g_mstxContext.mstxDomainDestroyPtr;
    if (local != 0) {
        (*local)(domain);
    }
#endif // MSTX_DISABLE
}

MSTX_DECLSPEC void mstxDomainMarkA(mstxDomainHandle_t domain, const char *message, aclrtStream stream)
{
#ifndef MSTX_DISABLE
    mstxDomainMarkAFunc local = g_mstxContext.mstxDomainMarkAPtr;
    if (local != 0) {
        (*local)(domain, message, stream);
    }
#endif
}

MSTX_DECLSPEC mstxRangeId mstxDomainRangeStartA(mstxDomainHandle_t domain, const char *message, aclrtStream stream)
{
#ifndef MSTX_DISABLE
    mstxDomainRangeStartAFunc local = g_mstxContext.mstxDomainRangeStartAPtr;
    if (local != 0) {
        return (*local)(domain, message, stream);
    } else {
        return (mstxRangeId)0;
    }
#else
    return (mstxRangeId)0;
#endif
}

MSTX_DECLSPEC void mstxDomainRangeEnd(mstxDomainHandle_t domain, mstxRangeId id)
{
#ifndef MSTX_DISABLE
    mstxDomainRangeEndFunc local = g_mstxContext.mstxDomainRangeEndPtr;
    if (local != 0) {
        (*local)(domain, id);
    }
#endif
}
#endif