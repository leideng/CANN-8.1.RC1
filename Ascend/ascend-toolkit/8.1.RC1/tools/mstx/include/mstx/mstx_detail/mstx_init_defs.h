/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 */
#ifndef MSTX_INIT_DEFS_H
#define MSTX_INIT_DEFS_H
#ifndef MSTX_IMPL_GUARD
#error Do not include this file directly, please include msToolsExt.h instead.
#endif

MSTX_INNER_FUNC_DEFINE void mstxMarkAInit(const char* message, aclrtStream stream)
{
    mstxInitOnce();
    return mstxMarkA(message, stream);
}

MSTX_INNER_FUNC_DEFINE mstxRangeId mstxRangeStartAInit(const char* message, aclrtStream stream)
{
    mstxInitOnce();
    return mstxRangeStartA(message, stream);
}

MSTX_INNER_FUNC_DEFINE void mstxRangeEndInit(mstxRangeId id)
{
    mstxInitOnce();
    return mstxRangeEnd(id);
}

MSTX_INNER_FUNC_DEFINE mstxMemHeapHandle_t mstxMemHeapRegisterInit(mstxDomainHandle_t domain,
                                                                   mstxMemHeapDesc_t const *desc)
{
    mstxInitOnce();
    return mstxMemHeapRegister(domain, desc);
}

MSTX_INNER_FUNC_DEFINE void mstxMemHeapUnregisterInit(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap)
{
    mstxInitOnce();
    return mstxMemHeapUnregister(domain, heap);
}

MSTX_INNER_FUNC_DEFINE void mstxMemRegionsRegisterInit(mstxDomainHandle_t domain,
                                                       mstxMemRegionsRegisterBatch_t const *desc)
{
    mstxInitOnce();
    return mstxMemRegionsRegister(domain, desc);
}

MSTX_INNER_FUNC_DEFINE void mstxMemRegionsUnregisterInit(mstxDomainHandle_t domain,
                                                         mstxMemRegionsUnregisterBatch_t const *desc)
{
    mstxInitOnce();
    return mstxMemRegionsUnregister(domain, desc);
}

MSTX_INNER_FUNC_DEFINE mstxDomainHandle_t mstxDomainCreateAInit(const char *name)
{
    mstxInitOnce();
    return mstxDomainCreateA(name);
}

MSTX_INNER_FUNC_DEFINE void mstxDomainDestroyInit(mstxDomainHandle_t domain)
{
    mstxInitOnce();
    return mstxDomainDestroy(domain);
}

MSTX_INNER_FUNC_DEFINE void mstxDomainMarkAInit(mstxDomainHandle_t domain, const char *message, aclrtStream stream)
{
    mstxInitOnce();
    return mstxDomainMarkA(domain, message, stream);
}

MSTX_INNER_FUNC_DEFINE mstxRangeId mstxDomainRangeStartAInit(mstxDomainHandle_t domain, const char *message,
                                                             aclrtStream stream)
{
    mstxInitOnce();
    return mstxDomainRangeStartA(domain, message, stream);
}

MSTX_INNER_FUNC_DEFINE void mstxDomainRangeEndInit(mstxDomainHandle_t domain, mstxRangeId id)
{
    mstxInitOnce();
    return mstxDomainRangeEnd(domain, id);
}
#endif
