/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 */

#ifndef MSTX_TYPES_H
#define MSTX_TYPES_H
#ifndef MSTX_IMPL_GUARD
#error Do not include this file directly, please include msToolsExt.h instead.
#endif

typedef void (* mstxMarkAFunc)(const char* message, aclrtStream stream);
typedef mstxRangeId (* mstxRangeStartAFunc)(const char* message, aclrtStream stream);
typedef void (* mstxRangeEndFunc)(mstxRangeId id);
typedef mstxMemHeapHandle_t (* mstxMemHeapRegisterFunc)(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc);
typedef void (* mstxMemHeapUnregisterFunc)(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap);
typedef void (* mstxMemRegionsRegisterFunc)(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc);
typedef void (* mstxMemRegionsUnregisterFunc)(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc);
typedef mstxDomainHandle_t (* mstxDomainCreateAFunc)(const char* name);
typedef void (* mstxDomainDestroyFunc)(mstxDomainHandle_t domain);
typedef void (* mstxDomainMarkAFunc)(mstxDomainHandle_t domain, const char *message, aclrtStream stream);
typedef mstxRangeId (* mstxDomainRangeStartAFunc)(mstxDomainHandle_t domain, const char *message, aclrtStream stream);
typedef void (* mstxDomainRangeEndFunc)(mstxDomainHandle_t domain, mstxRangeId id);

typedef enum mstxFuncModule {
    MSTX_API_MODULE_INVALID                 = 0,
    MSTX_API_MODULE_CORE                    = 1,
    MSTX_API_MODULE_CORE_DOMAIN             = 2,
    MSTX_API_MODULE_CORE_MEM                = 3,
    MSTX_API_MODULE_SIZE,                   // end of the enum, new enum items must be added before this
    MSTX_API_MODULE_FORCE_INT               = 0x7fffffff
} mstxFuncModule;

typedef enum mstxImplCoreMemFuncId {
    MSTX_API_CORE_MEM_INVALID               = 0,
    MSTX_API_CORE_MEMHEAP_REGISTER          = 1,
    MSTX_API_CORE_MEMHEAP_UNREGISTER        = 2,
    MSTX_API_CORE_MEM_REGIONS_REGISTER      = 3,
    MSTX_API_CORE_MEM_REGIONS_UNREGISTER    = 4,
    MSTX_API_CORE_MEM_SIZE,                   // end of the enum, new enum items must be added before this
    MSTX_API_CORE_MEM_FORCE_INT             = 0x7fffffff
} mstxImplCoreMemFuncId;

typedef enum mstxImplCoreFuncId {
    MSTX_API_CORE_INVALID                   = 0,
    MSTX_API_CORE_MARK_A                    = 1,
    MSTX_API_CORE_RANGE_START_A             = 2,
    MSTX_API_CORE_RANGE_END                 = 3,
    MSTX_API_CORE_SIZE,                   // end of the enum, new enum items must be added before this
    MSTX_API_CORE_FORCE_INT = 0x7fffffff
} mstxImplCoreFuncId;

typedef enum mstxImplCoreDomainFuncId {
    MSTX_API_CORE2_INVALID                 =  0,
    MSTX_API_CORE2_DOMAIN_CREATE_A         =  1,
    MSTX_API_CORE2_DOMAIN_DESTROY          =  2,
    MSTX_API_CORE2_DOMAIN_MARK_A           =  3,
    MSTX_API_CORE2_DOMAIN_RANGE_START_A    =  4,
    MSTX_API_CORE2_DOMAIN_RANGE_END        =  5,
    MSTX_API_CORE2_SIZE,                   // end of the enum, new enum items must be added before this
    MSTX_API_CORE2_FORCE_INT = 0x7fffffff
} mstxImplCoreDomainFuncId;


typedef void (* mstxFuncPointer)(void);
typedef mstxFuncPointer** mstxFuncTable;

typedef int (* mstxGetModuleFuncTableFunc)(mstxFuncModule module, mstxFuncTable *outTable, unsigned int *outSize);
typedef int (* mstxInitInjectionFunc)(mstxGetModuleFuncTableFunc getFuncTable);
#endif // MSTX_TYPES_H
