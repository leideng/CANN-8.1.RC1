/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */

#ifndef MSTX_TOOLS_EXT_MEM
#define MSTX_TOOLS_EXT_MEM

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct mstxDomainRegistration_st;
typedef struct mstxDomainRegistration_st mstxDomainRegistration_t;
typedef mstxDomainRegistration_t* mstxDomainHandle_t;

struct mstxMemHeap_st;
typedef struct mstxMemHeap_st mstxMemHeap_t;
typedef mstxMemHeap_t* mstxMemHeapHandle_t;

struct mstxMemRegion_st;
typedef struct mstxMemRegion_st mstxMemRegion_t;
typedef mstxMemRegion_t* mstxMemRegionHandle_t;

/** @brief Structure to describe a memory virtual range
 * @member deviceId - device id which the memory belongs to
 * @member ptr - memory pointer
 * @member size - memory size
 */
typedef struct mstxMemVirtualRangeDesc_t {
    uint32_t deviceId;
    void const *ptr;
    uint64_t size;
} mstxMemVirtualRangeDesc_t;

/** @brief Usage characteristics of the heap
 * Usage characteristics help tools like memcheckers, sanitizer,
 * as well as other debugging and profiling tools to determine some
 * special behaviors they should apply to the heap and it's regions.
 */
typedef enum mstxMemHeapUsageType {
     /** @brief This heap is a sub-allocator
     * Heap created with this usage should not be accessed by the user until regions are registered.
     */
    MSTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR = 0,
} mstxMemHeapUsageType;

/** @brief Memory type characteristics of the heap
 * The 'type' indicates how to interpret the ptr field of the heapDesc.
 * This is intended to support many additional types of memory, beyond
 * standard process virtual memory, such as API specific memory only
 * addressed by handles or multi-dimensional memory requiring more complex
 * descriptions to handle features like strides, tiling, or interlace.
 */
typedef enum mstxMemType {
     /** @brief Standard process userspace virtual addresses for linear allocations.
     * mstxMemHeapRegister receives a heapDesc of type mstxMemVirtualRangeDesc_t
     */
    MSTX_MEM_TYPE_VIRTUAL_ADDRESS = 0,
} mstxMemType;

/** @brief Structure to describe a memory heap register description
 * @member usage - usage characteristics of the heap
 * @member type - memory type characteristics of the heap
 * @member typeSpecificDesc - memory heap description for specific memory type
 */
typedef struct mstxMemHeapDesc_t {
    mstxMemHeapUsageType usage;
    mstxMemType type;
    void const *typeSpecificDesc;
} mstxMemHeapDesc_t;

/** @brief Description for a batch of memory regions
 * @member heap - the memory pool which the memory regions belong to
 * @member regionType - memory type characteristics of the heap
 * @member regionCount - amount of memory region descriptions
 * @member regionDescArray - array of memory region descriptions
 * @member regionHandleArrayOut - [out] array of handles that stands for registered memory regions
 */
typedef struct mstxMemRegionsRegisterBatch_t {
    mstxMemHeapHandle_t heap;
    mstxMemType regionType;
    size_t regionCount;
    void const *regionDescArray;
    mstxMemRegionHandle_t* regionHandleArrayOut;
} mstxMemRegionsRegisterBatch_t;


typedef enum mstxMemRegionRefType {
    MSTX_MEM_REGION_REF_TYPE_POINTER = 0,
    MSTX_MEM_REGION_REF_TYPE_HANDLE
} mstxMemRegionRefType;

/** @brief Memory region reference
  * @member refType - the way the reference is described
  * @member pointer - reference is described via pointer, and the pointer is saved
  * @member handle - reference is described via handle, and the handle is saved
  */
typedef struct mstxMemRegionRef_t {
    mstxMemRegionRefType refType;
    union {
        void const *pointer;
        mstxMemRegionHandle_t handle;
    };
} mstxMemRegionRef_t;

typedef struct mstxMemRegionsUnregisterBatch_t {
    size_t refCount;
    mstxMemRegionRef_t const *refArray;
} mstxMemRegionsUnregisterBatch_t;

static mstxDomainHandle_t const globalDomain = NULL;

/**
 * @ingroup MSTX
 * @brief Register a memory heap as a memory pool
 *
 * @param domain [IN]    domain the memory pool belongs to
 * @param desc   [IN]    memory range description for memory pool
 * @retval mstxMemHeapHandle_t Return memory pool
 */
MSTX_DECLSPEC mstxMemHeapHandle_t mstxMemHeapRegister(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc);

/**
 * @ingroup MSTX
 * @brief Unregister a memory pool
 *
 * @param domain [IN]    domain the memory pool belongs to
 * @param heap   [IN]    memory pool to unregister
 */
MSTX_DECLSPEC void mstxMemHeapUnregister(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap);

/**
 * @ingroup MSTX
 * @brief Register suballocations within a pool
 *
 * @param domain - domain the memory pool belongs to
 * @param desc - description of a batch of memory regions
 */
MSTX_DECLSPEC void mstxMemRegionsRegister(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc);

/**
 * @ingroup MSTX
 * @brief Unregister suballocations within a pool
 * @param domain - domain the suballocations belongs to
 * @param desc - description of a batch of region references
 */
MSTX_DECLSPEC void mstxMemRegionsUnregister(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // MSTX_TOOLS_EXT_MEM
