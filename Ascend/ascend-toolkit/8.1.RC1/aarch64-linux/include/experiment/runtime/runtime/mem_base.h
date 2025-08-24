/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: mem_base.h
 * Create: 2025-03-04
 */

#ifndef CCE_MEM_BASE_H
#define CCE_MEM_BASE_H

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum {
    RT_MEMCPY_KIND_HOST_TO_HOST = 0,  // host to host
    RT_MEMCPY_KIND_HOST_TO_DEVICE,    // host to device
    RT_MEMCPY_KIND_DEVICE_TO_HOST,    // device to host
    RT_MEMCPY_KIND_DEVICE_TO_DEVICE,  // device to device, 1P && P2P
    RT_MEMCPY_KIND_DEFAULT,           // auto infer copy dir  
    RT_MEMCPY_KIND_HOST_TO_BUF_TO_DEVICE, // host to device ex (only used for 8 bytes) 解决host内存是栈内存和需要立即回收的场景 
    RT_MEMCPY_KIND_INNER_DEVICE_TO_DEVICE,  // 片内 D2D
    RT_MEMCPY_KIND_INTER_DEVICE_TO_DEVICE,  // 跨片 D2D
    RT_MEMCPY_KIND_MAX,
} rtMemcpyKind;

typedef enum {
    RT_HOST_REGISTER_MAPPED = 0, // HOST_MEM map to device  
    RT_HOST_REGISTER_MAX
} rtHostRegisterTpye;

typedef rtHostRegisterTpye rtHostRegisterType;

typedef enum  {     
    RT_MEMORY_LOC_HOST = 0, 
    RT_MEMORY_LOC_DEVICE,
    RT_MEMORY_LOC_MAX,
} rtMemLocationType;  
 
typedef struct {
    uint32_t id; // type=RT_MEMORY_LOC_DEVICE, id为deviceId
    rtMemLocationType type;  // 当前仅支持RT_MEMORY_LOC_DEVICE，Device上的内存
} rtMemLocation;
 
typedef struct {    
    rtMemLocation location;       
    uint32_t pageSize;
    uint32_t rsv[4];  // 预留字段 ，后续待驱动整改后返回内存类型
} rtPtrAttributes_t;

typedef struct{
    void *dst;
    void *src;
    uint64_t dstPitch;
    uint64_t srcPitch;
    uint64_t width;
    uint64_t height;
    rtMemcpyKind kind;
} rtMemcpy2DParams_t;

/* 
    新的接口命名规则，enum类型不加 _t 
*/
typedef enum { 
    RT_MEM_ADVISE_NONE = 0, 
    RT_MEM_ADVISE_DVPP, 
    RT_MEM_ADVISE_TS, 
    RT_MEM_ADVISE_CACHED, 
} rtMallocAdvise;

/*
 与aclrtMemMallocPolicy保持一致
*/
typedef enum {
    RT_MEM_MALLOC_HUGE_FIRST,
    RT_MEM_MALLOC_HUGE_ONLY,
    RT_MEM_MALLOC_NORMAL_ONLY,
    RT_MEM_MALLOC_HUGE_FIRST_P2P,
    RT_MEM_MALLOC_HUGE_ONLY_P2P,
    RT_MEM_MALLOC_NORMAL_ONLY_P2P,

    RT_MEM_TYPE_LOW_BAND_WIDTH = 0x0100, // DDR type -> RT_MEMORY_DDR
    RT_MEM_TYPE_HIGH_BAND_WIDTH = 0x1000, // HBM type -> RT_MEMORY_HBM

    RT_MEM_ACCESS_USER_SPACE_READONLY = 0x100000, // use for dvpp
} rtMallocPolicy;

typedef enum {
    RT_MEM_MALLOC_ATTR_RSV = 0,
    RT_MEM_MALLOC_ATTR_MODULE_ID,  // 申请内存的模块id
    RT_MEM_MALLOC_ATTR_DEVICE_ID, // 指定deviceId申请内存
    RT_MEM_MALLOC_ATTR_MAX
} rtMallocAttr;

typedef union {
    uint16_t moduleId;  // 默认不配置时，为RUNTIME_ID
    uint32_t deviceId; // 默认不配置时，为ctx的deviceId
    uint8_t rsv[8]; // 预留8字节
} rtMallocAttrValue;

typedef struct {
    rtMallocAttr attr;
    rtMallocAttrValue value;
} rtMallocAttribute_t;

typedef struct {
    rtMallocAttribute_t *attrs;
    size_t numAttrs;
} rtMallocConfig_t;

typedef struct {  // use for rtMallocAttrValue
    uint16_t moduleId;
    uint32_t deviceId;
}rtConfigValue_t; 

#if defined(__cplusplus)
}
#endif

#endif  // CCE_MEM_BASE_H
