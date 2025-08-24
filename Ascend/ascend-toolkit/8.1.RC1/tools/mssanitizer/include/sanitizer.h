// Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

#ifndef __HOOKS_API_SANITIZER_H__
#define __HOOKS_API_SANITIZER_H__

#ifdef __cplusplus
extern "C" {
#endif
__attribute__((weak)) void __sanitizer_report_malloc(void *ptr, uint64_t size);
__attribute__((weak)) void __sanitizer_report_free(void *ptr);
#ifdef __cplusplus
}
#endif

/// User interface
static void SanitizerReportMalloc(void *ptr, uint64_t size)
{
    if (__sanitizer_report_malloc) {
        __sanitizer_report_malloc(ptr, size);
    }
}

static void SanitizerReportFree(void *ptr)
{
    if (__sanitizer_report_free) {
        __sanitizer_report_free(ptr);
    }
}

#endif  // __HOOKS_API_SANITIZER_H__
