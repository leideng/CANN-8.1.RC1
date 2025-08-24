/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description: type_def.h
 * Create: 2023-03-10
 */
#ifndef API_C_TYPE_DEF_H_
#define API_C_TYPE_DEF_H_
#include <stdint.h>
#include "error.h"

enum {
    tprt_stack_protect_weak, // 协程栈弱保护，不影响执行性能
    tprt_stack_protect_strong, // 协程栈强保护，会申请额外两页内存，影响执行性能
    tprt_stack_protect_max,
};

enum {
    tprt_sched_policy_fifo,
    tprt_sched_policy_fifo_with_parallel_constraint,
    tprt_sched_policy_max,
};

enum {
    tprt_task_priority_low,
    tprt_task_priority_normal,
    tprt_task_priority_high,
    tprt_task_priority_max,
};

typedef void* tprt_task_handle_t;
typedef tprt_task_handle_t tprt_node_handle_t;

// task define
const uint32_t tprt_task_attr_storage_size = 128U;
typedef struct {
    uint32_t branch_index;
    void *extra_info; // 用的时候再进行申请
    uint32_t extra_len;
} tprt_task_result;

typedef int32_t(*tprt_activate_cb_t)(const uint64_t, const uint32_t, uint32_t*);

typedef void* tprt_esched_context_t;

typedef tprtErr_t(*tprt_function_ptr_t)(void*);
typedef void(*tprt_void_function_ptr_t)(void*);

typedef struct {
    tprt_function_ptr_t exec;
    tprt_void_function_ptr_t destroy;
    tprt_task_result *result;
    uint64_t reserve[2];
} tprt_function_header_t;

typedef struct {
    uint32_t len;
    const void* const * items;
} tprt_deps_t;

typedef struct {
    char storage[tprt_task_attr_storage_size];
} tprt_task_attr_t;

const uint32_t tprt_auto_managed_function_storage_size = 64U + sizeof(tprt_function_header_t);

// config define
typedef void* tprt_config_t;
typedef tprtErr_t(*tprt_worker_init_cb_t)(uint32_t);

// grah define
typedef void* tprt_graph_t;

// parallel constraint define
typedef void* tprt_parallel_constraint_t;

#ifdef __cplusplus
namespace tprt {

enum class StackProtect {
    WEAK = tprt_stack_protect_weak,
    STRONG = tprt_stack_protect_strong,
    MAX = tprt_stack_protect_max,
};

enum class SchedPolicy {
    FIFO = tprt_sched_policy_fifo,
    FIFO_WITH_PARALLEL_CONSTRAINT = tprt_sched_policy_fifo_with_parallel_constraint,
    MAX = tprt_sched_policy_max,
};

enum class TaskPriority {
    LOW = tprt_task_priority_low,
    NORMAL = tprt_task_priority_normal,
    HIGH = tprt_task_priority_high,
    MAX = tprt_task_priority_max,
};
}
#endif
#endif
