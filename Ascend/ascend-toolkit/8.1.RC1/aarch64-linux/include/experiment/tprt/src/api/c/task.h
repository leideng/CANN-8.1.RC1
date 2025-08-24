/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description: task.h
 * Create: 2023-03-10
 */
#ifndef API_C_TASK_H_
#define API_C_TASK_H_
#include <stdlib.h>
#include "type_def.h"
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* @ingroup task
* @brief init tprt instance
* @param [in] cfg       tprt init config
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_init(const tprt_config_t cfg);

/**
* @ingroup task
* @brief destory tprt instance
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_destroy();

/**
* @ingroup task
* @brief init tprt task attr
* @param [in] attr       task attr addr
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_init(tprt_task_attr_t *attr);

/**
* @ingroup task
* @brief set tprt task name
* @param [in] attr       task attr addr
* @param [in] name       task name
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_set_name(tprt_task_attr_t *attr, const char *name);

/**
* @ingroup task
* @brief get tprt task name
* @param [in] attr       task attr addr
* @param [out] name      task name
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_get_name(const tprt_task_attr_t *attr, const char **name);

/**
* @ingroup task
* @brief set tprt task dev
* @param [in] attr       task attr addr
* @param [in] dev        task dev {tprt_dev_cpu, tprt_dev_npu, tprt_dev_gpu}
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_set_dev(tprt_task_attr_t *attr, uint8_t dev);

/**
* @ingroup task
* @brief get tprt task dev
* @param [in] attr       task attr addr
* @param [out] dev       task dev
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_get_dev(const tprt_task_attr_t *attr, uint8_t *dev);

/**
* @ingroup task
* @brief set tprt task type
* @param [in] attr       task attr addr
* @param [in] type       task type
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_set_type(tprt_task_attr_t *attr, uint8_t type);

/**
* @ingroup task
* @brief get tprt task type
* @param [in] attr       task attr addr
* @param [out] type      task type
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_get_type(const tprt_task_attr_t *attr, uint8_t *type);

/**
* @ingroup task
* @brief set tprt task priority
* @param [in] attr       task attr addr
* @param [in] priority   task priority {tprt_task_priority_low, tprt_task_priority_normal, tprt_task_priority_high}
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_set_priority(tprt_task_attr_t *attr, uint8_t priority);

/**
* @ingroup task
* @brief get tprt task priority
* @param [in] attr       task attr addr
* @param [out] priority  task priority
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_get_priority(const tprt_task_attr_t *attr, uint8_t *priority);

/**
* @ingroup task
* @brief set tprt task group id
* @param [in] attr       task attr addr
* @param [in] group id
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_set_group_id(tprt_task_attr_t *attr, uint64_t group_id);

/**
* @ingroup task
* @brief get tprt task group id
* @param [in] attr       task attr addr
* @param [out] group id
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_get_group_id(const tprt_task_attr_t *attr, uint64_t *group_id);

/**
* @ingroup task
* @brief destory task attr
* @param [in] attr       task attr addr
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_destroy(tprt_task_attr_t *attr);

/**
* @ingroup task
* @brief set branch node num
* @param [in] attr                  task attr addr
* @param [in] branch_node_num       branch node num
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_set_branch_node_num(tprt_task_attr_t *attr, uint32_t branch_node_num);

/**
* @ingroup task
* @brief get branch node num
* @param [in] attr                   task attr addr
* @param [out] branch_node_num       branch node num
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_get_branch_node_num(tprt_task_attr_t *attr, uint32_t *branch_node_num);

/**
* @ingroup task
* @brief set wait any
* @param [in] attr                   task attr addr
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_set_wait_any(tprt_task_attr_t *attr);

/**
* @ingroup task
* @brief set parallel constraint
* @param [in] attr                   task attr addr
* @param [in] constraint             parallel constraint attr addr
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_set_parallel_constraint(tprt_task_attr_t *attr, const tprt_parallel_constraint_t constraint);

/**
* @ingroup task
* @brief get parallel constraint
* @param [in] attr         task attr addr
* @param [out] constraint  parallel constraint
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_task_attr_get_parallel_constraint(tprt_task_attr_t *attr, tprt_parallel_constraint_t *constraint);

/**
* @ingroup task
* @brief alloc a control task and return task func storage
* @return task func storage addr
*/
void* tprt_alloc_auto_managed_control_task_function_storage();

/**
* @ingroup task
* @brief alloc a task and return task func storage
* @return task func storage addr
*/
void* tprt_alloc_auto_managed_function_storage();
#ifdef __cplusplus
}
#endif

#ifdef __clang__
// clang wrapper
typedef tprtErr_t(^tprt_block_t)(tprt_task_result *result);

typedef struct {
    tprt_function_header_t header;
    tprt_block_t closure;
} c_function;

// task execute function
static tprtErr_t tprt_exec_function_wrapper(void *t)
{
    c_function *f = (c_function*)t;
    return f->closure(f->header.result);
}

// task destory function
static void tprt_destroy_function_wrapper(void *t)
{
    c_function *f = (c_function*)t;
    f->closure = NULL;
}

// check static info error
static inline void tprt_static_assert(bool cond)
{
    int32_t x(int32_t static_assertion_failed[(cond) ? 1 : -1]);
}

/**
* @ingroup task
* @brief create wrapper function
* @param [in] func       user define function execute info
* @return tprt_function_header_t
*/
static tprt_function_header_t* tprt_create_function_wrapper(const tprt_block_t func, bool is_data_task_ctx)
{
    tprt_static_assert(sizeof(c_function) <= tprt_auto_managed_function_storage_size);
    c_function *f = nullptr;
    if (is_data_task_ctx) {
        f = (c_function*)tprt_alloc_auto_managed_function_storage();
    } else {
        f = (c_function*)tprt_alloc_auto_managed_control_task_function_storage();
    }
    if (f == nullptr) {
        return nullptr;
    }
    f->header.exec = tprt_exec_function_wrapper;
    f->header.destroy = tprt_destroy_function_wrapper;
    f->header.result->branch_index = 0U;
    f->header.result->extra_info = nullptr;
    f->header.result->extra_len = 0U;
    f->closure = func;
    return (tprt_function_header_t*)f;
}
#endif
#endif
