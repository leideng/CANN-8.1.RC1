/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description: error.h
 * Create: 2023-03-10
 */
#ifndef API_C_ERROR_H_
#define API_C_ERROR_H_
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t tprtErr_t;

static const int32_t TPRT_ERROR_NONE                     = 0;

// common
static const int32_t TPRT_ERROR_CPY_ERROR                = 0x00000001;
static const int32_t TPRT_ERROR_SYSTEM_CALL              = 0x00000002;

// drv
static const int32_t TPRT_ERROR_DRV_ERR                  = 0x11000001;
static const int32_t TPRT_ERROR_DRV_TIME_OUT             = 0x11000002;

// esched
static const int32_t TPRT_ERROR_WAIT_FOR_STOP            = 0x22000001;
static const int32_t TPRT_ERROR_STOP                     = 0x22000002;
static const int32_t TPRT_ERROR_REG_EVENT_FUNC           = 0x22000003;
static const int32_t TPRT_ERROR_EXE_EVENT_FUNC           = 0x22000004;

// scheduler
static const int32_t TPRT_ERROR_SCHEDULER_QUEUE_FULL     = 0x33000001;

// event
static const int32_t TPRT_ERROR_EVENT_STATUS             = 0xAA000001;
static const int32_t TPRT_ERROR_EVENT_REPEAT_WAIT        = 0xAA000002;
static const int32_t TPRT_ERROR_EVENT_NOT_EXIST          = 0xAA000003;
static const int32_t TPRT_ERROR_EVENT_REPEAT_ADD         = 0xAA000004;
static const int32_t TPRT_ERROR_EVENT_TYPE_ALLOC         = 0xAA000005;
static const int32_t TPRT_ERROR_EVENT_TYPE_NOT_REGISTER  = 0xAA000006;
static const int32_t TPRT_ERROR_FUNC_REPEAT_REGISTER     = 0xAA000007;
static const int32_t TPRT_ERROR_FUNC_NOT_EXIST           = 0xAA000008;
static const int32_t TPRT_ERROR_POLL_EXIT                = 0xAA000009;
static const int32_t TPRT_ERROR_EVENT_USING              = 0xAA00000A;
static const int32_t TPRT_ERROR_EVENT_NEW_FAILED         = 0xAA00000B;
static const int32_t TPRT_ERROR_EVENT_TYPE_REGISTER      = 0xAA00000C;

// mem
static const int32_t TPRT_ERROR_MEM_ALLOC                = 0xBB000001;
static const int32_t TPRT_ERROR_TASK_ALLOC               = 0xBB000002;
static const int32_t TPRT_ERROR_VERSION_ALLOC            = 0xBB000003;
static const int32_t TPRT_ERROR_MEM_COPY                 = 0xBB000004;

// task
static const int32_t TPRT_ERROR_MUTEX_BUSY               = 0xCC000001;
static const int32_t TPRT_ERROR_DEPS_ILLEGAL             = 0xCC000002; // parent indep is child outdep
static const int32_t TPRT_ERROR_HANDLE_USED_OUT_DEP      = 0xCC000003; // handle can't be used as out dependence
static const int32_t TPRT_ERROR_PARENT_IN_IS_CHILD_OUT   = 0xCC000004; // parent's indep cannot be child's outdep
static const int32_t TPRT_ERROR_PARAM_NULL               = 0xCC000005;
static const int32_t TPRT_ERROR_PARAM_INVALID            = 0xCC000006;
static const int32_t TPRT_ERROR_TPRT_UNINIT              = 0xCC000007;
static const int32_t TPRT_ERROR_TPRT_REPEAT_INIT         = 0xCC000008;
static const int32_t TPRT_ERROR_TPRT_CREATE_FAILED       = 0xCC000009;
static const int32_t TPRT_ERROR_COND_TIMEOUT             = 0xCC00000A;
static const int32_t TPRT_ERROR_READ_CFG_FILE_FAILED     = 0xCC00000B;
static const int32_t TPRT_ERROR_TASK_BRANCH_INDEX        = 0xCC00000C;
static const int32_t TPRT_ERROR_TASK_RE_EXE              = 0xCC00000D;

// group
static const int32_t TPRT_ERROR_REPEAT_ADD_WORKER        = 0xDD000001;
static const int32_t TPRT_ERROR_GROUP_WORKER_MAX         = 0xDD000002;
static const int32_t TPRT_ERROR_NO_WORKER_RESOURCE       = 0xDD000003;
static const int32_t TPRT_ERROR_REPEAT_DEL_WORKER        = 0xDD000004;
static const int32_t TPRT_ERROR_ALLOC_GROUP              = 0xDD000005;
static const int32_t TPRT_ERROR_DESTROY_GROUP            = 0xDD000006;
static const int32_t TPRT_ERROR_INVALID_GROUP_ID         = 0xDD000007;
static const int32_t TPRT_ERROR_WORKER_ACTIVATE_FAILED   = 0xDD000008;

// worker
static const int32_t TPRT_ERROR_ALLOC_WORKER             = 0xEE000001;
static const int32_t TPRT_ERROR_DESTROY_WORKER           = 0xEE000002;
static const int32_t TPRT_ERROR_THREAD_LOOP              = 0xEE000003;
static const int32_t TPRT_ERROR_GET_WORKER_ID            = 0xEE000004;

// graph
static const int32_t TPRT_ERROR_GRAPH_STATUS             = 0xFF000001;
static const int32_t TPRT_ERROR_GRAPH_EXE_FAILED         = 0xFF000002;
static const int32_t TPRT_ERROR_NORMAL_EXIT              = 0xFF000003;


#ifdef __cplusplus
}
#endif
#endif