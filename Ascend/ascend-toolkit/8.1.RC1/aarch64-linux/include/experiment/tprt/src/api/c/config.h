/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description: config.h
 * Create: 2023-03-10
 */
#ifndef API_C_CONFIG_H_
#define API_C_CONFIG_H_
#include "type_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* @ingroup config
* @brief create config instance
* @param [inout] cfg      def config handle, return a config instance
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_create(tprt_config_t *cfg);

/**
* @ingroup config
* @brief set cpu worker num
* @param [in] cfg      config handle
* @param [in] cpu_worker_num  set config tprt worker thread num, cpu_worker_num must greater than 0
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_set_cpu_worker_num(tprt_config_t cfg, const uint32_t cpu_worker_num);

/**
* @ingroup config
* @brief set cpu worker range
* @param [in] cfg      config handle
* @param [in] cpu_worker_begin  set config tprt worker thread begin index,
*                               cpu_worker_begin must greater than or equal to 0
* @param [in] cpu_worker_end    set config tprt worker thread end index
* cpu_worker_end must greater than cpu_worker_begin
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_set_cpu_worker_range(tprt_config_t cfg, const uint32_t cpu_worker_begin,
    const uint32_t cpu_worker_end);

/**
* @ingroup config
* @brief get cpu worker num
* @param [in] cfg      config handle
* @param [out] cpu_worker_num  get config tprt worker thread num
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_get_cpu_worker_num(tprt_config_t cfg, uint32_t *cpu_worker_num);

/**
* @ingroup config
* @brief set tprt coroutine stack size
* @param [in] cfg      config handle
* @param [in] stack_size  set coroutine stack size, stack_size must greater than or equal to 4K
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_set_stack_size(tprt_config_t cfg, const uint32_t stack_size);

/**
* @ingroup config
* @brief get tprt coroutine stack size
* @param [in] cfg      config handle
* @param [out] stack_size  get coroutine stack size
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_get_stack_size(tprt_config_t cfg, uint32_t *stack_size);

/**
* @ingroup config
* @brief set tprt coroutine stack protect type
* @param [in] cfg      config handle
* @param [in] stack_protect_type  set coroutine protect type, {tprt_stack_protect_weak, tprt_stack_protect_strong}
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_set_stack_protect(tprt_config_t cfg, const uint32_t stack_protect_type);

/**
* @ingroup config
* @brief get tprt coroutine stack protect type
* @param [in] cfg      config handle
* @param [out] stack_protect_type  coroutine protect type
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_get_stack_protect(tprt_config_t cfg, uint32_t *stack_protect_type);

/**
* @ingroup config
* @brief set tprt scheduler policy
* @param [in] cfg      config handle
* @param [in] sched_policy  set scheduler policy, {tprt_sched_policy_fifo,
                                                   tprt_sched_policy_fifo_with_parallel_constraint}
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_set_sched_policy(tprt_config_t cfg, const uint32_t sched_policy);

/**
* @ingroup config
* @brief get tprt scheduler policy
* @param [in] cfg      config handle
* @param [out] sched_policy  scheduler policy
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_get_sched_policy(tprt_config_t cfg, uint32_t *sched_policy);

/**
* @ingroup config
* @brief set tprt create thread flag on worker activation
* @param [in] cfg      config handle
* @param [in] create_thread_on_activation  set the thread creation flag when the worker is on activation, {true. false}
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_set_create_thread_on_activation(tprt_config_t cfg,
    const bool create_thread_on_activation);

/**
* @ingroup config
* @brief get tprt create thread flag on worker activation
* @param [in] cfg      config handle
* @param [out] create_thread_on_activation  the thread creation flag when the worker is on activation
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_get_create_thread_on_activation(tprt_config_t cfg, bool *create_thread_on_activation);

/**
* @ingroup config
* @brief set tprt group alloc num in advance
* @param [in] cfg      config handle
* @param [in] group_alloc_num_in_advance  set group alloc num in advance, group_alloc_num_in_advance must greater than 0
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_set_group_alloc_num_in_advance(tprt_config_t cfg, const uint32_t group_alloc_num_in_advance);

/**
* @ingroup config
* @brief get tprt group alloc num in advance
* @param [in] cfg      config handle
* @param [out] group_alloc_num_in_advance  group alloc num in advance
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_get_group_alloc_num_in_advance(tprt_config_t cfg, uint32_t *group_alloc_num_in_advance);

/**
* @ingroup config
* @brief set tprt sleep timeout
* @param [in] cfg      config handle
* @param [in] sleep_timeout  set sleep timeout, sleep_timeout must greater than 0(us) and less than 3600000000(us)
    When sleep_timeout is set to -1, the worker never enters the deactivation state after entering the sleep state
    When sleep_timeout is set to 0, the worker enters the deactivation state immediately after entering the sleep state
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_set_sleep_timeout(tprt_config_t cfg, const int64_t sleep_timeout);

/**
* @ingroup config
* @brief get tprt sleep timeout
* @param [in] cfg      config handle
* @param [out] sleep_timeout  sleep timeout
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_get_sleep_timeout(tprt_config_t cfg, int64_t *sleep_timeout);

/**
* @ingroup config
* @brief set tprt activation callback function
* @param [in] cfg      config handle
* @param [in] cb  set activation callback function
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_set_activation_callback(tprt_config_t cfg, tprt_activate_cb_t cb);

/**
* @ingroup config
* @brief set tprt esched context
* @param [in] cfg      config handle
* @param [in] ctx      esched context handle
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_set_esched_context(tprt_config_t cfg, const tprt_esched_context_t ctx);

/**
* @ingroup config
* @brief set tprt worerk init callback function
* @param [in] cfg      config handle
* @param [in] cb  set worker init callback function
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_set_worker_init_callback(tprt_config_t cfg, tprt_worker_init_cb_t worker_init_callback);

/**
* @ingroup config
* @brief set tprt device id
* @param [in] cfg      config handle
* @param [in] device_id
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_set_device_id(tprt_config_t cfg, const int32_t device_id);

/**
* @ingroup config
* @brief set tprt wait timeout
* @param [in] cfg      config handle
* @param [in] wait_timeout  set wait timeout, wait_timeout must greater than 0(us)
    When wait_timeout is set to -1, the wait does not time out
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_set_wait_timeout(tprt_config_t cfg, const int64_t wait_timeout);

/**
* @ingroup config
* @brief get tprt wait timeout
* @param [in] cfg      config handle
* @param [out] wait_timeout  wait timeout
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_get_wait_timeout(tprt_config_t cfg, int64_t *wait_timeout);

/**
* @ingroup config
* @brief destory config handle
* @param [in] cfg      config handle
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_config_destroy(tprt_config_t cfg);

#ifdef __cplusplus
}
#endif
#endif
