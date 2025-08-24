/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description: config.hpp
 * Create: 2023-03-10
 */
#ifndef API_CPP_CONFIG_HPP_
#define API_CPP_CONFIG_HPP_
#include "c/config.h"
#ifndef ONLY_COMPILE_OPEN_SRC
#include "esched.hpp"
#endif
#include <functional>

using tprt_worker_init_cb_func_t = std::function<tprtErr_t(uint32_t)>;
#ifdef __cplusplus
extern "C" {
#endif
tprtErr_t tprt_config_set_worker_init_callback_pro(tprt_config_t pimpl,
    tprt_worker_init_cb_func_t worker_init_callback);
#ifdef __cplusplus
}
#endif

namespace tprt {
class config {
public:
    /**
     * @brief Construct a new config object
     */
    config()
    {
        tprt_config_create(&pimpl);
    }

    /**
     * @brief Destroy the config object
     */
    ~config() noexcept
    {
        tprt_config_destroy(pimpl);
    }

    config(config const&) = delete;
    void operator=(config const&) = delete;

    /**
     * @brief Set cpu worker num
     * @param num
     */
    inline config& set_cpu_worker_num(const uint32_t cpu_worker_num)
    {
        tprt_config_set_cpu_worker_num(pimpl, cpu_worker_num);
        return *this;
    }

    /**
     * @brief Get cpu worker num
     */
    inline uint32_t get_cpu_worker_num()
    {
        uint32_t cpu_worker_num = 0U;
        tprt_config_get_cpu_worker_num(pimpl, &cpu_worker_num);
        return cpu_worker_num;
    }

    /**
     * @brief Set stack size
     * @param size
     */
    inline config& set_stack_size(const uint32_t stack_size)
    {
        tprt_config_set_stack_size(pimpl, stack_size);
        return *this;
    }

    /**
     * @brief Get stack size
     */
    inline uint32_t get_stack_size()
    {
        uint32_t stack_size = 0U;
        tprt_config_get_stack_size(pimpl, &stack_size);
        return stack_size;
    }

    /**
     * @brief Set stack protect
     * @param protect type
     */
    inline config& set_stack_protect(const enum StackProtect StackProtect)
    {
        tprt_config_set_stack_protect(pimpl, static_cast<uint32_t>(StackProtect));
        return *this;
    }

    /**
     * @brief Get stack protect type
     */
    inline enum StackProtect get_stack_protect()
    {
        uint32_t stack_size = 0U;
        tprt_config_get_stack_protect(pimpl, &stack_size);
        return static_cast<enum StackProtect>(stack_size);
    }

    /**
     * @brief Set sched policy
     * @param policy type
     */
    inline config& set_sched_policy(const enum SchedPolicy sched_policy)
    {
        tprt_config_set_sched_policy(pimpl, static_cast<uint32_t>(sched_policy));
        return *this;
    }

    /**
     * @brief Get sched policy
     */
    inline enum SchedPolicy get_sched_policy()
    {
        uint32_t sched_policy = 0U;
        tprt_config_get_sched_policy(pimpl, &sched_policy);
        return static_cast<enum SchedPolicy>(sched_policy);
    }

    /**
     * @brief Set create thread on activation flag
     * @param flag
     */
    inline config& set_create_thread_on_activation(const bool create_thread_on_activation)
    {
        tprt_config_set_create_thread_on_activation(pimpl, create_thread_on_activation);
        return *this;
    }

    /**
     * @brief Get create thread on activation flag
     */
    inline bool get_create_thread_on_activation()
    {
        bool create_thread_on_activation = true;
        tprt_config_get_create_thread_on_activation(pimpl, &create_thread_on_activation);
        return create_thread_on_activation;
    }

    /**
     * @brief Set group alloc num in advance
     * @param num
     */
    inline config& set_group_alloc_num_in_advance(const uint32_t group_alloc_num_in_advance)
    {
        tprt_config_set_group_alloc_num_in_advance(pimpl, group_alloc_num_in_advance);
        return *this;
    }

    /**
     * @brief Get group alloc num in advance
     */
    inline uint32_t get_group_alloc_num_in_advance()
    {
        uint32_t group_alloc_num_in_advance = 0U;
        tprt_config_get_group_alloc_num_in_advance(pimpl, &group_alloc_num_in_advance);
        return group_alloc_num_in_advance;
    }

    /**
     * @brief Set sleep timeout
     * @param time
     */
    inline config& set_sleep_timeout(const int64_t sleep_timeout)
    {
        tprt_config_set_sleep_timeout(pimpl, sleep_timeout);
        return *this;
    }

    /**
     * @brief Get sleep timeout
     */
    inline int64_t get_sleep_timeout()
    {
        int64_t sleep_timeout = -1;
        tprt_config_get_sleep_timeout(pimpl, &sleep_timeout);
        return sleep_timeout;
    }

    /**
     * @brief Set activation callback func
     * @param func
     */
    inline config& set_activation_callback(tprt_activate_cb_t activation_callback)
    {
        tprt_config_set_activation_callback(pimpl, activation_callback);
        return *this;
    }

    /**
     * @brief Set esched context
     * @param context
     */
#ifndef ONLY_COMPILE_OPEN_SRC
    inline config& set_esched_context(const esched_context &ctx)
    {
        tprt_config_set_esched_context(pimpl, ctx.GetHandle());
        return *this;
    }
#endif

    /**
     * @brief Set worker init callback func
     * @param func
     */
    inline config& set_worker_init_callback(tprt_worker_init_cb_func_t worker_init_callback)
    {
        tprt_config_set_worker_init_callback_pro(pimpl, worker_init_callback);
        return *this;
    }

    /**
     * @brief Set device id
     * @param func
     */
    inline config& set_device_id(int32_t device_id)
    {
        tprt_config_set_device_id(pimpl, device_id);
        return *this;
    }

    /**
     * @brief Set wait timeout
     * @param time
     */
    inline config& set_wait_timeout(const int64_t wait_timeout)
    {
        tprt_config_set_wait_timeout(pimpl, wait_timeout);
        return *this;
    }

    /**
     * @brief Get wait timeout
     */
    inline int64_t get_wait_timeout()
    {
        int64_t wait_timeout = -1;
        tprt_config_get_wait_timeout(pimpl, &wait_timeout);
        return wait_timeout;
    }
private:
    tprt_config_t pimpl;
};
}
#endif
