/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description: task.hpp
 * Create: 2023-03-10
 */
#ifndef API_CPP_TASK_HPP_
#define API_CPP_TASK_HPP_
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include "c/task.h"
#include "config.hpp"

namespace tprt {
using task_func = std::function<tprtErr_t(tprt_task_result *)>;

class task_attr : public tprt_task_attr_t {
public:
    /**
     * @brief Construct a task attr object
     */
    task_attr()
    {
        tprt_task_attr_init(this);
    }

    /**
     * @brief Destory a task attr object
     */
    ~task_attr()
    {
        tprt_task_attr_destroy(this);
    }

    /**
     * @brief Set task name
     */
    inline task_attr& set_name(const char *name)
    {
        tprt_task_attr_set_name(this, name);
        return *this;
    }

    /**
     * @brief Get task name
     */
    inline const char* get_name() const
    {
        const char *name = nullptr;
        tprt_task_attr_get_name(this, &name);
        return name;
    }

    /**
     * @brief Set priority
     */
    inline task_attr& set_priority(uint8_t priority)
    {
        tprt_task_attr_set_priority(this, priority);
        return *this;
    }

    /**
     * @brief Get priority
     */
    inline uint8_t get_priority() const
    {
        uint8_t priority = 0;
        tprt_task_attr_get_priority(this, &priority);
        return priority;
    }

    /**
     * @brief Set dev
     */
    inline task_attr& set_dev(uint8_t dev)
    {
        tprt_task_attr_set_dev(this, dev);
        return *this;
    }

    /**
     * @brief Get dev
     */
    inline uint8_t get_dev() const
    {
        uint8_t dev = 0;
        tprt_task_attr_get_dev(this, &dev);
        return dev;
    }

    /**
     * @brief Set type
     */
    inline task_attr& set_type(uint8_t type)
    {
        tprt_task_attr_set_type(this, type);
        return *this;
    }

    /**
     * @brief Get type
     */
    inline uint8_t get_type() const
    {
        uint8_t type = 0;
        tprt_task_attr_get_type(this, &type);
        return type;
    }

    /**
     * @brief Set groupId
     */
    inline task_attr& set_group_id(const uint64_t &group_id)
    {
        tprt_task_attr_set_group_id(this, group_id);
        return *this;
    }

    /**
     * @brief Get groupId
     */
    inline uint64_t get_group_id() const
    {
        uint64_t group_id = 0UL;
        tprt_task_attr_get_group_id(this, &group_id);
        return group_id;
    }

    /**
     * @brief Set branchNodeNum
     */
    inline task_attr& set_branch_node_num(const uint32_t &branch_node_num)
    {
        (void)tprt_task_attr_set_branch_node_num(this, branch_node_num);
        return *this;
    }

    /**
     * @brief Get branchNodeNum
     */
    inline uint32_t get_branch_node_num()
    {
        uint32_t branchNodeNum = 0U;
        (void)tprt_task_attr_get_branch_node_num(this, &branchNodeNum);
        return branchNodeNum;
    }


    /**
     * @brief Set wait any 
     */
    inline task_attr& set_wait_any()
    {
        (void)tprt_task_attr_set_wait_any(this);
        return *this;
    }


    /**
     * @brief Set parallel constraint
     */
    inline task_attr& set_parallel_constraint(tprt_parallel_constraint_t constraint)
    {
        (void)tprt_task_attr_set_parallel_constraint(this, constraint);
        return *this;
    }

    /**
     * @brief Get parallel constraint
     */
    inline tprt_parallel_constraint_t get_parallel_constraint()
    {
        tprt_parallel_constraint_t constraint = nullptr;
        tprt_task_attr_get_parallel_constraint(this, &constraint);
        return constraint;
    }
};

#ifdef TPRT_USE_NATIVE_API
#else
template<class T>
struct get_raw_type {
    using type = T;
};

template<class T>
struct get_raw_type<const T &> {
    using type = T;
};

template<class T>
struct get_raw_type<T &&> {
    using type = T;
};

template<class T>
struct function {
    template<class CT>
    function(tprt_function_header_t h, CT &&c) : header(h), closure(std::forward<CT>(c)) {}
    tprt_function_header_t header;
    T closure;
};

template<class T>
tprtErr_t exec_function_wrapper(void *t)
{
    using function_type = function<typename get_raw_type<T>::type>;
    auto f = (function_type*)t;
    return f->closure(f->header.result);
}

template<class T>
void destroy_function_wrapper(void *t)
{
    using function_type = function<typename get_raw_type<T>::type>;
    auto f = (function_type*)t;
    f->closure = nullptr;
}

template<class T>
inline tprt_function_header_t* create_control_task_function_wrapper(T &&func)
{
    using function_type = function<typename get_raw_type<T>::type>;
    static_assert(sizeof(function_type) <= tprt_auto_managed_function_storage_size,
        "size of function must be less than tprt_auto_managed_function_storage_size");
    auto p = tprt_alloc_auto_managed_control_task_function_storage();
    if (p == nullptr) {
        return nullptr;
    }
    auto f = new (p) function_type({
        exec_function_wrapper<T>, destroy_function_wrapper<T>, nullptr, {0UL, 0UL}
    }, std::forward<T>(func));
    return (tprt_function_header_t*)f;
}
#endif
/**
 * @brief  Init tprt instance
 */
static inline tprtErr_t initialization(const config &cfg)
{
    return tprt_init(*(tprt_config_t*)&cfg);
}

/**
 * @brief  Destory tprt instance
 */
static inline tprtErr_t destroy()
{
    return tprt_destroy();
}

}
#endif
