/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description: graph.hpp
 * Create: 2023-06-07
 */
#ifndef API_CPP_GRAPH_HPP_
#define API_CPP_GRAPH_HPP_
#include "c/graph.h"
#include "task.hpp"

namespace tprt {
class graph {
public:
    /**
     * @brief Construct a new graph object
     */
    graph()
    {
        tprt_graph_create(&graph_impl_);
    }

    /**
     * @brief Destroy a graph handle
     */
    ~graph() noexcept
    {
        tprt_graph_destroy(graph_impl_);
    }

    /**
     * @brief Add node in graph with given func and its dependency nodes
     */
    tprt_node_handle_t add_node(task_func &func,
        const std::vector<tprt_node_handle_t> &pred_nodes, const task_attr &attr)
    {
        return add_node(std::move(func), pred_nodes, attr);
    }

#ifdef TPRT_USE_NATIVE_API
    tprt_node_handle_t add_node(task_func &&func,
        const std::vector<tprt_node_handle_t> &pred_nodes, const task_attr &attr);
#else
    inline tprt_node_handle_t add_node(task_func &&func,
        const std::vector<tprt_node_handle_t> &pred_nodes, const task_attr &attr)
    {
        tprt_deps_t in_deps{static_cast<uint32_t>(pred_nodes.size()), pred_nodes.data()};
        return tprt_graph_add_node_base(graph_impl_,
            create_control_task_function_wrapper(std::move(func)), &in_deps, &attr);
    }
#endif

    /**
     * @brief Add pre-order dependencies for constructed node
     */
    inline tprtErr_t add_pred(tprt_node_handle_t node, const std::vector<tprt_node_handle_t> &pred_nodes)
    {
        tprt_deps_t in_deps{static_cast<uint32_t>(pred_nodes.size()), pred_nodes.data()};
        return tprt_graph_node_add_pred(graph_impl_, node, &in_deps);
    }

    /**
     * @brief Add successor nodes for constructed node
     */
    inline tprtErr_t add_succ(tprt_node_handle_t node, const std::vector<tprt_node_handle_t> &succ_nodes)
    {
        tprt_deps_t out_deps{static_cast<uint32_t>(succ_nodes.size()), succ_nodes.data()};
        return tprt_graph_node_add_succ(graph_impl_, node, &out_deps);
    }

    inline tprtErr_t exec_sync()
    {
        return tprt_graph_exec_sync(graph_impl_);
    }

    inline tprtErr_t exec_async()
    {
        return tprt_graph_exec_async(graph_impl_);
    }

    /**
     * @brief Wait until graph execute completed
     */
    tprtErr_t wait()
    {
        return tprt_graph_wait(graph_impl_);
    }

    /**
     * @brief Set branch index into result
     */
    tprtErr_t set_task_branch_idx(tprt_task_result *task_result, const uint32_t branch_index)
    {
        return tprt_graph_set_task_branch_idx(task_result, branch_index);
    }

    /**
     * @brief Set task error into result
     */
    tprtErr_t set_task_error(tprt_task_result *task_result, const void *result, const uint32_t size)
    {
        return tprt_graph_set_task_error(task_result, result, size);
    }

    /**
     * @brief Get task error info addr pointer in graph
     */
    inline void *get_error_storage() const
    {
        void *error_info = nullptr;
        tprt_graph_get_error_storage(graph_impl_, &error_info);
        return error_info;
    }

    /**
     * @brief Add parallel constraint obj to graph
     */
    inline tprt_parallel_constraint_t add_parallel_constraint(const uint32_t parallel_num)
    {
        tprt_parallel_constraint_t constraint = nullptr;
        if (tprt_graph_add_parallel_constraint(graph_impl_, parallel_num, &constraint) != TPRT_ERROR_NONE) {
            return nullptr;
        }
        return constraint;
    }

    /**
     * @brief Get parallel num from constraint obj
     */
    inline uint32_t get_parallel_num(tprt_parallel_constraint_t constraint)
    {
        uint32_t parallel_num  = 0U;
        (void)tprt_graph_get_parallel_num(constraint, &parallel_num);
        return parallel_num;
    }

    graph(graph const&) = delete;
    void operator=(graph const&) = delete;
private:
    uint32_t get_graph_status();

private:
    tprt_graph_t graph_impl_;
};

}
#endif
