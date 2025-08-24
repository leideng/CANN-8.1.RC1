/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description: group.h
 * Create: 2023-05-04
 */
#ifndef API_C_GRAPH_H_
#define API_C_GRAPH_H_
#include "type_def.h"
#include "task.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* @ingroup graph
* @brief add node in graph with given func and its dependency nodes
* @param [in] graph          graph handle, which specifies the static graph object of the task to be added
* @param [in] func_header    user define function execute info
* @param [in] pred_nodes     pre-node dependency list of submmit node
* @param [in] attr           task attr handle
* @return tprt_task_handle_t
*/
tprt_node_handle_t tprt_graph_add_node_base(tprt_graph_t graph, tprt_function_header_t *func_header,
    const tprt_deps_t *pred_nodes, const tprt_task_attr_t *attr);
/**
* @ingroup graph
* @brief create a tprt compute graph
* @param [inout] graph    def graph handle, return a graph instance
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_graph_create(tprt_graph_t *graph);

/**
* @ingroup graph
* @brief destory graph handle
* @param [in] graph     graph handle
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_graph_destroy(tprt_graph_t graph);

/**
* @ingroup graph
* @brief add pre-order dependencies for constructed node in static graph
* @param [in] graph           graph handle
* @param [in] node            node handle
* @param [in] pred_nodes      pred nodes handle
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_graph_node_add_pred(tprt_graph_t graph, tprt_node_handle_t node, const tprt_deps_t *pred_nodes);

/**
* @ingroup graph
* @brief add successor nodes for constructed node in static graph
* @param [in] graph           graph handle
* @param [in] node            node handle
* @param [in] succ_nodes      succ nodes handle
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_graph_node_add_succ(tprt_graph_t graph, tprt_node_handle_t node, const tprt_deps_t *succ_nodes);

/**
* @ingroup graph
* @brief exe graph sync
* @param [in] graph           graph handle
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_graph_exec_sync(tprt_graph_t graph);

/**
* @ingroup graph
* @brief exe graph async
* @param [in] graph           graph handle
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_graph_exec_async(tprt_graph_t graph);

/**
* @ingroup graph
* @brief wait until graph execute completed
* @param [in] graph           graph handle
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_graph_wait(tprt_graph_t graph);

/**
* @ingroup graph
* @brief set branch index into result
* @param [in] task_result        task result handle
* @param [in] branch_idx         branch index
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_graph_set_task_branch_idx(tprt_task_result *task_result, const uint32_t branch_idx);

/**
* @ingroup graph
* @brief set task error into result
* @param [in] task_result        task result handle
* @param [in] addr               error info addr
* @param [in] size               error info length
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_graph_set_task_error(tprt_task_result *task_result, const void *addr, const uint32_t size);

/**
* @ingroup graph
* @brief get task error info addr pointer in graph
* @param [in] graph           graph handle
* @param [in] error_info      error info addr pointer
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_graph_get_error_storage(void *graph, void **error_info);

/**
* @ingroup graph
* @brief add parallel constraint obj to graph
* @param [in] graph             graph handle
* @param [in] parallel_num      parallel num
* @param [inout] constraint     def constraint handle, return a constraint instance
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_graph_add_parallel_constraint(void *graph, const uint32_t parallel_num,
    tprt_parallel_constraint_t *constraint);

/**
* @ingroup graph
* @brief get parallel num from constraint obj
* @param [in] constraint        constraint handle
* @param [out] parallel_num      parallel num
* @return TPRT_ERROR_NONE for success
* @return !=TPRT_ERROR_NONE for failed
*/
tprtErr_t tprt_graph_get_parallel_num(const tprt_parallel_constraint_t constraint, uint32_t *parallel_num);

#ifdef __cplusplus
}
#endif

#ifdef __clang__

/**
* @ingroup graph
* @brief add node in graph
* @param [in] graph          graph handle, which specifies the static graph object of the task to be added
* @param [in] func    user define function execute info
* @param [in] pred_nodes     pre-node dependency list of submmit node
* @param [in] attr           task attr handle
* @return tprt_task_handle_t
*/
static inline tprt_node_handle_t tprt_graph_add_node(tprt_graph_t graph, const tprt_block_t func,
    const tprt_deps_t *pred_nodes, const tprt_task_attr_t *attr)
{
    return tprt_graph_add_node_base(graph, tprt_create_function_wrapper(func, false), pred_nodes, attr);
}
#endif

#endif
