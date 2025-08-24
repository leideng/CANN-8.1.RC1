
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MOE_FINALIZE_ROUTING_H_
#define ACLNN_MOE_FINALIZE_ROUTING_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeFinalizeRoutingGetWorkspaceSize
 * parameters :
 * expandedX : required
 * x1 : required
 * x2Optional : optional
 * bias : required
 * scales : required
 * expandedRowIdx : required
 * expandedExpertIdx : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeFinalizeRoutingGetWorkspaceSize(
    const aclTensor *expandedX,
    const aclTensor *x1,
    const aclTensor *x2Optional,
    const aclTensor *bias,
    const aclTensor *scales,
    const aclTensor *expandedRowIdx,
    const aclTensor *expandedExpertIdx,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMoeFinalizeRouting
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeFinalizeRouting(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
