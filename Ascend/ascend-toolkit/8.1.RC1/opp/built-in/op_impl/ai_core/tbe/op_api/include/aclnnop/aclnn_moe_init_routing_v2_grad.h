
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MOE_INIT_ROUTING_V2GRAD_H_
#define ACLNN_MOE_INIT_ROUTING_V2GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeInitRoutingV2GradGetWorkspaceSize
 * parameters :
 * gradExpandedX : required
 * expandedRowIdx : required
 * topK : required
 * dropPadMode : optional
 * activeNum : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeInitRoutingV2GradGetWorkspaceSize(
    const aclTensor *gradExpandedX,
    const aclTensor *expandedRowIdx,
    int64_t topK,
    int64_t dropPadMode,
    int64_t activeNum,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMoeInitRoutingV2Grad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeInitRoutingV2Grad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
