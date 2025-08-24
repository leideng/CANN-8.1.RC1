
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MOE_FINALIZE_ROUTING_V2GRAD_H_
#define ACLNN_MOE_FINALIZE_ROUTING_V2GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize
 * parameters :
 * gradY : required
 * expandedRowIdx : required
 * expandedXOptional : optional
 * scalesOptional : optional
 * expertIdxOptional : optional
 * biasOptional : optional
 * dropPadMode : optional
 * activeNum : optional
 * expertNum : optional
 * expertCapacity : optional
 * gradExpandedXOut : required
 * gradScalesOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(
    const aclTensor *gradY,
    const aclTensor *expandedRowIdx,
    const aclTensor *expandedXOptional,
    const aclTensor *scalesOptional,
    const aclTensor *expertIdxOptional,
    const aclTensor *biasOptional,
    int64_t dropPadMode,
    int64_t activeNum,
    int64_t expertNum,
    int64_t expertCapacity,
    const aclTensor *gradExpandedXOut,
    const aclTensor *gradScalesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMoeFinalizeRoutingV2Grad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeFinalizeRoutingV2Grad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
