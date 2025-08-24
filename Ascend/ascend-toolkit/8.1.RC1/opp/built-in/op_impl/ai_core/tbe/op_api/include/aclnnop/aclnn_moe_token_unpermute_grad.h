
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MOE_TOKEN_UNPERMUTE_GRAD_H_
#define ACLNN_MOE_TOKEN_UNPERMUTE_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeTokenUnpermuteGradGetWorkspaceSize
 * parameters :
 * permutedTokens : required
 * unpermutedTokensGrad : required
 * sortedIndices : required
 * probsOptional : optional
 * paddedMode : optional
 * restoreShapeOptional : optional
 * permutedTokensGradOut : required
 * probsGradOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeTokenUnpermuteGradGetWorkspaceSize(
    const aclTensor *permutedTokens,
    const aclTensor *unpermutedTokensGrad,
    const aclTensor *sortedIndices,
    const aclTensor *probsOptional,
    bool paddedMode,
    const aclIntArray *restoreShapeOptional,
    const aclTensor *permutedTokensGradOut,
    const aclTensor *probsGradOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMoeTokenUnpermuteGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeTokenUnpermuteGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
