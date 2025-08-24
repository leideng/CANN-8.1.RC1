
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MOE_TOKEN_UNPERMUTE_H_
#define ACLNN_MOE_TOKEN_UNPERMUTE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeTokenUnpermuteGetWorkspaceSize
 * parameters :
 * permutedTokens : required
 * sortedIndices : required
 * probsOptional : optional
 * paddedMode : optional
 * restoreShapeOptional : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeTokenUnpermuteGetWorkspaceSize(
    const aclTensor *permutedTokens,
    const aclTensor *sortedIndices,
    const aclTensor *probsOptional,
    bool paddedMode,
    const aclIntArray *restoreShapeOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMoeTokenUnpermute
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeTokenUnpermute(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
