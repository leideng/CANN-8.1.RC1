
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MOE_COMPUTE_EXPERT_TOKENS_H_
#define ACLNN_MOE_COMPUTE_EXPERT_TOKENS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeComputeExpertTokensGetWorkspaceSize
 * parameters :
 * sortedExperts : required
 * numExperts : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeComputeExpertTokensGetWorkspaceSize(
    const aclTensor *sortedExperts,
    int64_t numExperts,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMoeComputeExpertTokens
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeComputeExpertTokens(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
