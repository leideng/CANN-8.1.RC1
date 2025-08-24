
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MOE_GATING_TOP_KSOFTMAX_H_
#define ACLNN_MOE_GATING_TOP_KSOFTMAX_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeGatingTopKSoftmaxGetWorkspaceSize
 * parameters :
 * x : required
 * finishedOptional : optional
 * k : required
 * yOut : required
 * expertIdxOut : required
 * rowIdxOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeGatingTopKSoftmaxGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *finishedOptional,
    int64_t k,
    const aclTensor *yOut,
    const aclTensor *expertIdxOut,
    const aclTensor *rowIdxOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMoeGatingTopKSoftmax
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeGatingTopKSoftmax(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
