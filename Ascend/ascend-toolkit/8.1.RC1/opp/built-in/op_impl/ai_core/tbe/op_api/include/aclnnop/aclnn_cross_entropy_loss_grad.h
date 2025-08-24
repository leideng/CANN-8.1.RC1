
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CROSS_ENTROPY_LOSS_GRAD_H_
#define ACLNN_CROSS_ENTROPY_LOSS_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCrossEntropyLossGradGetWorkspaceSize
 * parameters :
 * gradLoss : required
 * logProb : required
 * target : required
 * weightOptional : optional
 * gradZlossOptional : optional
 * lseForZlossOptional : optional
 * reductionOptional : optional
 * ignoreIndex : optional
 * labelSmoothing : optional
 * lseSquareScaleForZloss : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCrossEntropyLossGradGetWorkspaceSize(
    const aclTensor *gradLoss,
    const aclTensor *logProb,
    const aclTensor *target,
    const aclTensor *weightOptional,
    const aclTensor *gradZlossOptional,
    const aclTensor *lseForZlossOptional,
    char *reductionOptional,
    int64_t ignoreIndex,
    double labelSmoothing,
    double lseSquareScaleForZloss,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnCrossEntropyLossGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCrossEntropyLossGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
