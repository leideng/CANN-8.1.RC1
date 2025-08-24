
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CROSS_ENTROPY_LOSS_H_
#define ACLNN_CROSS_ENTROPY_LOSS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCrossEntropyLossGetWorkspaceSize
 * parameters :
 * input : required
 * target : required
 * weightOptional : optional
 * reductionOptional : optional
 * ignoreIndex : optional
 * labelSmoothing : optional
 * lseSquareScaleForZloss : optional
 * returnZloss : optional
 * lossOut : required
 * logProbOut : required
 * zlossOut : required
 * lseForZlossOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCrossEntropyLossGetWorkspaceSize(
    const aclTensor *input,
    const aclTensor *target,
    const aclTensor *weightOptional,
    char *reductionOptional,
    int64_t ignoreIndex,
    double labelSmoothing,
    double lseSquareScaleForZloss,
    bool returnZloss,
    const aclTensor *lossOut,
    const aclTensor *logProbOut,
    const aclTensor *zlossOut,
    const aclTensor *lseForZlossOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnCrossEntropyLoss
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCrossEntropyLoss(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
