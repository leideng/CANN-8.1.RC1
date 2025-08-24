
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MOE_GATING_TOP_KSOFTMAX_V2_H_
#define ACLNN_MOE_GATING_TOP_KSOFTMAX_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize
 * parameters :
 * x : required
 * finishedOptional : optional
 * k : required
 * renorm : optional
 * outputSoftmaxResultFlag : optional
 * yOut : required
 * expertIdxOut : required
 * softmaxResultOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *finishedOptional,
    int64_t k,
    int64_t renorm,
    bool outputSoftmaxResultFlag,
    const aclTensor *yOut,
    const aclTensor *expertIdxOut,
    const aclTensor *softmaxResultOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMoeGatingTopKSoftmaxV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeGatingTopKSoftmaxV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
