
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_RING_ATTENTION_UPDATE_H_
#define ACLNN_RING_ATTENTION_UPDATE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRingAttentionUpdateGetWorkspaceSize
 * parameters :
 * prevAttnOut : required
 * prevSoftmaxMax : required
 * prevSoftmaxSum : required
 * curAttnOut : required
 * curSoftmaxMax : required
 * curSoftmaxSum : required
 * actualSeqQlenOptional : optional
 * inputLayoutOptional : optional
 * attnOutOut : required
 * softmaxMaxOut : required
 * softmaxSumOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRingAttentionUpdateGetWorkspaceSize(
    const aclTensor *prevAttnOut,
    const aclTensor *prevSoftmaxMax,
    const aclTensor *prevSoftmaxSum,
    const aclTensor *curAttnOut,
    const aclTensor *curSoftmaxMax,
    const aclTensor *curSoftmaxSum,
    const aclTensor *actualSeqQlenOptional,
    char *inputLayoutOptional,
    const aclTensor *attnOutOut,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRingAttentionUpdate
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRingAttentionUpdate(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
