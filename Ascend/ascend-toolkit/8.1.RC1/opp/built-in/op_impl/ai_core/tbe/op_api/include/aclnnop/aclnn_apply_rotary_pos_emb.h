
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_APPLY_ROTARY_POS_EMB_H_
#define ACLNN_APPLY_ROTARY_POS_EMB_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnApplyRotaryPosEmbGetWorkspaceSize
 * parameters :
 * queryRef : required
 * keyRef : required
 * cos : required
 * sin : required
 * layout : optional
 * queryRef : required
 * keyRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnApplyRotaryPosEmbGetWorkspaceSize(
    aclTensor *queryRef,
    aclTensor *keyRef,
    const aclTensor *cos,
    const aclTensor *sin,
    int64_t layout,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnApplyRotaryPosEmb
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnApplyRotaryPosEmb(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
