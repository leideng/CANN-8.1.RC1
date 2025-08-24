
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_APPLY_FUSED_EMA_ADAM_H_
#define ACLNN_APPLY_FUSED_EMA_ADAM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnApplyFusedEmaAdamGetWorkspaceSize
 * parameters :
 * grad : required
 * varRef : required
 * mRef : required
 * vRef : required
 * sRef : required
 * step : required
 * lr : optional
 * emaDecay : optional
 * beta1 : optional
 * beta2 : optional
 * eps : optional
 * mode : optional
 * biasCorrection : optional
 * weightDecay : optional
 * varRef : required
 * mRef : required
 * vRef : required
 * sRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnApplyFusedEmaAdamGetWorkspaceSize(
    const aclTensor *grad,
    aclTensor *varRef,
    aclTensor *mRef,
    aclTensor *vRef,
    aclTensor *sRef,
    const aclTensor *step,
    double lr,
    double emaDecay,
    double beta1,
    double beta2,
    double eps,
    int64_t mode,
    bool biasCorrection,
    double weightDecay,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnApplyFusedEmaAdam
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnApplyFusedEmaAdam(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
