
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_RMS_NORM_H_
#define ACLNN_RMS_NORM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRmsNormGetWorkspaceSize
 * parameters :
 * x : required
 * gamma : required
 * epsilon : optional
 * yOut : required
 * rstdOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmsNormGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma,
    double epsilon,
    const aclTensor *yOut,
    const aclTensor *rstdOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRmsNorm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmsNorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
