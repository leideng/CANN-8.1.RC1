
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ADD_RMS_NORM_H_
#define ACLNN_ADD_RMS_NORM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAddRmsNormGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * gamma : required
 * epsilon : optional
 * yOut : required
 * rstdOut : required
 * xOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddRmsNormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *gamma,
    double epsilon,
    const aclTensor *yOut,
    const aclTensor *rstdOut,
    const aclTensor *xOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAddRmsNorm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddRmsNorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
