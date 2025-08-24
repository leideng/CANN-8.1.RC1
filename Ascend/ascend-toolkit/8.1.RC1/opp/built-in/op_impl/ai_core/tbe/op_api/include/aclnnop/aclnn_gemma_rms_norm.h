
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GEMMA_RMS_NORM_H_
#define ACLNN_GEMMA_RMS_NORM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGemmaRmsNormGetWorkspaceSize
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
aclnnStatus aclnnGemmaRmsNormGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma,
    double epsilon,
    const aclTensor *yOut,
    const aclTensor *rstdOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGemmaRmsNorm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGemmaRmsNorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
