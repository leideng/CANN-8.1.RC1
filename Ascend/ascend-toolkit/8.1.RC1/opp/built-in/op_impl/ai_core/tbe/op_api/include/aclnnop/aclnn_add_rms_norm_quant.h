
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ADD_RMS_NORM_QUANT_H_
#define ACLNN_ADD_RMS_NORM_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAddRmsNormQuantGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * gamma : required
 * scales1 : required
 * scales2Optional : optional
 * zeroPoints1Optional : optional
 * zeroPoints2Optional : optional
 * axis : optional
 * epsilon : optional
 * divMode : optional
 * y1Out : required
 * y2Out : required
 * xOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddRmsNormQuantGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *gamma,
    const aclTensor *scales1,
    const aclTensor *scales2Optional,
    const aclTensor *zeroPoints1Optional,
    const aclTensor *zeroPoints2Optional,
    int64_t axis,
    double epsilon,
    bool divMode,
    const aclTensor *y1Out,
    const aclTensor *y2Out,
    const aclTensor *xOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAddRmsNormQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddRmsNormQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
