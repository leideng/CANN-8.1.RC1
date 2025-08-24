
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FATRELU_MUL_H_
#define ACLNN_FATRELU_MUL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFatreluMulGetWorkspaceSize
 * parameters :
 * input : required
 * threshold : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFatreluMulGetWorkspaceSize(
    const aclTensor *input,
    const aclTensor *threshold,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFatreluMul
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFatreluMul(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
