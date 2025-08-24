
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ADD_LORA_H_
#define ACLNN_ADD_LORA_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAddLoraGetWorkspaceSize
 * parameters :
 * y : required
 * x : required
 * weightB : required
 * indices : required
 * weightAOptional : optional
 * layerIdx : optional
 * scale : optional
 * yOffset : optional
 * ySliceSize : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddLoraGetWorkspaceSize(
    const aclTensor *y,
    const aclTensor *x,
    const aclTensor *weightB,
    const aclTensor *indices,
    const aclTensor *weightAOptional,
    int64_t layerIdx,
    double scale,
    int64_t yOffset,
    int64_t ySliceSize,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAddLora
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddLora(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
