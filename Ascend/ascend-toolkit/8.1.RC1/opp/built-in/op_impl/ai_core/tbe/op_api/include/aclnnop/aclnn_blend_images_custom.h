
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BLEND_IMAGES_CUSTOM_H_
#define ACLNN_BLEND_IMAGES_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBlendImagesCustomGetWorkspaceSize
 * parameters :
 * rgb : required
 * alpha : required
 * frame : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBlendImagesCustomGetWorkspaceSize(
    const aclTensor *rgb,
    const aclTensor *alpha,
    const aclTensor *frame,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBlendImagesCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBlendImagesCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
