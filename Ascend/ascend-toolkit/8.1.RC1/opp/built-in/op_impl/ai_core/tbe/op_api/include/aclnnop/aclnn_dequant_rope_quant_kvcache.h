
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DEQUANT_ROPE_QUANT_KVCACHE_H_
#define ACLNN_DEQUANT_ROPE_QUANT_KVCACHE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDequantRopeQuantKvcacheGetWorkspaceSize
 * parameters :
 * x : required
 * cos : required
 * sin : required
 * kCacheRef : required
 * vCacheRef : required
 * indices : required
 * scaleK : required
 * scaleV : required
 * offsetKOptional : optional
 * offsetVOptional : optional
 * weightScaleOptional : optional
 * activationScaleOptional : optional
 * biasOptional : optional
 * sizeSplits : required
 * quantModeOptional : optional
 * layoutOptional : optional
 * kvOutput : optional
 * cacheModeOptional : optional
 * qOut : required
 * kOut : required
 * vOut : required
 * kCacheRef : required
 * vCacheRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDequantRopeQuantKvcacheGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *cos,
    const aclTensor *sin,
    aclTensor *kCacheRef,
    aclTensor *vCacheRef,
    const aclTensor *indices,
    const aclTensor *scaleK,
    const aclTensor *scaleV,
    const aclTensor *offsetKOptional,
    const aclTensor *offsetVOptional,
    const aclTensor *weightScaleOptional,
    const aclTensor *activationScaleOptional,
    const aclTensor *biasOptional,
    const aclIntArray *sizeSplits,
    char *quantModeOptional,
    char *layoutOptional,
    bool kvOutput,
    char *cacheModeOptional,
    const aclTensor *qOut,
    const aclTensor *kOut,
    const aclTensor *vOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDequantRopeQuantKvcache
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDequantRopeQuantKvcache(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
