
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SWIN_TRANSFORMER_LN_QKV_QUANT_H_
#define ACLNN_SWIN_TRANSFORMER_LN_QKV_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSwinTransformerLnQkvQuantGetWorkspaceSize
 * parameters :
 * x : required
 * gamma : required
 * beta : required
 * weight : required
 * bias : required
 * quantScale : required
 * quantOffset : required
 * dequantScale : required
 * headNum : required
 * seqLength : required
 * epsilon : required
 * oriHeight : required
 * oriWeight : required
 * hWinSize : required
 * wWinSize : required
 * weightTranspose : required
 * queryOutputOut : required
 * keyOutputOut : required
 * valueOutputOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwinTransformerLnQkvQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma,
    const aclTensor *beta,
    const aclTensor *weight,
    const aclTensor *bias,
    const aclTensor *quantScale,
    const aclTensor *quantOffset,
    const aclTensor *dequantScale,
    int64_t headNum,
    int64_t seqLength,
    double epsilon,
    int64_t oriHeight,
    int64_t oriWeight,
    int64_t hWinSize,
    int64_t wWinSize,
    bool weightTranspose,
    const aclTensor *queryOutputOut,
    const aclTensor *keyOutputOut,
    const aclTensor *valueOutputOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSwinTransformerLnQkvQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwinTransformerLnQkvQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
