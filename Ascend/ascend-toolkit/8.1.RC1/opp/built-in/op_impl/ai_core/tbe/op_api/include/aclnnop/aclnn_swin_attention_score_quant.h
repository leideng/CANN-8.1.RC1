
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SWIN_ATTENTION_SCORE_QUANT_H_
#define ACLNN_SWIN_ATTENTION_SCORE_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSwinAttentionScoreQuantGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * value : required
 * scaleQuant : required
 * scaleDequant1 : required
 * scaleDequant2 : required
 * biasQuantOptional : optional
 * biasDequant1Optional : optional
 * biasDequant2Optional : optional
 * paddingMask1Optional : optional
 * paddingMask2Optional : optional
 * queryTranspose : optional
 * keyTranspose : optional
 * valueTranspose : optional
 * softmaxAxes : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwinAttentionScoreQuantGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *scaleQuant,
    const aclTensor *scaleDequant1,
    const aclTensor *scaleDequant2,
    const aclTensor *biasQuantOptional,
    const aclTensor *biasDequant1Optional,
    const aclTensor *biasDequant2Optional,
    const aclTensor *paddingMask1Optional,
    const aclTensor *paddingMask2Optional,
    bool queryTranspose,
    bool keyTranspose,
    bool valueTranspose,
    int64_t softmaxAxes,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSwinAttentionScoreQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwinAttentionScoreQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
