#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn
from transformers.modeling_bert import BertConfig, BertModel
from transformers.optimization import AdamW
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer

from dpr.utils.data_utils import Tensorizer
from .biencoder import BiEncoder, BiEncoderNllLoss
from .reader import Reader
from .kmeans import lloyd
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _average_sequence_embeddings(sequence_output, valid_mask):
    flags = valid_mask == 1
    lengths = torch.sum(flags, dim=-1)
    lengths = torch.clamp(lengths, 1, None)
    sequence_embeddings = torch.sum(sequence_output * flags[:, :, None], dim=1)
    sequence_embeddings = sequence_embeddings / lengths[:, None]
    return sequence_embeddings


# def attn_scores(a: T, b: T) -> T:
#     # q_vector: n1 x D, ctx_vectors: n2 x len x D, result n1 x n2
#     weights = F.softmax((a * b).sum(-1), dim=-1)
#     c = torch.sum(weights.unsqueeze(-1) * b, dim=1)
#     output = torch.sum(a * c, dim=-1)
#     return output


def dot_attention(q, v, v_mask=None, dropout=None):
    # q [bs1, dim]
    # v [bs2, n_cluster, dim]
    # output [bs1, bs2]
    attention_weights = torch.matmul(v, q.T)  # [bs2, n_cluster, bs1]
    attention_weights = F.softmax(attention_weights, dim=1)
    final_v = torch.einsum('jki,jkl->ijl', [attention_weights, v])  # [bs1, bs2, dim]
    output = torch.sum(final_v * q.unsqueeze(1), dim=-1)  # [bs1, bs2]
    return output


def dot_attention_with_vis(q, v, v_mask=None, dropout=None):
    # q [bs1, dim]
    # v [bs2, n_cluster, dim]
    # output [bs1, bs2]
    attention_logits = torch.matmul(v, q.T)  # [bs2, n_cluster, bs1]
    attention_weights = F.softmax(attention_logits, dim=1)  # [bs2, n_cluster, bs1]
    final_v = torch.einsum('jki,jkl->ijl', [attention_weights, v])  # [bs1, bs2, dim]
    output = torch.sum(final_v * q.unsqueeze(1), dim=-1)  # [bs1, bs2]

    output_reward_1 = attention_weights.permute((-1, 0, 1))  # [bs1, bs2, n_cluster]
    output_reward_2 = (attention_weights * (1 - attention_weights) * attention_logits).permute((-1, 0, 1))
    # other_cluster_sum_v
    n_cluster = v.size(1)
    other_cluster_sum_v = final_v.unsqueeze(2).expand(-1, -1, n_cluster, -1)  # [bs1, bs2, n_cluster, dim]
    self_cluster_sum_v = v.unsqueeze(-2) * attention_weights.unsqueeze(-1)  # [bs2, n_cluster, bs1, dim]
    self_cluster_sum_v = self_cluster_sum_v.permute((2, 0, 1, 3))  # [bs1, bs2, n_cluster, dim]
    other_cluster_sum_v = (other_cluster_sum_v - self_cluster_sum_v) * q.unsqueeze(1).unsqueeze(1)  # [bs1, bs2, n_cluster, dim]
    other_cluster_sum_v = torch.sum(other_cluster_sum_v, dim=-1)  # [bs1, bs2, n_cluster]
    output_reward_3 = - attention_weights.permute((-1, 0, 1)) * other_cluster_sum_v  # [bs1, bs2, n_cluster]

    return output, output_reward_1, output_reward_2, output_reward_3


class ClusterBertEncoder(BiEncoder):

    def __init__(self, question_model, ctx_model, fix_q_encoder=False,
                 fix_ctx_encoder=False, n_cluster=4, max_kmeans_iter=50, random_select=True,
                 use_cls=False, use_roberta=False):
        super(ClusterBertEncoder, self).__init__(question_model, ctx_model, fix_q_encoder, fix_ctx_encoder)
        self.n_cluster = n_cluster
        self.max_kmeans_iter = max_kmeans_iter
        self.random_select = random_select
        self.use_cls = use_cls
        self.use_roberta = use_roberta
        print(f'ClusterBertEncoder use_cls: {self.use_cls}')

    def forward(self, question_ids: T, question_segments: T, question_attn_mask: T, context_ids: T, ctx_segments: T,
                ctx_attn_mask: T) -> Tuple[T, T]:

        _q_seq, q_pooled_out, _q_hidden = self.get_representation(self.question_model, question_ids, question_segments,
                                                                  question_attn_mask, self.fix_q_encoder)

        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(self.ctx_model, context_ids, ctx_segments,
                                                                        ctx_attn_mask, self.fix_ctx_encoder)

        if _q_seq is not None:
            q_mean_out = _average_sequence_embeddings(_q_seq, question_attn_mask)
        else:
            q_mean_out = None
        if self.use_roberta:
            q_mean_out = q_pooled_out

        _, clustered_doc_vecs = lloyd(_ctx_seq, ctx_attn_mask, self.n_cluster, random_select=self.random_select)

        if self.use_cls:
            clustered_doc_vecs = torch.cat([ctx_pooled_out.unsqueeze(1), clustered_doc_vecs], dim=1)

        return q_mean_out, clustered_doc_vecs


class FirstBertEncoder(BiEncoder):

    def __init__(self, question_model, ctx_model, fix_q_encoder=False,
                 fix_ctx_encoder=False, n_cluster=4, max_kmeans_iter=50, random_select=True,
                 use_cls=False, use_roberta=False):
        super(FirstBertEncoder, self).__init__(question_model, ctx_model, fix_q_encoder, fix_ctx_encoder)
        self.n_cluster = n_cluster
        self.max_kmeans_iter = max_kmeans_iter
        self.random_select = random_select
        self.use_cls = use_cls
        self.use_roberta = use_roberta

    def forward(self, question_ids: T, question_segments: T, question_attn_mask: T, context_ids: T, ctx_segments: T,
                ctx_attn_mask: T) -> Tuple[T, T]:

        _q_seq, q_pooled_out, _q_hidden = self.get_representation(self.question_model, question_ids, question_segments,
                                                                  question_attn_mask, self.fix_q_encoder)

        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(self.ctx_model, context_ids, ctx_segments,
                                                                        ctx_attn_mask, self.fix_ctx_encoder)
        if _q_seq is not None:
            q_mean_out = _average_sequence_embeddings(_q_seq, question_attn_mask)
        else:
            q_mean_out = None
        if self.use_roberta:
            q_mean_out = q_pooled_out

        doc_vecs = _ctx_seq[:, 1:1+self.n_cluster, :]

        return q_mean_out, doc_vecs

class ClusterNllLoss(BiEncoderNllLoss):

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = ClusterNllLoss.get_similarity_function()
        assert len(q_vector.shape) == 2 and len(ctx_vectors.shape) == 3, print(q_vector.shape, ctx_vectors.shape)
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_attention


class ClusterNllLossVis(BiEncoderNllLoss):

    def calc(self, q_vectors: T, ctx_vectors: T, positive_idx_per_question: list,
             hard_negative_idx_per_question: list = None, return_scores=False) -> Tuple[T, int]:
        scores, reward_1, reward_2, reward_3 = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(softmax_scores, torch.tensor(positive_idx_per_question).to(softmax_scores.device),
                          reduction='mean')

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        if return_scores:
            return loss, correct_predictions_count, softmax_scores, (reward_1, reward_2, reward_3)
        else:
            return loss, correct_predictions_count, (reward_1, reward_2, reward_3)

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T):
        f = ClusterNllLossVis.get_similarity_function()
        assert len(q_vector.shape) == 2 and len(ctx_vectors.shape) == 3, print(q_vector.shape, ctx_vectors.shape)
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_attention_with_vis

