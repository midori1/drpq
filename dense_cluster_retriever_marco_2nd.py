#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator

import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from dpr.data.qa_validation import calculate_matches
from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseClusterFlatIndexer

from dpr.models.cluster_models import _average_sequence_embeddings
import heapq
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer, index: DenseIndexer,
                 n_cluster, use_knn=True):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index
        self.use_knn = use_knn
        self.n_cluster = n_cluster

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [self.tensorizer.text_to_tensor(q) for q in
                                       questions[batch_start:batch_start + bsz]]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                all_out, pooled_out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                out = _average_sequence_embeddings(all_out, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info('Encoded queries %d', len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info('Total encoded queries tensor %s', query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def get_top_docs_knn(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        if self.use_knn:
            results = self.index.search_knn(query_vectors, top_docs)
        else:
            if isinstance(self.index, DenseClusterFlatIndexer):
                results = self.index.compute_without_knn(query_vectors, top_docs)
            else:
                raise TypeError('Only DenseClusterFlatIndexer supports not using knn')
        logger.info('index search time: %f sec.', time.time() - time0)
        return results

    def get_top_docs_without_knn(self, query_vectors: np.array, vector_files, top_docs: int, n_cluster,
                                 buffer_size, top1000_map, question_ids):
        # 单纯计算对每个query计算每个document的分数
        def _reshape(buffer):
            db_ids = ["{}_{}".format(t[0], i) for t in buffer for i in range(self.n_cluster)]
            vectors = [np.reshape(t[1], (self.n_cluster, -1)) for t in buffer]
            vectors = np.concatenate(vectors, axis=0)
            return db_ids, vectors

        def _iterate(vector_files, buffer_size):
            buffer = []
            start_time = time.time()
            total_iter = 0
            for i, item in enumerate(iterate_encoded_files(vector_files)):
                db_id, doc_vector = item
                buffer.append((db_id, doc_vector))
                if 0 < buffer_size == len(buffer):
                    db_ids, vectors = _reshape(buffer)
                    yield db_ids, vectors
                    total_iter += buffer_size
                    logger.info(f'vector iterated {total_iter}, used_time: {time.time() - start_time} sec.')
                    buffer = []
            if len(buffer) > 0:
                db_ids, vectors = _reshape(buffer)
                yield db_ids, vectors
                total_iter += len(db_ids) / n_cluster
            logger.info('Total data iterated %d', total_iter)
            logger.info('Data iterating completed.')

        all_doc_query_scores = []
        all_doc_ids = []
        # move to gpu
        query_vectors = torch.from_numpy(query_vectors).to(torch.device("cuda"))

        for db_ids, vectors in _iterate(vector_files, buffer_size):
            assert len(vectors) % self.n_cluster == 0, print(len(vectors))
            num_docs = len(vectors) // self.n_cluster
            doc_vectors = np.reshape(vectors, (num_docs, self.n_cluster, -1))

            for dbid in db_ids:
                docid = dbid[:dbid.find('_')]
                if len(all_doc_ids) == 0 or docid != all_doc_ids[-1]:  # 按照顺序增加，当前和前一个不同则加入
                    all_doc_ids.append(docid)

            # move to gpu
            doc_vectors = torch.from_numpy(doc_vectors).to(torch.device("cuda"))

            # mini-batch query vectors
            query_batch_size = 8
            num_queries = query_vectors.shape[0]
            # 向上取整
            query_batch_num = (num_queries + query_batch_size - 1) // query_batch_size

            cur_doc_query_scores = []  # 当前doc相对于所有query的得分
            for i in tqdm(range(query_batch_num), desc='Computing scores of queries'):
                batch_query_vectors = query_vectors[i * query_batch_size:(i + 1) * query_batch_size]

                # attn_logits = torch.matmul(doc_vectors, batch_query_vectors.T)  # [bs2, n_cluster, bs1]
                # attn_probs = torch.softmax(attn_logits, dim=1)
                # new_doc_vectors = torch.einsum('jki,jkl->ijl', [attn_probs, doc_vectors])  # [bs1, bs2, dim]
                # doc_scores = torch.sum(new_doc_vectors * batch_query_vectors.unsqueeze(1), dim=-1)  # [bs1, bs2]
                #
                attn_logits = torch.einsum('ik,jlk->ijl', batch_query_vectors, doc_vectors)
                attn_probs = torch.softmax(attn_logits, dim=2)
                new_doc_vectors = torch.einsum('ijl,jlk->ijk', attn_probs, doc_vectors)
                doc_scores = torch.einsum('ik,ijk->ij', batch_query_vectors, new_doc_vectors)

                cur_doc_query_scores.append(doc_scores.transpose(0, 1))
                del new_doc_vectors

            scores = torch.cat(cur_doc_query_scores, dim=1)
            # move to cpu
            all_doc_query_scores.append(scores.detach().cpu())

        # assert
        all_query_doc_scores = torch.cat(all_doc_query_scores, dim=0).transpose(0, 1)
        assert len(all_doc_ids) == all_query_doc_scores.size(1), print(len(all_doc_ids), all_query_doc_scores.size(1))

        del query_vectors
        del doc_vectors
        del attn_logits
        del attn_probs
        del scores
        torch.cuda.empty_cache()

        # 取topk的document
        if top_docs > all_query_doc_scores.shape[1]:
            top_docs = all_query_doc_scores.shape[1]

        # filter scores
        for qid, query_doc_scores in tqdm(zip(question_ids, all_query_doc_scores), desc='filtering top1000...'):
            select_bool = []
            for docid in all_doc_ids:
                if docid not in top1000_map[qid]:
                    select_bool.append(True)
                else:
                    select_bool.append(False)
            query_doc_scores.masked_fill_(torch.BoolTensor(select_bool), -10000.0)

            # for idx in range(len(query_doc_scores)):
            #     if all_doc_ids[idx] not in top1000_map[qid]:
            #         query_doc_scores[idx] = -10000.0

        topk_doc_scores, topk_doc_idx = torch.topk(all_query_doc_scores, k=top_docs, dim=1)
        del all_query_doc_scores
        topk_doc_scores = topk_doc_scores.numpy()
        topk_doc_idx = topk_doc_idx.numpy()

        # convert to external ids
        result_doc_ids = [[all_doc_ids[i] for i in docidxs] for docidxs in topk_doc_idx]
        result = [(result_doc_ids[i], topk_doc_scores[i]) for i in range(len(result_doc_ids))]
        return result


def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers


def parse_question_file(location):
    with open(location) as infile:
        for line in infile:
            qid, query = line.strip('\n').split('\t')
            yield qid, query


def parse_qrels_file(location):
    with open(location) as infile:
        for line in infile:
            qid, _, docid, _ = line.strip('\n').split('\t')
            yield qid, docid


def parse_trec_file(location):
    with open(location) as infile:
        for line in infile:
            qid, _, docid, rel = line.strip('\n').split(' ')
            yield qid, docid, rel


def validate(passages: Dict[object, Tuple[str, str]], answers: List[List[str]],
             result_ctx_ids: List[Tuple[List[object], List[float]]],
             workers_num: int, match_type: str) -> List[List[bool]]:
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return match_stats.questions_doc_hits


def save_results(passages: Dict[object, Tuple[str, str]], questions: List[str], answers: List[List[str]],
                 top_passages_and_scores: List[Tuple[List[object], List[float]]], per_question_hits: List[List[bool]],
                 out_file: str
                 ):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append({
            'question': q,
            'answers': q_answers,
            'ctxs': [
                {
                    'id': results_and_scores[0][c],
                    'title': docs[c][1],
                    'text': docs[c][0],
                    'score': scores[c],
                    'has_answer': hits[c],
                } for c in range(ctxs_num)
            ]
        })

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info('Saved results * scores  to %s', out_file)


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info('Reading file %s', file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')

    prefix_len = len('question_model.')
    question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                              key.startswith('question_model.')}
    model_to_load.load_state_dict(question_encoder_state)
    vector_size = model_to_load.get_out_size()
    logger.info('Encoder vector_size=%d', vector_size)

    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size, args.index_buffer)
    elif not args.no_knn:
        index = DenseClusterFlatIndexer(vector_size, args.n_cluster, args.index_buffer)

    if args.no_knn:
        retriever = DenseRetriever(encoder, args.batch_size, tensorizer, None, n_cluster=args.n_cluster, use_knn=False)
    else:
        retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index, n_cluster=args.n_cluster, use_knn=True)

    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)

    # get questions & truth document id
    question_ids = []
    questions = []
    qid2docids = {}

    for ds_item in parse_question_file(args.query_file):
        qid, question = ds_item
        questions.append(question)
        question_ids.append(qid)

    if args.trec:
        for item in parse_trec_file(args.qrels_file):
            qid, docid, rel = item
            if qid in qid2docids:
                qid2docids[qid][docid] = rel
            else:
                qid2docids[qid] = {docid: rel}
    else:
        for item in parse_qrels_file(args.qrels_file):
            qid, docid = item
            if qid in qid2docids:
                qid2docids[qid].add(docid)
            else:
                qid2docids[qid] = {docid}

    top1000_map = {}
    if args.top1000:
        with open(args.top1000) as f:
            for line in f:
                qid, docid, rank = line.strip('\n').split('\t')
                # if int(rank) > 100:
                #     continue
                if qid not in top1000_map:
                    top1000_map[qid] = {docid}
                else:
                    top1000_map[qid].add(docid)
        print(f'{args.top1000} has been loaded.')

    questions_tensor = retriever.generate_question_vectors(questions)

    all_top_ids_and_scores = []

    # 读取的每个文件对应一个新的index，而不是全部放进一个index中
    for input_path in input_paths:
        logger.info('Reading passages data from file: %s', input_path)

        if args.no_knn:
            top_ids_and_scores = retriever.get_top_docs_without_knn(query_vectors=questions_tensor.numpy(),
                                                                    vector_files=[input_path],
                                                                    top_docs=args.n_docs,
                                                                    n_cluster=args.n_cluster,
                                                                    buffer_size=args.index_buffer,
                                                                    top1000_map=top1000_map,
                                                                    question_ids=question_ids)
        else:
            retriever.index.index_data([input_path])
            # get top k results
            top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs * args.expand_factor)

        # zip top_ids_and_scores
        top_ids_and_scores = [list(zip(ex[0], ex[1])) for ex in top_ids_and_scores]

        if len(all_top_ids_and_scores) == 0:
            all_top_ids_and_scores = top_ids_and_scores
        else:
            assert len(all_top_ids_and_scores) == len(top_ids_and_scores)
            for i, ex in enumerate(top_ids_and_scores):
                all_top_ids_and_scores[i].extend(ex)

        # retriever.index.cluster_index.reset()
        # del retriever.index

        # retriever.index = DenseClusterFlatIndexer(vector_size, args.n_cluster, args.index_buffer)


    # heapify all_top_ids_and_scores
    for i in range(len(all_top_ids_and_scores)):
        # get topk docs
        query_top_doc_scores = heapq.nlargest(args.n_docs, all_top_ids_and_scores[i], key=lambda x: x[1])
        # sort top docs
        all_top_ids_and_scores[i] = sorted(query_top_doc_scores, key=lambda x: x[1], reverse=True)
        # unzip
        all_top_ids_and_scores[i] = zip(*all_top_ids_and_scores[i])

    # save all_top_ids_and_scores
    with open(input_paths[0] + '.scores', 'wb') as f:
        pickle.dump(all_top_ids_and_scores, f)

    # all_passages = load_passages(args.ctx_file)
    #
    # if len(all_passages) == 0:
    #     raise RuntimeError('No passages data found. Please specify ctx_file param properly.')
    #
    # questions_doc_hits = validate(all_passages, question_answers, all_top_ids_and_scores, args.validation_workers,
    #                               args.match)
    #
    # if args.out_file:
    #     save_results(all_passages, questions, question_answers, all_top_ids_and_scores, questions_doc_hits, args.out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--query_file', required=True, type=str, default=None,
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--qrels_file', required=True, type=str, default=None,
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")

    parser.add_argument('--encoded_ctx_file', type=str, default=None,
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--out_file', type=str, default=None,
                        help='output .json file path to write results to ')
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'],
                        help="Answer matching logic type")
    parser.add_argument('--n-docs', type=int, default=200, help="Amount of top docs to return")
    parser.add_argument('--expand_factor', type=int, default=1, help="multiply factor for each cluster in cluster "
                                                                     "retrieving")

    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')
    parser.add_argument("--trec", action='store_true', help='whether to receive trec format as the question file')
    parser.add_argument("--no_knn", action='store_true', help='whether to not to use knn searching')
    parser.add_argument("--top1000", type=str, default=None)


    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)
    print_args(args)
    main(args)
