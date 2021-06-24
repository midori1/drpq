import argparse
import glob
import pickle
import heapq
from dense_cluster_retriever import parse_qrels_file, parse_question_file, parse_trec_file
from ms_marco_eval import compute_metrics
import pytrec_eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--doc_score_files', required=True, type=str, default=None,
                        help="glob path containing doc and scores")
    parser.add_argument('--query_file', required=True, type=str, default=None,
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--qrels_file', required=True, type=str, default=None,
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--output_file', required=True, type=str, default=None)
    parser.add_argument("--trec", action='store_true', help='whether to receive trec format as the question file')

    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'],
                        help="Answer matching logic type")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--n-docs', type=int, default=200, help="Amount of top docs to return")

    args = parser.parse_args()

    input_paths = glob.glob(args.doc_score_files)

    # get questions & answers
    questions = []
    question_ids = []
    question_answers = []
    qid2docids = {}

    if not args.trec:
        for item in parse_qrels_file(args.qrels_file):
            qid, docid = item
            if qid in qid2docids:
                qid2docids[qid].append(docid)
            else:
                qid2docids[qid] = [docid]
    else:
        for item in parse_trec_file(args.qrels_file):
            qid, docid, rel = item
            if qid in qid2docids:
                qid2docids[qid][docid] = int(rel)
            else:
                qid2docids[qid] = {docid: int(rel)}

    for ds_item in parse_question_file(args.query_file):
        qid, question = ds_item
        questions.append(question)
        question_ids.append(qid)

    all_doc_scores = []
    for input_path in input_paths:
        print("reading doc and scores from {}".format(input_path))
        data = pickle.load(open(input_path, 'rb'))
        # unzip data
        data = [list(zip(*ex)) for ex in data]
        if len(all_doc_scores) == 0:
            all_doc_scores = data
        else:
            assert len(all_doc_scores) == len(data)
            for i in range(len(all_doc_scores)):
                all_doc_scores[i].extend(data[i])

    # heapify all_doc_scores
    print("merging doc scores...")
    for i in range(len(all_doc_scores)):
        # get topk docs
        query_top_doc_scores = heapq.nlargest(args.n_docs, all_doc_scores[i], key=lambda x: x[1])
        # sort top docs
        all_doc_scores[i] = sorted(query_top_doc_scores, key=lambda x: x[1], reverse=True)
        # unzip
        all_doc_scores[i] = list(zip(*all_doc_scores[i]))

    assert len(all_doc_scores) == len(questions), print(len(all_doc_scores), len(questions))
    # for eval
    eval_cand_documents = {}
    for qid, doc_scores in zip(question_ids, all_doc_scores):
        if qid not in eval_cand_documents:
            eval_cand_documents[qid] = []
        doc_ids, scores = doc_scores
        for rank, doc_id in enumerate(doc_ids):
            eval_cand_documents[qid].append((doc_id, rank+1))

    # save results
    with open(args.output_file, 'w') as outfile:
        for qid, scores in eval_cand_documents.items():
            for doc_id, rank in scores:
                outfile.write(f'{qid}\t{doc_id}\t{rank}\n')
    #
    if args.trec:
        evaluator = pytrec_eval.RelevanceEvaluator(
            qid2docids, {'map_cut', 'ndcg_cut'})
        eval_query_cnt = 0
        eval_cand_documents = {qid: {docid: -rank for docid, rank in doclist} for qid, doclist in eval_cand_documents.items()}
        result = evaluator.evaluate(eval_cand_documents)
        ndcg = 0

        for k in result.keys():
            eval_query_cnt += 1
            ndcg += result[k]["ndcg_cut_10"]

        final_ndcg = ndcg / eval_query_cnt
        print("NDCG@10:" + str(final_ndcg))
    else:
        metrics = compute_metrics(qid2docids, eval_cand_documents, set())
        print('#####################')
        for metric in sorted(metrics):
            print('{}: {}'.format(metric, metrics[metric]))
        print('#####################')




