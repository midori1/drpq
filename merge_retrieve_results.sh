DATA_DIR=your data dir...
python merge_retrieve_results_marco.py \
    --doc_score_files "output files of 'dense_cluster_retriever.py" \
    --query_file ${DATA_DIR}msmarco-test2019-queries.tsv \
    --qrels_file ${DATA_DIR}/2019qrels-docs.txt \
    --output_file "your output file" \
    --trec \
    --n-docs 100