CKPT_FILE=checkpoint trained by "train_cluster_encoder.py"
MODEL_PATH=untrained checkpoint path (initial huggingface checkpoint)
DATA_DIR=your data dir...
for i in {0..15}
do
python dense_cluster_retriever.py \
    --model_file CKPT_FILE \
    --query_file $DATA_DIR/msmarco-test2019-queries.tsv \
    --qrels_file $DATA_DIR/msmarco-docdev-qrels.tsv \
    --encoded_ctx_file "output files of 'generate_cluster_embeddings.py'" \
    --out_file "output file name" \
    --expand_factor 10 \
    --encoder_model_type hf_roberta \
    --init_from_path \
    --model_path $MODEL_PATH \
    --config_path $MODEL_PATH/config.json \
    --vocab_path $MODEL_PATH \
    --n-docs 100 \
    --use_cls \
    --index_buffer 20000 \
    --validation_workers 32 \
    --n_cluster 4 \
    --no_knn \
    --batch_size 64
done