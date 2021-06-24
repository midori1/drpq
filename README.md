# Dense Retrieval via Pseudo Query embeddings 

Source code for the following paper, 
Hongyin Tang, Xingwu Sun, Beihong Jin, Jingang Wang, Fuzheng Zhang, Wei Wu [Improving Document Representations by Generating Pseudo Query Embeddings for Dense Retrieval](https://arxiv.org/abs/2105.03599).


## Train

run `train_cluster_encoder.sh`

```
DATADIR=directory of data...
PSGS_FILE=$DATADIR/msmarco-docstrain.tsv
DEV_PSGS_FILE=$DATADIR/msmarco-docdev.tsv
TRAIN_QUERIES_FILE=$DATADIR/msmarco-doctrain-queries.tsv
TRAIN_QRELS_FILE=$DATADIR/msmarco-doctrain-qrels.tsv
TRAIN_TREC_FILE=$DATADIR/msmarco-doctrain-top100
DEV_QUERIES_FILE=$DATADIR/msmarco-docdev-queries.tsv
DEV_QRELS_FILE=$DATADIR/msmarco-docdev-qrels.tsv
DEV_TREC_FILE=$DATADIR/msmarco-docdev-top100
MODEL_NAME=your model name...
MODEL_PATH=your model path of Bert, Roberta, etc. ...
python -m torch.distributed.launch \
    --nproc_per_node=4 train_cluster_encoder.py \
    --max_grad_norm 2.0 \
    --encoder_model_type hf_roberta \
    --init_from_path \
    --model_path $MODEL_PATH \
    --config_path $MODEL_PATH/config.json \
    --vocab_path $MODEL_PATH \
    --global_loss_buf_sz 100000000 \
    --seed 12345 \
    --clustering \
    --n_cluster 4 \
    --projection_dim 768 \
    --warmup_steps 5000 \
    --use_cls \
    --sequence_length 512 \
    --query_sequence_length 48 \
    --warmup_steps 1000 \
    --batch_size 4 \
    --train_hard_negatives 4 \
    --dev_hard_negatives 10 \
    --do_lower_case \
    --random_init_cluster False \
    --use_dict_input \
    --dev_psgs_file $DEV_PSGS_FILE \
    --psgs_file $PSGS_FILE \
    --train_queries_file $TRAIN_QUERIES_FILE \
    --train_qrels_file $TRAIN_QRELS_FILE \
    --train_trec_file $TRAIN_TREC_FILE \
    --dev_queries_file $DEV_QUERIES_FILE \
    --dev_qrels_file $DEV_QRELS_FILE \
    --dev_trec_file $DEV_TREC_FILE \
    --log_batch_step 100 \
    --output_dir ./output/$MODEL_NAME \
    --log_dir ./output/log/$MODEL_NAME \
    --learning_rate 2e-06 \
    --num_train_epochs 10 \
    --dev_batch_size 8 \
    --eval_per_epoch 2 \
    --val_av_rank_start_epoch 100
```

## Generate document embeddings
Since there are large number of documents, we split the documents into multiple shards to prevent memory issue.

run `generate_cluster_embeddings.sh`
```
NUM_SHARD=16  # run by shards
DATA_DIR=your data dir...
for i in {0..15}
do
SHARD_ID=$i
CTX_FILE=${DATA_DIR}/msmarco-docs.tsv
MODEL_FILE=your model file...
OUTPUT_DIR=your outout dir...
python generate_cluster_embeddings.py \
    --model_file $MODEL_FILE \
    --ctx_file $CTX_FILE \
    --shard_id $SHARD_ID \
    --num_shards $NUM_SHARD \
    --encoder_model_type hf_roberta \
    --batch_size 256 \
    --n_cluster 4 \
    --marco \
    --out_file $OUTPUT_DIR
done
```

## Retrieve documents by the generated document embeddings
load the document embeddings and then retrieve the documents. 
Finally, for each shard, we get top-k retrieved documents for the queries. 

run `dense_cluster_retriever.sh`
```
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
```

## Merge the retrieved documents in each shard
run `merge_retrieve_results.sh`
```
DATA_DIR=your data dir...
python merge_retrieve_results_marco.py \
    --doc_score_files "output files of 'dense_cluster_retriever.py" \
    --query_file ${DATA_DIR}msmarco-test2019-queries.tsv \
    --qrels_file ${DATA_DIR}/2019qrels-docs.txt \
    --output_file "your output file" \
    --trec \
    --n-docs 100
```