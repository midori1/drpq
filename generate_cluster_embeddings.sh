NUM_SHARD=16
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