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