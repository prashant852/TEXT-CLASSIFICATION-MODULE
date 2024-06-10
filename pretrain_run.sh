python train.py \
--train_csv "/home/TEXT-CLASSIFICATION/notebooks/data/comp/train.csv" \
--label2id "/home/TEXT-CLASSIFICATION/config/label2id.csv" \
--external_data_path "/home/TEXT-CLASSIFICATION/notebooks/data/aes2-llama-3-pseudo-labels/llama-3-pseudo-labels.csv" \
--base_model "microsoft/deberta-v3-base" \
--base_tokenizer "microsoft/deberta-v3-base" \
--num_labels 6 \
--pretraining_mode "True" \
--text_col "full_text" \
--target_col "score" \
--id_col "essay_id" \
--ext_text_col "full_text" \
--ext_target_col "score" \
--ext_id_col "essay_id" \
--train_max_len 1792 \
--metric "qwk" \
--external_data_strategy "append" \
--newline_token_strategy "pipe" \
--trainer "default" \
--amp "True" \
--greater_is_better "True" \
--epochs 2 \
--batch_size 1 \
--eval_batch_size 1 \
--grad_acc_steps 8 \
--learning_rate 5e-5 \
--lr_scheduler "linear" \
--warmup_ratio 0.0 \
--freeze_layers -1 \
--freeze_embeddings "False" \
--reinit_layers -1 \
--llrd 0.9 \
--max_grad_norm 1.0 \
--weight_decay 0.01 \
--label_smoothing_factor 0.1 \
--eval_delay 1000 \
--eval_steps 500 \
--report_to "wandb" \
--experiment "Pretraining-001" 