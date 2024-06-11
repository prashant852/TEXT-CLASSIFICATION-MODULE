python train.py \
--train_csv "/home/TEXT-CLASSIFICATION/notebooks/data/comp/train.csv" \
--label2id "/home/TEXT-CLASSIFICATION/config/label2id.csv" \
--fold_file "/home/TEXT-CLASSIFICATION/data/PERSUADE-LB-EVENTRATE-SEED-8.csv" \
--base_model "/home/TEXT-CLASSIFICATION/notebooks/data/datasets" \
--base_tokenizer "microsoft/deberta-v3-large" \
--num_labels 6 \
--fold_strategy "oof-csv" \
--text_col "full_text" \
--target_col "score" \
--id_col "essay_id" \
--train_max_len 1792 \
--metric "qwk" \
--external_data_strategy "None" \
--newline_token_strategy "pipe" \
--trainer "default" \
--amp "True" \
--greater_is_better "True" \
--epochs 2 \
--batch_size 1 \
--eval_batch_size 1 \
--grad_acc_steps 8 \
--learning_rate 3.5e-5 \
--lr_scheduler "linear" \
--warmup_ratio 0.1 \
--freeze_layers -1 \
--freeze_embeddings "False" \
--reinit_layers -1 \
--max_grad_norm 1.0 \
--weight_decay 0.01 \
--label_smoothing_factor 0.1 \
--eval_delay 1000 \
--eval_steps 100 \
--report_to "wandb" \
--experiment "EX040-SEED-8" 