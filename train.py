import os
import logging
import warnings

log_file_path = './logs/warnings.log'
log_format = '%(asctime)s - %(levelname)s - %(message)s'

if not os.path.exists('./logs'):
    os.mkdir('./logs')

logging.basicConfig(filename=log_file_path, level=logging.WARNING, format=log_format)
warnings.filterwarnings('default')
logging.captureWarnings(True)

import argparse
from training import training_pipeline

if __name__ == '__main__':
    '''
    IMPORTANT NOTES:
    1. Pretraining model on Pseudo Labels and validate on default data, set pretraining_mode to 'True' along with external_data_strategy to 'append'
    2. LLRD is preferred with warmup_ratio set to 0.0, freeze_layers to -1, and freeze_embeddings to 'False' (recommended no reinit)
    '''


    parser = argparse.ArgumentParser()

    #Path arguments
    parser.add_argument('--train_csv', type=str, default='./data/train.csv', help="Path for train.csv")
    parser.add_argument('--label2id', type=str, default='./config/label2id.csv', help="Path of target labels to their categorical encoding (label2id) csv file")
    parser.add_argument('--external_data_path', default='./external_data.csv', help='Path of external data')
    parser.add_argument('--logging_dir', type=str, default='./logs/model', help="Logging directory of HuggingFace Trainer")
    parser.add_argument('--fold_file', type=str, default="./data/train-99.csv", help="Path of validation data identifier in a csv (used in fold_strategy)")
    parser.add_argument('--output_dir', type=str, default='./output', help='Path of output directory')
    
    #Huggingface model arguments
    parser.add_argument('--base_model', type=str, default='./models/deberta-v3-base', help="Base model path or Huggingface path")
    parser.add_argument('--base_tokenizer', type=str, default='./models/deberta-v3-base', help="Base model tokenizer path or Huggingface path")
    parser.add_argument('--num_labels', type=int, default=6, help='Number of labels in target')
    parser.add_argument('--use_regression', type=str, default='False', help="Default task is classification, set it to true for regression")
    
    #Data arguments
    parser.add_argument('--pretraining_mode', type=str, default="False", help='Set to True for training model on pseudo label and validate on default data')
    parser.add_argument('--fold_strategy', type=str, default='oof-csv', help="Data splitting strategy from: 'random', 'kfold', 'topic', 'oof-csv'")
    parser.add_argument('--fold_num', type=int, default=0, help='Use -1 to run on all folds, or specify kfold number (useful only when fold_strategy is set to kfold)')
    parser.add_argument('--text_col', type=str, default="full_text", help='Input text column in default data')
    parser.add_argument('--target_col', type=str, default="score", help='Target label column in default data')
    parser.add_argument('--id_col', type=str, default="essay_id", help='Essay unique ID column in default data')
    parser.add_argument('--ext_text_col', type=str, default="full_text", help='Input text column in external data')
    parser.add_argument('--ext_target_col', type=str, default="score", help='Target label column in external data')
    parser.add_argument('--ext_id_col', type=str, default="essay_id", help='Essay unique ID column in external data')
    parser.add_argument('--append_rubrik', type = str, default='None', help="Prepend text with the grading rubrik")


    #Modeling arguments
    parser.add_argument('--train_max_len', type=int, default=1792, help='Training maximum sequence length')
    parser.add_argument('--metric', type=str, default='qwk', help="Evaluation metric to use either from: 'accuracy' or 'qwk'")
    parser.add_argument('--optimize_threshold', type=str, default='False', help="Set to True to optimize threshold for regression")
    
    parser.add_argument('--sigma_reparam', default='False', type=str, help="Convert model to SigamReparam variant")
    parser.add_argument('--external_data_strategy', default='None', type=str, help="Strategy to include external data: 'None', 'append', 'only-external'")
    parser.add_argument('--newline_token_strategy', default='True', type=str, help="Handling newline character by: 'add-to-tokenizer' or 'pipe'")

    #Training arguments
    parser.add_argument('--trainer', type=str, default='default', help="HuggingFace Trainer class: 'default', 'wkl_trainer' or 'oll_trainer'")
    parser.add_argument('--amp', default='True', type=str, help='Set True to allow fp16 training')
    parser.add_argument('--greater_is_better', default='True', type=str, help='Strategy to save best model based on metric')
    
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=2, help='Batch size for evaluation')
    parser.add_argument('--grad_acc_steps', type=int, default=16, help='Number of gradient accumulation steps for training')

    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for training')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help="Learning rate scheduler type: 'linear','cosine', or 'constant'")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for training')
    

    parser.add_argument('--freeze_layers', type=int, default=6, help='Number of layers to freeze')
    parser.add_argument('--freeze_embeddings', default='False', type=str, help='Embeddings to freeze')
    parser.add_argument('--reinit_layers', type=int, default=-1, help='Number of last layers (closer to output) to reinitialize')
    parser.add_argument('--llrd', type=float, default=-1.0, help='Value of layerwise_learning_rate_decay (Note: use warmup_ratio to 0, no freezing and no reinit)')

    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm (for gradient clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay to apply (if not zero) to all layers')
    parser.add_argument('--label_smoothing_factor', type=float, default=0.0, help='Label smoothing factor to use')

    parser.add_argument('--save_total_limit', type=int, default=1, help='Number of checkpoints to save')
    parser.add_argument('--eval_delay', type=int, default=1000, help='Steps to delay before first evaluation')
    parser.add_argument('--eval_steps', type=int, default=100, help='Evaluation steps')
    parser.add_argument('--report_to', type=str, default='none', help="report_to argument 'none', 'wandb' or 'tensorboard'")
    parser.add_argument('--experiment', type=str, default='EX001', help="Experiment name for reporting")
    
    args = parser.parse_args()
    print("-"*100)
    for arg in vars(args):
        print(arg, ':', getattr(args, arg), type(getattr(args, arg)))
    print("-"*100)

    training_pipeline(args)