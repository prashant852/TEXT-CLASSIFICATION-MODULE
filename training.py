import os
import argparse
import gc
import torch
import pandas as pd
import numpy as np
import wandb

from functools import partial
from tokenizers import AddedToken
from transformers import (
    AutoConfig,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    AdamW
)

from model_utils import freeze_and_modify_weights
from preprocess_data import preprocess_data
from metrics import get_metric
from trainer import *

os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
os.environ["WANDB_WATCH"] = "all"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training_pipeline(_args):
    if _args.fold_num == -1:
        #Train all folds
        folds = [0,1,2,3]
    else:
        #Train single fold
        folds = [_args.fold_num]
    
    #Load target labels to their categorical encoding, and reverse mapping
    label2id = pd.read_csv(_args.label2id).set_index('Label')['ID'].to_dict()
    id2label = {v:k for k,v in label2id.items()}
    print("label2id:",label2id)
    
    print(f"Total folds for training: {len(folds)}")
    for fold in folds:
        print(f"--------------------------------------------CURRENT FOLD: {fold}--------------------------------------------")

        #Loading tokenizer
        print("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(_args.base_tokenizer)

        if _args.newline_token_strategy == "add-to-tokenizer":
            #Adding \n to tokenizer vocabulary, since deberta ignore this
            print("Adding NEWLINE character to tokenizer vocabulary")
            tokenizer.add_tokens([AddedToken("\n", normalized=False)])
        
        #Loading data
        train, valid = preprocess_data(_args, fold, tokenizer)

        #Loading model
        if _args.use_regression == "True":
            print("Loading model for regression")
            reg_config = AutoConfig.from_pretrained(_args.base_model)
            reg_config.attention_probs_dropout_prob = 0.0 
            reg_config.hidden_dropout_prob = 0.0 
            reg_config.num_labels = 1
            model = AutoModelForSequenceClassification.from_pretrained(
                _args.base_model,
                config = reg_config,
            ).to(device)
        else:
            print("Loading model for classification")
            model = AutoModelForSequenceClassification.from_pretrained(
                _args.base_model,
                num_labels = _args.num_labels,
                id2label=id2label,
                label2id=label2id,
            ).to(device)

        if _args.newline_token_strategy == "add-to-tokenizer":
            print("Resizing model vocabulary")
            model.resize_token_embeddings(len(tokenizer))
        
        #Reinitialization and freezing model layers
        model = freeze_and_modify_weights(model, _args)

        #LLRD (Layer-Wise Learning Rate)
        if _args.llrd > 0:
            model_config = AutoConfig.from_pretrained(_args.base_model)
            _model_type = model_config.model_type
            adam_epsilon = 1e-6
            use_bertadam = False

            def get_optimizer_grouped_parameters(
                model, model_type, 
                learning_rate, weight_decay, 
                layerwise_learning_rate_decay
            ):
                no_decay = ["bias", "LayerNorm.weight"]
                # initialize lr for task specific layer
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
                        "weight_decay": 0.0,
                        "lr": learning_rate,
                    },
                ]
                # initialize lrs for every layer
                num_layers = model.config.num_hidden_layers
                layers = [model.deberta.embeddings] + list(model.deberta.encoder.layer)
                layers.reverse()
                lr = learning_rate
                for layer in layers:
                    lr *= layerwise_learning_rate_decay
                    optimizer_grouped_parameters += [
                        {
                            "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                            "weight_decay": weight_decay,
                            "lr": lr,
                        },
                        {
                            "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                            "weight_decay": 0.0,
                            "lr": lr,
                        },
                    ]
                return optimizer_grouped_parameters
            
            grouped_optimizer_params = get_optimizer_grouped_parameters(
                model, _model_type, 
                _args.learning_rate, _args.weight_decay, 
                _args.llrd
            )
            optimizer = AdamW(
                grouped_optimizer_params,
                lr=_args.learning_rate,
                eps=adam_epsilon,
                correct_bias=not use_bertadam
            )
            batches_per_epoch = len(train)//_args.batch_size + 1
            max_steps = _args.epochs*batches_per_epoch
            print(f"Initializing LLRD scheduler with {max_steps} training steps")
            if _args.lr_scheduler == 'linear':
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=max_steps
                )
            elif _args.lr_scheduler == 'cosine':
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=max_steps
                )
            else:
                raise ValueError("Unknown lr_scheduler for LLRD")

        print("Loading training arguments")
        #Training parameters
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=16)
        compute_metrics = get_metric(_args)

        args = TrainingArguments(
            output_dir=os.path.join(_args.output_dir, f"fold{fold}"),
            fp16=_args.amp,
            learning_rate=_args.learning_rate,
            num_train_epochs=_args.epochs,
            per_device_train_batch_size=_args.batch_size,
            per_device_eval_batch_size=_args.eval_batch_size,
            gradient_accumulation_steps=_args.grad_acc_steps,
            report_to=_args.report_to,
            run_name = f"{_args.experiment}-FOLD{str(fold)}",
            eval_steps=_args.eval_steps,
            save_steps=_args.eval_steps,
            logging_steps=_args.eval_steps,
            logging_dir = _args.logging_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=_args.save_total_limit,
            metric_for_best_model=f"eval_{str(_args.metric)}",
            greater_is_better=_args.greater_is_better,
            load_best_model_at_end=True,
            overwrite_output_dir=True,
            lr_scheduler_type=_args.lr_scheduler,
            warmup_ratio=_args.warmup_ratio,
            weight_decay=_args.weight_decay,
            max_grad_norm  = _args.max_grad_norm,
            eval_delay = _args.eval_delay,
            label_smoothing_factor = _args.label_smoothing_factor,
        )

        if _args.trainer == "oll_trainer":
            if _args.llrd > 0:
                print("Trainer: OLL, LLRD: True")
                trainer = OLL2Trainer(
                        args=args,
                        model=model,
                        train_dataset=train,
                        eval_dataset=valid,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics,
                        data_collator=data_collator,
                        optimizers = (optimizer, scheduler)
                    )
            else:
                print("Trainer: OLL, LLRD: False")
                trainer = OLL2Trainer(
                    args=args,
                    model=model,
                    train_dataset=train,
                    eval_dataset=valid,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                    data_collator=data_collator,
                )
        elif _args.trainer == "wkl_trainer":
            if _args.llrd > 0:
                print("Trainer: WKL, LLRD: True")
                trainer = WKLTrainer(
                        args=args,
                        model=model,
                        train_dataset=train,
                        eval_dataset=valid,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics,
                        data_collator=data_collator,
                        optimizers = (optimizer, scheduler)
                    )
            else:
                print("Trainer: WKL, LLRD: False")
                trainer = WKLTrainer(
                    args=args,
                    model=model,
                    train_dataset=train,
                    eval_dataset=valid,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                    data_collator=data_collator,
                )
        else:
            if _args.llrd > 0:
                print("Trainer: Default, LLRD: True")
                trainer = Trainer(
                        args=args,
                        model=model,
                        train_dataset=train,
                        eval_dataset=valid,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics,
                        data_collator=data_collator,
                        optimizers = (optimizer, scheduler)
                    )
            else:
                print("Trainer: Default, LLRD: False")
                trainer = Trainer(
                    args=args,
                    model=model,
                    train_dataset=train,
                    eval_dataset=valid,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                    data_collator=data_collator,
                )
        
        print("Training started")
        #Train model
        trainer.train()

        #Clear cache
        print("Clearing cache")
        del model, trainer, args, data_collator, compute_metrics, train, valid
        torch.cuda.empty_cache()
        gc.collect()