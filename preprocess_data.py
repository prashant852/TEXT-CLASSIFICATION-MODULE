import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from metadata import text_to_append

def read_data(_args):
    comp_data = pd.read_csv(_args.train_csv)
    if _args.external_data_strategy == "None":
        #Don't include external data in training
        print("Default data loaded")
        return comp_data
    elif _args.external_data_strategy == "append":
        #Combine external data with competition data, useful for pretraining
        print("Pseudo label data mixed with default data (Pretraining)")
        ext_data = pd.read_csv(_args.external_data_path)
        ext_data['data_source'] = 'pseudo-label'
        comp_data['data_source'] = 'default'

        #Data specific assumption
        ext_data = ext_data[ext_data['is_match']==False].reset_index(drop=True)
        
        ext_data.rename(columns={_args.ext_text_col : _args.text_col, _args.ext_target_col : _args.target_col, _args.ext_id_col : _args.id_col}, inplace=True)
        return pd.concat([comp_data, ext_data[comp_data.columns]])
    elif _args.external_data_strategy == "only-external":
        #Return external data only
        print("Pseudo label data loaded (Pretraining)")
        ext_data = pd.read_csv(_args.external_data_path)

        #Data specific assumption
        ext_data = ext_data[ext_data['is_match']==False].reset_index(drop=True)
        ext_data.rename(columns={_args.ext_text_col : _args.text_col, _args.ext_target_col : _args.target_col, _args.ext_id_col : _args.id_col}, inplace=True)
        return ext_data[comp_data.columns]
    else:
        raise ValueError(f"Error: external_data_strategy {_args.external_data_strategy} not defined!")

def split_data(data, _args, fold=0):
    if _args.pretraining_mode == 'True':
        #Split based on data source
        print("Splitting strategy: data_source")
        train = data[data['data_source']=='pseudo-label'].reset_index(drop=True)
        valid = data[data['data_source']=='default'].reset_index(drop=True)
    else:
        if _args.fold_strategy == "random":
            #Random train-test split
            print("Splitting strategy: random")
            train, valid = train_test_split(data, test_size = 0.2, random_state=42, stratify=data[_args.target_col])
        elif _args.fold_strategy == "kfold":
            #Split based on fold
            print("Splitting strategy: kfold")
            fold_data = pd.read_csv(_args.fold_file)
            train = fold_data[fold_data['kfold']!=fold].reset_index(drop=True)
            valid = fold_data[fold_data['kfold']==fold].reset_index(drop=True)
        elif _args.fold_strategy =="topic":
            #Split based on column topic
            print("Splitting strategy: topic")
            data = pd.read_csv(_args.fold_file)
            train = data[~data['topic'].isin([1, 12, 13, 2, -1])]
            valid = data[data['topic'].isin([1, 12, 13, 2, -1])]
        elif _args.fold_strategy =="oof-csv":
            #Split based on specified OOF csv file
            print("Splitting strategy: oof-csv")
            oof_ids = pd.read_csv(_args.fold_file)[_args.id_col].tolist()
            train = data[~data[_args.id_col].isin(oof_ids)]
            valid = data[data[_args.id_col].isin(oof_ids)]
        else:
            raise ValueError(f"Error: fold_strategy {_args.fold_strategy} not defined!")
    
    train = train[[_args.text_col, _args.target_col, _args.id_col]].reset_index(drop=True)
    valid = valid[[_args.text_col, _args.target_col, _args.id_col]].reset_index(drop=True)
    print("Modeling data shape:",train.shape, valid.shape)

    #Inference optimization for batch processing
    valid['text_len'] = valid[_args.text_col].apply(len)
    valid = valid.sort_values(by=['text_len']).reset_index(drop=True)
    valid.drop(columns=['text_len'],inplace=True)
    return train, valid

def tokenize_data(examples, _args, tokenizer):
    #Tokenize function
    return tokenizer(examples[_args.text_col], truncation=True, max_length = _args.train_max_len)

def preprocessing(examples, _args, tokenizer, shuffle = True):
    #Read csv file containing mapping of labels to their categorical encoding 
    label2id = pd.read_csv(_args.label2id).set_index('Label')['ID'].to_dict()

    if _args.append_rubrik != "None":
        #Prepend scoring guideline text
        print("Prepending rubrik scoring guideline text")
        examples[_args.text_col] = text_to_append + examples[_args.text_col]
    
    if _args.newline_token_strategy == "pipe":
        # \n token is not available in deberta tokenizer, we replace it with |
        print(f"Replacing NEWLINE character with pipe operator: '|'")
        examples[_args.text_col] = examples[_args.text_col].str.replace("\n","|")
    
    #Creating target column based on mapping
    if _args.use_regression == "True":
        print("Converting target to float for regression")
        examples['label'] = examples[_args.target_col].apply(lambda x: float(x)-1.0)
    else:
        print("Converting target into encoded classes for classification")
        examples['label'] = examples[_args.target_col].apply(lambda x: label2id[x])
    
    if shuffle == True:
        #Shuffle dataset
        print("Shuffling dataset")
        examples = Dataset.from_pandas(examples.sample(frac=1).reset_index(drop=True))
    else:
        examples = Dataset.from_pandas(examples.reset_index(drop=True))
    

    examples = examples.map(tokenize_data, fn_kwargs={"_args":_args, "tokenizer":tokenizer}, batched=True, remove_columns = [_args.text_col,_args.target_col])
    return examples

def preprocess_data(_args, fold, tokenizer):
    print("Reading data")
    data = read_data(_args)
    print("Total data shape:",data.shape)

    print("Splitting data into TRAIN-VALIDATION")
    train, valid = split_data(data, _args, fold)

    print("Preprocessing TRAIN data")
    train = preprocessing(train, _args, tokenizer, shuffle=True)

    print("Preprocessing VALIDATION data")
    valid = preprocessing(valid, _args, tokenizer, shuffle=False)
    return train, valid