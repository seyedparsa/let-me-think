import argparse
from ast import literal_eval
import os
import random
import re
import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from datasets import load_dataset
import yaml

from accelerate import Accelerator
from transformers import MistralForCausalLM, MistralConfig, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

from mission import Mission, MissionTokenizer, MissionDataCollator
from utils import get_missions_vocab, gen_mission_str
from eval.eval_utils import load_data, load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--models_file', type=str)
    parser.add_argument('--teaches', type=str, nargs='+')
    # parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument('--temperature', type=float, default=1.0)
    # parser.add_argument('--scale', action='store_true')
    # parser.add_argument('--log_trials', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)   
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed) 

    with open('eval/config/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open(f'eval/config/d{args.depth}.yaml', 'r') as f:
        config.update(yaml.safe_load(f))
    
    tokenizer, dataset = load_data(config, eval_data=False)    
    question_strs = [example['mission_strs'] for example in dataset]
    missions = [Mission(example, from_dict=True) for example in dataset]
    node_ids = [example['node_ids'] for example in dataset]

    sweep_name = config['sweep_name']
    metric_name = config['metric_for_best_model']['name']
    models_file = args.models_file    
    if models_file is None:
        models_file = f"csv/sweeps/{sweep_name}_best_{metric_name}.csv"
    df = pd.read_csv(models_file)
    teaches_to_eval = args.teaches
    teaches_to_eval = args.teaches
    if teaches_to_eval is None:
        teaches_to_eval = df['teach'].unique()
    print(df)
    print(f"Evaluating models with teaches: {teaches_to_eval}")
    
    results = []
    for index, row in df.iterrows():
        model_name = row['name']
        sweep_id = row['sweep_id']
        teach = row['teach']
        if teach not in teaches_to_eval:
            continue
        model = load_model(config['output_dir'], sweep_id, model_name)
        model.config.max_length = config['context_length']
