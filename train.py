import argparse
from ast import literal_eval
import os
import random
import re
import numpy as np
import torch
import wandb
from datasets import load_dataset
import yaml

# from accelerate import Accelerator
from transformers import MistralForCausalLM, MistralConfig, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments

from mission import Mission, MissionTokenizer, MissionDataCollator
from utils import get_missions_vocab, gen_mission_str

if __name__ == '__main__':
    # parse arguments TODO: add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    print(args)

    # read training config from a json file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    # set seeds
    seed = config['seed']
    random.seed(seed)
    torch.manual_seed(seed)

    # set up accelerator
    # accelerator = Accelerator()

    # read model config from a json file
    with open(config['model_config'], 'r') as f:
        model_config = yaml.safe_load(f)
    print(model_config)

    # set up model and tokenizer TODO: from_pretrained
    num_node_tokens = config['num_node_tokens']
    vocab = get_missions_vocab(num_node_tokens)
    model_config['vocab_size'] = len(vocab)
    model_config = MistralConfig(**model_config)
    model = MistralForCausalLM(model_config)
    print(model.num_parameters())
    print(model)
    context_length = config['context_length']
    tokenizer = MissionTokenizer(vocab)
    tokenizer.model_max_length = context_length
    # tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.3')

    # set up wandb TODO
    if args.wandb:
        wandb_config = config.get('wandb')
        print(wandb_config)
        wandb.init(config=config, **wandb_config)

    # load dataset TODO
    data_dir = config['data_dir']
    train_file = os.path.join(data_dir, config['train_file'])
    val_file = os.path.join(data_dir, config['val_file'])
    datasets = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "val": val_file,
        },
    )
    print(datasets)
    datasets['train'] = datasets['train'].select(range(config['num_train']))

    # tokenize dataset
    def tokenize(example):
        mission = Mission(example, from_dict=True)
        node_ids = np.random.permutation(num_node_tokens)
        clues = {'Search': literal_eval(example['search']), 'Decision': [example['target']]}
        mission_str = gen_mission_str(mission, node_ids, clues)
        token_ids = tokenizer.encode(mission_str, config['context_length'])
        return {'input_ids': token_ids}

    tokenized_datasets = datasets.map(tokenize, batched=False, remove_columns=datasets["train"].column_names)
    print(tokenized_datasets)
    data_collator = MissionDataCollator(tokenizer, mlm=False)

    # sample = datasets['train'][0]
    # mission = Mission(sample, from_dict=True)
    # node_id = np.random.permutation(num_node_tokens)
    # search = literal_eval(sample['search'])
    # mission_str = gen_mission_str(mission, node_id, {'Search': search, 'Decision': [mission.target]})
    # pattern = r"([A-Za-z]+|\d+|\S)"
    # tokens = re.findall(pattern, mission_str)
    # print(tokenizer.encode(mission_str, 100))
    # mission = datasets['train'][0]
    # s = 'Graph: ' + mission['graph'] + ' Task: ' + str(mission['start']) + ' to ' + str(mission['target']) + ' Search:' + mission['search']
    # print(s)
    # pattern = r"[A-Za-z]+|\d+|\S"
    # print(re.findall(pattern, s))

    # TODO: attention mask
    # prepare training
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['lr'],
        lr_scheduler_type=config['lr_scheduler_type'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
        num_train_epochs=config['num_train_epochs'],
        seed=config['seed'],
        fp16=True,
        logging_strategy='steps',
        logging_steps=config['log_steps'],
        eval_strategy='steps',
        eval_steps=config['eval_steps'],
        report_to='wandb' if args.wandb else 'none',
        run_name=config['name'],
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
    )

    # train TODO: resume
    trainer.train()


# TrainingArguments(
        # output_dir=config["output_dir"],
        # per_device_train_batch_size=config["batch_size"],
        # evaluation_strategy="steps",
        # eval_steps=config["eval_steps"],
        # logging_steps=config["log_steps"],
        # gradient_accumulation_steps=config["gradient_accumulation_steps"],
        ## gradient_checkpointing=True,
        # num_train_epochs=config["num_train_epochs"],
        # weight_decay=config["weight_decay"],
        # warmup_steps=config["warmup_steps"],
        # lr_scheduler_type=config["lr_scheduler_type"],
        # learning_rate=config["lr"],
        # save_strategy="steps",
        # save_total_limit=config["save_total_limit"],
        ## save_steps=config["save_steps"],
        # seed=config["seed"],
        ## bf16=True,
        # push_to_hub=False,
        # report_to="wandb",
        # run_name=config["name"],
        ## ddp_find_unused_parameters=False,
        # load_best_model_at_end=True,
        ## torch_compile=True,
        # metric_for_best_model="valid_loss",
        ## greater_is_better=False,
    # )


    # TrainingArguments(
        # output_dir='./output_dir/' + wandb_name,
        ## do_train=True,
        ## do_eval=True,
        # per_device_train_batch_size=args.train_batch_size,
        # per_device_eval_batch_size=args.eval_batch_size,
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        # learning_rate=args.learning_rate,
        # weight_decay=args.weight_decay,
        ## max_steps=max_train_steps,
        # lr_scheduler_type="linear",
        # warmup_steps=warmup_steps,
        # other args and kwargs here
        # report_to="wandb",  # enable logging to W&B
        # run_name=wandb_name,  # name of the W&B run (optional)
        # logging_strategy="steps",
        # logging_steps=50,  # how often to log to W&B
        ## save_strategy="no",
        # save_steps=500,
        # save_total_limit=2,
        # seed=args.seed,
        # fp16=True if torch.cuda.is_available() else False,
        # fsdp="full_shard",
        ## torch_compile=False,
        # eval_strategy="steps",
        # eval_steps=100,
        ## remove_unused_columns=False,
    # )