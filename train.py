import argparse
from ast import literal_eval
import os
import random
import re
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from datasets import load_dataset
import yaml

# from accelerate import Accelerator
from transformers import MistralForCausalLM, MistralConfig, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments

from mission import Mission, MissionTokenizer, MissionDataCollator
from utils import get_missions_vocab, gen_mission_str


def create_compute_metrics(model, tokenizer, eval_dataset):
    def compute_metrics(eval_pred):
        input_ids = eval_dataset['input_ids']
        mission_strs = eval_dataset['mission_strs']
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=-1)
        preds, labels = preds[:, :-1], labels[:, 1:]
        output_mask = (labels != -100)
        num_outputs = np.sum(output_mask, axis=-1)
        num_corr_outputs = np.sum((preds == labels) & output_mask, axis=-1)
        metrics = {
            'acc': np.mean(num_corr_outputs == num_outputs),
            'char_acc': np.mean(num_corr_outputs / num_outputs),
        }
        question_strs = [s[:s.find(tokenizer.sep_token) + 1] for s in mission_strs]
        questions = tokenizer(question_strs, return_tensors='pt', padding=True, truncation=True, max_length=tokenizer.model_max_length).to(model.device)
        gens = model.generate(input_ids=questions['input_ids'], attention_mask=questions['attention_mask'], pad_token_id=tokenizer.pad_token_id, max_length=tokenizer.model_max_length, do_sample=False, num_beams=1)
        answers = [gen[gen.tolist().index(tokenizer.sep_token_id) + 1:] for gen in gens]
        answer_strs = tokenizer.batch_decode(answers)

        decision_acc = []
        evidence_acc = []
        for i in range(len(mission_strs)):
            mission = Mission(eval_dataset[i], from_dict=True)
            node_ids = eval_dataset[i]['node_ids']
            node_id_revs = {node_ids[i]: i for i in range(len(node_ids))}
            answer_str = answer_strs[i]
            evidence = False
            decision = False
            work_pattern = r'(optimal|search|decision):\[(\d+(?:,\d+)*)\]'
            work_match = re.findall(work_pattern, answer_str.replace(' ', ''))
            for work_type, work_str in work_match:
                work_list = list(map(int, work_str.split(',')))                
                work_list = [node_id_revs.get(node_id, -1) for node_id in work_list]
                if work_type == 'optimal':
                    valid_walk = all(mission.graph.has_edge(work_list[i], work_list[i + 1]) for i in range(len(work_list) - 1))
                    evidence = valid_walk and (work_list[0] == mission.start) and (work_list[-1] == mission.target)
                elif work_type == 'decision':
                    decision = (work_list[0] == mission.target)
            decision_acc.append(decision)
            evidence_acc.append(evidence)
        decision_acc = np.mean(decision_acc)
        evidence_acc = np.mean(evidence_acc)
        metrics['decision_acc'] = decision_acc
        metrics['evidence_acc'] = evidence_acc
        return metrics
    
    return compute_metrics


if __name__ == '__main__':
    # parse arguments: add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--reset", action="store_true")    

    args = parser.parse_args()
    # print(args)

    # read training config from a json file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # print(config)

    # set seeds
    seed = config['seed']
    random.seed(seed)
    torch.manual_seed(seed)

    # set up accelerator
    # accelerator = Accelerator()

    # read model config from a json file
    with open(config['model_config'], 'r') as f:
        model_config = yaml.safe_load(f)
    # print(model_config)
    run_name = f"{config['task']}_{config['teach']}_n{config['num_train']}"

    # set up model and tokenizer: from_pretrained
    num_node_tokens = config['num_node_tokens']    
    context_length = config['context_length']
    vocab = get_missions_vocab(num_node_tokens)    
    tokenizer = MissionTokenizer(vocab)
    tokenizer.model_max_length = context_length  
    tokenizer.padding_side = 'left'      
    if args.reset:
        model = MistralForCausalLM.from_pretrained(args.ckpt).to('cuda')
    else:
        model_config['vocab_size'] = len(vocab)
        model_config['pad_token_id'] = tokenizer.pad_token_id
        model_config['bos_token_id'] = tokenizer.bos_token_id
        model_config['eos_token_id'] = tokenizer.eos_token_id
        model_config = MistralConfig(**model_config)
        model = MistralForCausalLM(model_config)
    # print('model parameters:', model.num_parameters())
    # print(tokenizer.vocab)
    # print(model)  

    # set up wandb
    if args.wandb:
        wandb_config = config.get('wandb')
        wandb_config['name'] = run_name
        print(wandb_config)
        wandb.init(config=config, **wandb_config)

    # load dataset
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

    teach = config['teach']

    # tokenize dataset
    def tokenize(example):
        mission = Mission(example, from_dict=True)
        clues = {teach: literal_eval(example[teach]), 'decision': [example['target']]}
        node_ids = np.random.permutation(num_node_tokens)[:mission.number_of_nodes()]
        mission_str = gen_mission_str(mission, node_ids=node_ids, clues=clues)
        token_ids = tokenizer.encode(mission_str, config['context_length'])
        return {'input_ids': token_ids, 'mission_strs': mission_str, 'node_ids': node_ids} 

    tokenized_datasets = datasets.map(tokenize, batched=False) #, remove_columns=datasets["train"].column_names)
    data_collator = MissionDataCollator(tokenizer, mlm=False)
    print(tokenized_datasets, flush=True)
    # prepare training
    output_dir = f"{config['output_dir']}/{run_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
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
        run_name=run_name,
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],
        # load_best_model_at_end=True,
        # metric_for_best_model='eval_loss',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        compute_metrics=create_compute_metrics(model, tokenizer, tokenized_datasets['val']),
    )

    # train
    if args.resume:
        trainer.train(resume_from_checkpoint=args.ckpt)
    else:
        trainer.train()