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

from accelerate import Accelerator
from transformers import MistralForCausalLM, MistralConfig, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments

from mission import Mission, MissionTokenizer, MissionDataCollator
from utils import get_missions_vocab, gen_mission_str


def verify_answer(mission, node_ids, answer_str, return_works=False):
    node_id_revs = {node_ids[i]: i for i in range(len(node_ids))}
    evidence = False
    decision = False
    work_pattern = r'(work|decision):\[(\d+(?:,\d+)*)\]'
    work_match = re.findall(work_pattern, answer_str.replace(' ', ''))
    works = []
    for work_type, work_str in work_match:
        work_list = list(map(int, work_str.split(',')))                        
        work_list = [node_id_revs.get(node_id, -1) for node_id in work_list]        
        works += work_list
        if work_type == 'decision':
            decision = (work_list[0] == mission.target)
        elif work_type == 'work':
            valid_walk = all(mission.graph.has_edge(work_list[i], work_list[i + 1]) for i in range(len(work_list) - 1))
            evidence = valid_walk and (work_list[0] == mission.start) and (work_list[-1] == mission.target)
        
    if return_works:
        return evidence, decision, works
    return evidence, decision


def answer_questions(model, tokenizer, question_strs, do_sample=False, **kwargs):
    questions = tokenizer(question_strs, return_tensors='pt', padding=True, truncation=False).to(model.device)
    assert(questions['input_ids'].shape[1] <= tokenizer.model_max_length)
    input_ids = questions['input_ids'][:, :-1] # remove eos token
    attention_mask = questions['attention_mask'][:, :-1]
    gens = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=do_sample, **kwargs)
    answers = gens[:, input_ids.shape[1]:] # remove the input
    answer_strs = tokenizer.batch_decode(answers)
    return answer_strs


best_metrics = {
            'str_accuracy': 0,
            'char_accuracy': 0,
            'decision_accuracy': 0,
            'evidence_accuracy': 0,
        }


def create_compute_metrics(model, tokenizer, eval_dataset):
    def compute_metrics(eval_pred):
        mission_strs = eval_dataset['mission_strs']
        all_preds, labels = eval_pred
        preds = np.argmax(all_preds, axis=-1)
        preds, labels = preds[:, :-1], labels[:, 1:]
        output_mask = (labels != -100)
        num_outputs = np.sum(output_mask, axis=-1)
        num_corr_outputs = np.sum((preds == labels) & output_mask, axis=-1)
        str_acc = np.mean(num_corr_outputs == num_outputs)
        char_acc = np.mean(num_corr_outputs / num_outputs)

        question_strs = [s[:s.find(tokenizer.sep_token) + 1] for s in mission_strs]
        answer_strs = answer_questions(model, tokenizer, question_strs)

        decision_acc = []
        evidence_acc = []
        for i in range(len(mission_strs)):
            mission = Mission(eval_dataset[i], from_dict=True)
            node_ids = eval_dataset[i]['node_ids']
            answer_str = answer_strs[i]
            evidence, decision, works = verify_answer(mission, node_ids, answer_str, return_works=True)
            decision_acc.append(decision)
            evidence_acc.append(evidence)
            # if i < 5:
            #     print(f'Question: {question_strs[i]}')
            #     print(f'Answer: {answer_str}')
            #     print(f'Decision: {decision}')
            #     print(f'Evidence: {evidence}')
            #     print(f'Works: {works}')
    
        decision_acc = np.mean(decision_acc)
        evidence_acc = np.mean(evidence_acc)
        metrics = {
            'str_accuracy': str_acc,
            'char_accuracy': char_acc,
            'decision_accuracy': decision_acc,
            'evidence_accuracy': evidence_acc,
        }
        for key, value in metrics.items():
            if value > best_metrics[key]:
                best_metrics[key] = value
        for key, value in best_metrics.items():
            metrics[f"best_{key}"] = value
        return metrics
    
    return compute_metrics


def get_run_name(config):
    run_name = '_'.join([config['teach']] + config['train_file'].split('_')[1:5]) 
    run_name += f"_p{config['num_node_tokens']}_n{config['num_train']}_e{config['num_epochs']}_lr{config['lr']}_wd{config['weight_decay']}_sc{config['lr_scheduler_type']}"
    return run_name


if __name__ == '__main__':
    # parse arguments: add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--seed", type=int, help="Override the random seed")
    parser.add_argument("--teach", type=str, help="Override the teaching strategy")  
    parser.add_argument("--num_train", type=int, help="Override the number of training samples")
    parser.add_argument("--num_epochs", type=int, help="Override the number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override the batch size")
    parser.add_argument("--weight_decay", type=float, help="Override the weight decay")
    parser.add_argument("--lr", type=float, help="Override the learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, help="Override the learning rate scheduler type")

    args = parser.parse_args()

    # read training config from a json file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # override config with command line arguments
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value

    # set seeds
    seed = config['seed']
    random.seed(seed)
    torch.manual_seed(seed)

    # set up accelerator
    accelerator = Accelerator()
    print(accelerator.device)

    # read model config from a json file
    with open(config['model_config'], 'r') as f:
        model_config = yaml.safe_load(f)
    
    run_name = get_run_name(config)
    print(run_name)
    
    # set up wandb
    report_to_wandb = args.wandb # and accelerator.is_main_process
    if report_to_wandb:
        wandb_config = config.get('wandb_config')
        wandb_config['name'] = run_name
        wandb.init(config=config, **wandb_config)

    # set up model and tokenizer: from_pretrained
    num_node_tokens = config['num_node_tokens']    
    context_length = config['context_length']
    vocab = get_missions_vocab(num_node_tokens)    
    tokenizer = MissionTokenizer(vocab)
    tokenizer.model_max_length = context_length  
    tokenizer.padding_side = 'left'      
    
    model_config['vocab_size'] = len(vocab)
    model_config['pad_token_id'] = tokenizer.pad_token_id
    model_config['bos_token_id'] = tokenizer.bos_token_id
    model_config['eos_token_id'] = tokenizer.eos_token_id
    model_config['max_length'] = tokenizer.model_max_length
    model_config = MistralConfig(**model_config)
    model = MistralForCausalLM(model_config)

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
        cache_dir=data_dir,
    )
    datasets['train'] = datasets['train'].select(range(config['num_train']))

    teach = config['teach'].split(',')
    print(teach)

    # tokenize dataset
    def tokenize(example, idx):
        mission = Mission(example, from_dict=True)
        search_type = teach[idx % len(teach)]
        clues = {'work': literal_eval(example[search_type]), 'decision': [example['target']]}
        node_ids = np.random.permutation(num_node_tokens)[:mission.number_of_nodes()]
        mission_str = gen_mission_str(mission, node_ids=node_ids, clues=clues)
        # if idx < 5:
        #     print(f"Mission {idx}: {mission_str}")
        token_ids = tokenizer.encode(mission_str)
        return {'input_ids': token_ids, 'mission_strs': mission_str, 'node_ids': node_ids} 

    tokenized_datasets = datasets.map(tokenize, with_indices=True, batched=False) #, remove_columns=datasets["train"].column_names)
    data_collator = MissionDataCollator(tokenizer, mlm=False)

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
        warmup_ratio=config['warmup_ratio'],
        num_train_epochs=config['num_epochs'],
        seed=config['seed'],
        fp16=True,
        logging_strategy='steps',
        logging_steps=config['log_steps'],
        eval_strategy='steps',
        eval_steps=config['eval_steps'],
        report_to='wandb' if report_to_wandb else 'none',
        run_name=run_name,
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model='decision_accuracy',
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
        checkpoint_path = os.path.join(training_args.output_dir, args.ckpt)
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()

    # if report_to_wandb:
    #     for key, value in best_metrics.items():
    #         wandb.run.summary[f"eval/best_{key}"] = value
    #     wandb.finish()