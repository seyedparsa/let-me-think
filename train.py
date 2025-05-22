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
import json
import pandas as pd

from accelerate import Accelerator
from transformers import MistralForCausalLM, MistralConfig, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

from mission import Mission, MissionTokenizer, MissionDataCollator
from utils import get_missions_vocab, gen_mission_str, load_model

def extract_works(node_ids, answer_str):
    node_id_revs = {node_ids[i]: i for i in range(len(node_ids))}
    work_pattern = r'(work|decision)\s*:\s*\[\s*([\d\s*]+)\s*\]'
    work_match = re.findall(work_pattern, answer_str)
    works = {
        'work': [],
        'decision': [],
    }
    for work_type, work_str in work_match:
        work_list = list(map(int, work_str.split()))                        
        work_list = [node_id_revs.get(node_id, -1) for node_id in work_list]      
        works[work_type] += work_list
    return works


def verify_works(mission, works, return_validity=False):
    decision, evidence = works.get('decision', []), works.get('work', [])
    decision_verdict = len(decision) == 1 and (decision[0] == mission.target)
    evidence_verdict = False
    valid_edge_rate = 0.0
    valid_walk = False
    if len(evidence) > 1:  
        num_graph_nodes = mission.graph.number_of_nodes()
        can_visit = [False for _ in range(num_graph_nodes)]   
        can_visit[mission.start] = True
        evidence_verdict = (evidence[0] == mission.start) and (evidence[-1] == mission.target)
        for node in evidence:
            if evidence_verdict == False:
                break
            if node >= 0 and node < num_graph_nodes and can_visit[node]:
                for neighbor in mission.graph.neighbors(node):
                    can_visit[neighbor] = True
            else:
                evidence_verdict = False
        valid_edges = [mission.graph.has_edge(evidence[i], evidence[i + 1]) for i in range(len(evidence) - 1)]
        valid_edge_rate = np.mean(valid_edges)
        valid_walk = all(valid_edges)
        # evidence_verdict = valid_walk and (evidence[0] == mission.start) and (evidence[-1] == mission.target)
    res = [decision_verdict, evidence_verdict]
    if return_validity:
        res += [valid_walk, valid_edge_rate]
    return res


def verify_answer(mission, node_ids, answer_str, return_validity=False, return_works=False):
    works = extract_works(node_ids, answer_str)
    res = verify_works(mission, works, return_validity=return_validity)
    if return_works:
        res.append(works)
    return res


def verify_aggregate(mission, node_ids, answer_strs, works_list=None, seq_budget=None):
    if works_list is None:
        works_list = [extract_works(node_ids, answer_str) for answer_str in answer_strs]
    if seq_budget is not None:
        for works in works_list:
            works['work'] = works['work'][:seq_budget]
    decision_list = [works['decision'] for works in works_list]
    evidence_verdict = any([verify_works(mission, works)[1] for works in works_list])
    decisions = [decision[0] for decision in decision_list if len(decision) == 1]
    decision = None
    if decisions:
        target_count, tegrat_count = decisions.count(mission.target), decisions.count(mission.tegrat)
        if target_count == tegrat_count:
            decision = random.choice([mission.target, mission.tegrat])
        else:
            decision = mission.target if target_count > tegrat_count else mission.tegrat
    decision_verdict = (decision is not None and decision == mission.target)
    return decision_verdict, evidence_verdict


def answer_questions(model, tokenizer, question_strs, do_sample=False, return_tokens=False, **kwargs):
    with torch.inference_mode():
        questions = tokenizer(question_strs, return_tensors='pt', padding=True, truncation=False).to(model.device)
        assert(questions['input_ids'].shape[1] <= tokenizer.model_max_length)
        input_ids = questions['input_ids'][:, :-1] # remove eos token
        attention_mask = questions['attention_mask'][:, :-1]
        gens = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=do_sample, **kwargs)
        answers = gens[:, input_ids.shape[1]:] # remove the input
        if return_tokens:
            return answers
        answer_strs = tokenizer.batch_decode(answers)
        return answer_strs


best_metrics = {
            'str_accuracy': 0,
            'char_accuracy': 0,
            'edge_validity': 0,
            'walk_validity': 0,
            'decision_accuracy': 0,
            'evidence_accuracy': 0,
        }


def create_compute_metrics(model, tokenizer, eval_dataset, play_file=None, **kwargs):
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
        answer_tokens = answer_questions(model, tokenizer, question_strs, return_tokens=True, **kwargs)
        answer_strs = tokenizer.batch_decode(answer_tokens)

        decision_acc = []
        evidence_acc = []
        valid_walk_acc = []
        valid_edge_acc = []
        num_thought_tokens = []
        num_valid_thought_tokens = []
        num_verified_thought_tokens = []
        # prints = np.random.permutation(len(mission_strs))[:10]
        verified = []
        non_verified = []
        # play_dataset = []
        for i in range(len(mission_strs)):
            mission = Mission(eval_dataset[i], from_dict=True)
            node_ids = eval_dataset[i]['node_ids']
            answer_str = answer_strs[i]
            decision, evidence, valid_walk, valid_edge_rate, works = verify_answer(mission, node_ids, answer_str, return_validity=True, return_works=True)
            valid_edge_acc.append(valid_edge_rate)
            valid_walk_acc.append(valid_walk)
            decision_acc.append(decision)
            evidence_acc.append(evidence)

            # print(len(works['work']), len(works['decision']), evidence, decision)
            
            if works['work']:
                thought_length = len(works['work'])
                # print(f"Work Found, Answer: {answer_str}")
                num_thought_tokens.append(thought_length)
                num_valid_thought_tokens.append(thought_length)
                if evidence:
                    num_verified_thought_tokens.append(thought_length)
                    # if play_file:
                    #     keys = ['graph_type', 'task_type', 'graph', 'start', 'target', 'phrag', 'tegrat']
                    #     play_data = {key: eval_dataset[i][key] for key in keys}                    
                    #     play_data['play'] = str(works['work'])
                    #     play_dataset.append(play_data)
            else:                         
                # print(f"Work Empty, Answer: {answer_str}")
                tokens = answer_tokens[i].tolist()
                # find the first token not in node_ids
                first_invalid = 3
                while first_invalid < len(tokens) and tokens[first_invalid] in node_ids:
                    first_invalid += 1                
                # print(f"First invalid token: {first_invalid}")
                num_thought_tokens.append(first_invalid - 3)
                continue
                decision_token_id = tokenizer.vocab['decision']
                if decision_token_id in tokens:
                    decision_index = tokens.index(decision_token_id)
                    num_thought_tokens.append(max(0, decision_index - 4))
                    # print(f"Decision token found at index {decision_index}")
                elif tokenizer.eos_token_id in tokens:
                    eos_index = tokens.index(tokenizer.eos_token_id)
                    num_thought_tokens.append(max(0, eos_index - 4))
                    # print(f"EOS token found at index {eos_index}")
                else:
                    num_thought_tokens.append(len(tokens) - 4)   
                    print(f"Work Empty, Answer: {answer_str}")     
                    print(f"Neither decision nor EOS token found {len(tokens)}")        

            if evidence:
                verified.append(works)
            else:
                non_verified.append(works)
                # print(f'Index: {i}')
                # print(f'Question: {question_strs[i]}')
                # print(f'Answer: {answer_str}')
                # print(f'Works: {works}')
                # print(f'Valid Walk: {valid_walk}')
                # print(f'Valid Edge Rate: {valid_edge_rate}')
                # print(f'Decision: {decision}')
                # print(f'Evidence: {evidence}')

        print(f"Verified: {len(verified)}")
        for i in range(min(10, len(verified))):
            print(f'Works: {verified[i]}')
        print()
        print(f"Non-verified: {len(non_verified)}")
        for i in range(min(10, len(non_verified))):
            print(f'Works: {non_verified[i]}')
                
        valid_edge_acc = np.mean(valid_edge_acc)
        valid_walk_acc = np.mean(valid_walk_acc)
        decision_acc = np.mean(decision_acc)
        evidence_acc = np.mean(evidence_acc)

        # generation_mask = (answer_tokens != tokenizer.pad_token_id) & (answer_tokens != tokenizer.eos_token_id)
        # num_generations = torch.sum(generation_mask, dim=-1)
        # avg_gens = torch.mean(num_generations.float()).item()
        avg_thoughts = np.mean(num_thought_tokens)
        std_thoughts = np.std(num_thought_tokens)
        avg_valid_thoughts = np.mean(num_valid_thought_tokens) if num_valid_thought_tokens else 0
        std_valid_thoughts = np.std(num_valid_thought_tokens) if num_valid_thought_tokens else 0
        avg_verified_thoughts = np.mean(num_verified_thought_tokens) if num_verified_thought_tokens else 0
        std_verified_thoughts = np.std(num_verified_thought_tokens) if num_verified_thought_tokens else 0
        # print(len(num_thought_tokens)/len(mission_strs), avg_gens)
        # print(f"Avg thoughts: {avg_thoughts}, Avg valid thoughts: {avg_valid_thoughts}, Avg verified thoughts: {avg_verified_thoughts}")

        metrics = {
            'str_accuracy': str_acc,
            'char_accuracy': char_acc,
            'edge_validity': valid_edge_acc,
            'walk_validity': valid_walk_acc,
            'decision_accuracy': decision_acc,
            'evidence_accuracy': evidence_acc,
            'avg_thoughts': avg_thoughts,
            'std_thoughts': std_thoughts,
            'avg_valid_thoughts': avg_valid_thoughts,
            'std_valid_thoughts': std_valid_thoughts,
            'avg_verified_thoughts': avg_verified_thoughts,
            'std_verified_thoughts': std_verified_thoughts,
        }
        for key, value in metrics.items():
            if key in best_metrics and value > best_metrics[key]:
                best_metrics[key] = value

        # if play_file:
        #     with open(play_file, 'w') as f:
        #         json.dump(play_dataset, f, indent=4)
        #     print(f"Play dataset of size {len(play_dataset)} saved to {play_file}")

        for key, value in best_metrics.items():
            metrics[f"best_{key}"] = value
        return metrics
    
    return compute_metrics


def get_run_name(config):
    run_name = '_'.join([config['teach']] + config['val_file'].split('_')[1:5]) 
    if config.get('num_train') is not None:
        run_name += f"_n{config['num_train']}"
    else:
        run_name += f"_t{config['num_tokens']}"
    run_name += f"_p{config['num_node_tokens']}_e{config['num_epochs']}_s{config['seed']}_lr{config['lr']}_wd{config['weight_decay']}_sc{config['lr_scheduler_type']}"
    return run_name


if __name__ == '__main__':
    # parse arguments: add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--resume_teach", type=str)
    parser.add_argument("--resume_depth", type=int, default=3)

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

    # load dataset
    data_dir = config['data_dir']    
    # max_num_train = config['max_num_train']    
    teach = config['teach']
    # train_dataset = train_dataset.select(range(max_num_train))    
    print(teach)
    

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

    print(f"Model config: {model_config}")
    
    
    metric = None
    greater_is_better = False
    if args.resume:
        with open('eval/config/base.yaml', 'r') as f:
            eval_config = yaml.safe_load(f)
        with open(f'eval/config/d{args.resume_depth}.yaml', 'r') as f:
            eval_config.update(yaml.safe_load(f))
        num_node_tokens = eval_config['num_node_tokens'] 
        context_length = eval_config['context_length']
        metric = eval_config['metric_for_best_model']['name']
        greater_is_better = (eval_config['metric_for_best_step']['mode'] == 'max')
        config['val_file'] = eval_config['val_file']
        config['num_node_tokens'] = num_node_tokens
        config['context_length'] = context_length
        val_file = os.path.join(data_dir, eval_config['val_file'])
        val_dataset = load_dataset("json", data_files=val_file, cache_dir=data_dir)
        train_file = os.path.join(data_dir, eval_config[teach])
        train_dataset = load_dataset("json", data_files=train_file, cache_dir=data_dir)
        eval_sweep_name = eval_config['sweep_name']
        eval_metric_name = eval_config['metric_for_best_model']['name']
        models_file = f"csv/sweeps/{eval_sweep_name}_best_{eval_metric_name}.csv"
        df = pd.read_csv(models_file)
        # set sweep_id and model_name from the row that has teach = args.resume_teach
        if args.resume_teach is not None:
            resume_teach = args.resume_teach
        else:
            assert 'play' in teach
            bteach, _, num_play = teach.split('-')
            num_play = int(num_play)
            if num_play == 1:
                resume_teach = bteach
            else:
                resume_teach = f"{bteach}-play-{num_play-1}"
        row = df[df['teach'] == resume_teach]
        eval_model_name = row['name'].values[0]
        eval_sweep_id = row['sweep_id'].values[0]
        model = load_model(config['output_dir'], eval_sweep_id, eval_model_name)
    else:
        num_node_tokens = config['num_node_tokens']    
        context_length = config['context_length']
        train_file = os.path.join(data_dir, config['train_file'])
        val_file = os.path.join(data_dir, config['val_file'])
        train_dataset = load_dataset("json", data_files=train_file, cache_dir=data_dir)
        val_dataset = load_dataset("json", data_files=val_file, cache_dir=data_dir)
    
    if config.get('num_train') is None and config.get('num_tokens') is None:
        config['num_train'] = len(train_dataset['train'])
    # print(f"Number of training samples: {config['num_train']}")
    # set up model and tokenizer: from_pretrained

    run_name = get_run_name(config)
    print(run_name)
    
    # set up wandb
    report_to_wandb = args.wandb # and accelerator.is_main_process
    sweep_name = None
    if report_to_wandb:
        wandb_config = config.get('wandb_config')
        wandb_config['name'] = run_name
        wandb.init(config=config, **wandb_config)
        sweep_name = wandb.run.sweep_id    
    
    vocab = get_missions_vocab(num_node_tokens)    
    tokenizer = MissionTokenizer(vocab)
    tokenizer.model_max_length = context_length  
    tokenizer.padding_side = 'left'      

    if not args.resume:
        model_config['vocab_size'] = len(vocab)
        model_config['pad_token_id'] = tokenizer.pad_token_id
        model_config['bos_token_id'] = tokenizer.bos_token_id
        model_config['eos_token_id'] = tokenizer.eos_token_id
        model_config['max_length'] = tokenizer.model_max_length
        model_config = MistralConfig(**model_config)
        model = MistralForCausalLM(model_config)    
    

    req_context_length = 0
    max_work_length = 0
    # tokenize dataset
    def tokenize(example, idx):
        global req_context_length
        global max_work_length
        mission = Mission(example, from_dict=True)
        clues = {}      
        if teach in example:
            clues['work'] = literal_eval(example[teach])
        clues['decision'] = [example['target']]
        node_ids = np.random.permutation(num_node_tokens)[:mission.number_of_nodes()]
        mission_str = gen_mission_str(mission, node_ids=node_ids, clues=clues)
        # print(f"Mission {idx}: {mission_str}")
        # if idx < 5:
        #     print(f"Mission {idx}: {mission_str}")
        token_ids = tokenizer.encode(mission_str)
        req_context_length = max(req_context_length, len(token_ids))
        max_work_length = max(max_work_length, len(clues['work']))
        return {'input_ids': token_ids, 'mission_strs': mission_str, 'node_ids': node_ids} 

    train_dataset = train_dataset.map(tokenize, with_indices=True, batched=False)['train']
    print(f"Max work length: {max_work_length}")
    val_dataset = val_dataset.map(tokenize, with_indices=True, batched=False)['train']
    # datasets = {'train': train_dataset, 'val': val_dataset}
    # tokenized_datasets = datasets.map(tokenize, with_indices=True, batched=False) #, remove_columns=datasets["train"].column_names)
    if config.get('num_train') is not None:
        num_train = config['num_train']
        num_tokens = 0
        for i in range(num_train):
            token_ids = train_dataset[i]['input_ids']
            num_tokens += len(token_ids) - (token_ids.index(tokenizer.sep_token_id) + 1)
    else:
        num_tokens = config['num_tokens']    
        num_train = 0
        partial_num_tokens = 0
        while partial_num_tokens < num_tokens:
            token_ids = train_dataset[num_train]['input_ids']
            partial_num_tokens += len(token_ids) - (token_ids.index(tokenizer.sep_token_id) + 1)
            num_train += 1
        assert partial_num_tokens >= num_tokens
    train_dataset = train_dataset.select(range(num_train))
    
    print(num_train, num_tokens, req_context_length)
    # print(f"Average number of tokens: {num_tokens / len(tokenized_datasets['train'])}")
    # print(f"Total number of tokens: {num_tokens}")
    data_collator = MissionDataCollator(tokenizer, mlm=False)

    # prepare training
    if sweep_name is not None:
        output_dir = f"{config['output_dir']}/sweeps/{sweep_name}/{run_name}"
    else:
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
        metric_for_best_model="eval_loss" if metric is None else metric,
        greater_is_better=greater_is_better,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
        compute_metrics=create_compute_metrics(model, tokenizer, val_dataset),
    )
    # print(trainer.evaluate())
    
    # train
    # if args.resume:
    #     checkpoint_path = os.path.join(args.ckpt)
    #     trainer.train(resume_from_checkpoint=checkpoint_path)
    # else:
    #     trainer.train()
    trainer.train()
    trainer.save_model(os.path.join(output_dir, 'best_model'))
