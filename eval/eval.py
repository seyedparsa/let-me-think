import argparse
import torch
import os
import yaml
import numpy as np
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from transformers import MistralForCausalLM
from mission import Mission, MissionTokenizer
from utils import get_missions_vocab, gen_mission_str
from train import create_compute_metrics, answer_questions, verify_aggregate, verify_answer
import json
import glob  
from transformers import Trainer, TrainingArguments
from mission import MissionDataCollator
from tqdm import tqdm
from sample import load_generations, load_decodings, set_seed
from eval_utils import load_model, load_data
from torch.utils.data import DataLoader



def evaluate_model(model, tokenizer, dataset, config, do_sample=False, temperature=1.0):
    evaluation_args = TrainingArguments(
            output_dir="./tmp-eval",
            per_device_eval_batch_size=config['batch_size'],            
            report_to=[],
            fp16=True,
        )
    if do_sample:
        compute_metrics = create_compute_metrics(model, tokenizer, dataset, do_sample=True, temperature=temperature)
    else:
        compute_metrics = create_compute_metrics(model, tokenizer, dataset, do_sample=False)
    trainer = Trainer(
            model=model,
            args=evaluation_args,
            data_collator=MissionDataCollator(tokenizer, mlm=False),
            eval_dataset=dataset,
            compute_metrics=compute_metrics,
        )
    metrics = trainer.evaluate()
    return metrics


def play_model(model, tokenizer, dataset, teach, do_sample, temperature, data_dir, playground):
    print('Let\'s play!', flush=True)
    bteach, num_play = teach, 0
    if "play" in teach:
        bteach, _, num_play = teach.split("-")
    new_teach = f'{bteach}-play-{int(num_play)+1}'    
    batch_size = 1000
    play_dataset = []
    for s in tqdm(range(0, len(dataset), batch_size)):
        t = min(s + batch_size, len(dataset))
        question_strs = []
        missions = []
        node_ids = []
        for i in range(t-s):
            example = dataset[s+i]
            question_strs.append(example['mission_strs'])
            missions.append(Mission(example, from_dict=True))
            node_ids.append(example['node_ids'])
        answer_strs = answer_questions(model, tokenizer, question_strs, do_sample=do_sample, temperature=temperature)
        for i in range(t-s):
            decision, evidence, works = verify_answer(missions[i], node_ids[i], answer_strs[i], return_works=True)
            if evidence:        
                keys = ['graph_type', 'task_type', 'graph', 'start', 'target', 'phrag', 'tegrat']
                play_data = {key: dataset[i][key] for key in keys}                    
                play_data[new_teach] = str(works['work'])
                play_dataset.append(play_data)
    
    play_file = os.path.join(data_dir, f'{new_teach}_n{len(play_dataset)}_on_{playground}')
    with open(play_file, 'w') as f:
        json.dump(play_dataset, f, indent=4)
    print(f"Play dataset of size {len(play_dataset)} saved to {play_file}")


def scale_until_learns(model, tokenizer, dataset, question_strs, missions, node_ids, temperature, accuracy_threshold, filename, seq_budget=None):
    set_seed(args.seed + args.offset)
    n_answers_strs = []
    n_works = []
    generations = load_generations(model, tokenizer, dataset, question_strs, missions, node_ids, args.seed, -1, temperature, filename)        
    generations = np.random.permutation(generations)
    for generation in generations:
        n_answers_strs.append([output['answer_str'] for output in generation])
        n_works.append([output['works'] for output in generation])
    per_example_answer_strs = []
    per_example_works = []
    for i in range(len(dataset)):
        per_example_answer_strs.append([n_answers_strs[j][i] for j in range(len(generations))])
        per_example_works.append([n_works[j][i] for j in range(len(generations))])
    l, r = 0, len(n_answers_strs) + 1
    while r - l > 1:
        n = (l + r) // 2
        print(f"Scaling to {n} answers")
        decision_verdicts, evidence_verdicts = [], []
        for i in range(len(dataset)):
            answer_strs = per_example_answer_strs[i][:n]
            works = per_example_works[i][:n]
            decision, evidence = verify_aggregate(missions[i], node_ids[i], answer_strs, works_list=works, seq_budget=seq_budget)
            decision_verdicts.append(decision)
            evidence_verdicts.append(evidence)
        decision_accuracy = np.mean(decision_verdicts)
        evidence_accuracy = np.mean(evidence_verdicts)
        # avg_thoughts = np.mean(n_answers_avgs)        
        # metric[f"avg_thoughts_{len(n_answers_strs)}"] = avg_thoughts
        print(f"Decision accuracy: {decision_accuracy}, Evidence accuracy: {evidence_accuracy}")
        if decision_accuracy >= accuracy_threshold or evidence_accuracy >= accuracy_threshold:            
            r = n
        else:
            l = n
    if r == len(n_answers_strs) + 1:
        raise ValueError(f"Model did not learn with {len(n_answers_strs)} generations")
    return r


def evaluate_scaling(model, tokenizer, dataset, question_strs, missions, node_ids, temperature, log_num_trials, filename):
    num_trials = 2 ** log_num_trials
    if filename:
        generations = load_generations(model, tokenizer, dataset, question_strs, missions, node_ids, args.seed, num_trials, temperature, filename)        
        n_answers_strs = []
        for generation in generations[:num_trials]:
            n_answers_strs.append([output['answer_str'] for output in generation])
    else:
        n_answers_strs = [answer_questions(model, tokenizer, question_strs, do_sample=True, temperature=temperature) for _ in tqdm(range(num_trials), desc="Generating answers")]
    #                       for _ in tqdm(range(num_trials), desc="Generating answers")]  
    # n_answers_tokens = [answer_questions(model, tokenizer, question_strs, do_sample=True, return_tokens=True, temperature=temperature) 
    #                       for _ in tqdm(range(num_trials), desc="Generating answers")]
    # n_answers_strs = [tokenizer.batch_decode(answers_tokens) for answers_tokens in n_answers_tokens]
    # n_answers_lengths = np.array([torch.sum((answers_tokens != tokenizer.pad_token_id) & 
    #                                 (answers_tokens != tokenizer.eos_token_id), dim=-1).cpu().numpy()
    #                                   for answers_tokens in n_answers_tokens])
    metric = {}
    for k in tqdm(range(log_num_trials + 1)):
        decision_verdicts, evidence_verdicts = [], []
        for i, example in enumerate(dataset):
            answers_strs = [n_answers_strs[j][i] for j in range(2 ** k)]
            decision, evidence = verify_aggregate(missions[i], node_ids[i], answers_strs)
            decision_verdicts.append(decision)
            evidence_verdicts.append(evidence)
            # count number of words separated by spaces in answer_strs
        metric[f"maj_{2**k}"] = np.mean(decision_verdicts)
        metric[f"any_{2**k}"] = np.mean(evidence_verdicts)
        # metric[f"avg_thoughts_{2**k}"] = np.mean(n_answers_lengths[:2 ** k])    
    return metric    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--play', action='store_true')
    parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--log_trials', type=int, default=0)
    parser.add_argument('--until_learns', action='store_true')
    parser.add_argument('--accuracy_threshold', type=float, default=0.90)
    parser.add_argument('--models_file', type=str)
    parser.add_argument('--teaches', type=str, nargs='+')
    args = parser.parse_args()
    torch.manual_seed(args.seed)   
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed) 

    print(args.depth)
    with open('eval/config/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open(f'eval/config/d{args.depth}.yaml', 'r') as f:
        config.update(yaml.safe_load(f))

    tokenizer, dataset = load_data(config, 'play_file' if args.play else 'val_file')
    print(f"Loaded {len(dataset)} examples from {config['val_file']}")
    if args.scale:
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
    if teaches_to_eval is None:
        teaches_to_eval = df['teach'].unique()
    print(df)
    print(df.columns)
    print(f"Evaluating models with teaches: {teaches_to_eval}")
    
    results = []
    for index, row in df.iterrows():
        model_name = row['name']
        sweep_id = row['sweep_id']
        teach = row['teach']
        if teach not in teaches_to_eval:
            continue
        print(f"\nLoading model: {teach}")
        model = load_model(config['output_dir'], sweep_id, model_name)
        model.config.max_length = config['context_length']
        if model is None:
            print(f"Model {model_name} not found. Skipping...")
            continue     
        metrics = {
            'teach': teach,
            'name': model_name,
            'run_id': row['run_id'],
            'sweep_id': sweep_id,
            'temperature': args.temperature,
            'seed': args.seed,
            'offset': args.offset,
        }            
        if args.scale:
            gens_file = f"{config['eval_dir']}/generations/{sweep_name}_{teach}_t{args.temperature}.json"
            if args.until_learns:
                n_trials = scale_until_learns(model, tokenizer, dataset, question_strs, missions, node_ids, args.temperature, accuracy_threshold=args.accuracy_threshold, filename=gens_file)
                metrics[f'reach{args.accuracy_threshold}'] = n_trials                
            else:
                metrics = evaluate_scaling(model, tokenizer, dataset, question_strs, missions, node_ids, args.temperature, args.log_trials, filename=gens_file)
        elif args.play:            
            play_model(model, tokenizer, dataset, teach, do_sample=args.sample, temperature=args.temperature, data_dir=config['data_dir'], playground=config['play_file'])
        else:
            metrics = evaluate_model(model, tokenizer, dataset, config, do_sample=args.sample, temperature=args.temperature)
        metrics = {key.replace("eval_", "") : value for key, value in metrics.items()}
        print(metrics)
        eval_row = {'teach' : teach, 'name' : model_name, 'sweep_id' : sweep_id, 'run_id' : row['run_id']}
        key_prefixes = ['avg', 'std', 'maj', 'any']
        eval_row.update({key : value for key, value in metrics.items() if key in df.columns or any(key.startswith(prefix) for prefix in key_prefixes)})
        print(eval_row)
        # print(eval_row)        
        results.append(eval_row)
        # for key, value in metrics.items():
        #     key_col = key.replace("eval_", "")
        #     if args.scale or key_col in df.columns or key_col.startswith("avg_"):
        #         updates.append((index, key_col, value))
    
    # Apply updates to the DataFrame all at once
    # for index, key_col, value in updates:
    #     if key_col not in df.columns:
    #         df[key_col] = np.nan
    #     df.loc[index, key_col] = value
    df = pd.DataFrame(results)
    file_name = models_file.replace("sweeps", "evals").replace(".csv", "_")
    if args.scale:
        if args.until_learns:
            file_name += f"learns_"
        else:
            file_name += f"sc{2**args.log_trials}_"
    if args.sample:
        file_name += f"t{args.temperature}_"
    else:
        file_name += f"gd_"
    file_name += f"n{len(dataset)}.csv"
    if args.scale and args.until_learns and os.path.exists(file_name):
        df.to_csv(file_name, index=False, mode='a', header=False)
    else:
        df.to_csv(file_name, index=False)
        
    print(f"CSV saved to {file_name}")
    