import argparse
import torch
import os
import yaml
import json
import numpy as np
import pandas as pd
from mission import Mission
from train import create_compute_metrics, answer_questions, verify_aggregate
from tqdm import tqdm
from eval_utils import load_model, load_data
from train import verify_answer


def set_seed(seed):
    torch.manual_seed(seed)   
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def sample_generations(model, tokenizer, dataset, question_strs, missions, node_ids, base_seed, offset, n_samples, temperature=1.0):    
    generations = []
    for t in tqdm(range(n_samples)):
        generation = []
        set_seed(base_seed + offset + t)
        answer_strs = answer_questions(model, tokenizer, question_strs, do_sample=True, temperature=temperature)
        for i in range(len(dataset)):
            decision, evidence, works = verify_answer(missions[i], node_ids[i], answer_strs[i], return_works=True)
            generation.append({
                'id': i,
                'answer_str': answer_strs[i],
                'decision': decision,
                'evidence': evidence,
                'works': works,
            })
        generations.append(generation)
    return generations


def greedy_decoding(model, tokenizer, dataset, question_strs, missions, node_ids, base_seed):
    set_seed(base_seed)
    answer_strs = answer_questions(model, tokenizer, question_strs, do_sample=False)
    decodings = []
    for i in range(len(dataset)):
        decision, evidence, works = verify_answer(missions[i], node_ids[i], answer_strs[i], return_works=True)
        decodings.append({
            'id': i,
            'answer_str': answer_strs[i],
            'decision': decision,
            'evidence': evidence,
            'works': works,
        })
    return decodings
    


def load_generations(model, tokenizer, dataset, question_strs, missions, node_ids, base_seed, n_gens, temperature, filename, reset=False):
    generations = []
    if not reset and os.path.exists(filename):
        with open(filename, 'r') as f:
            generations = json.load(f)
    if n_gens == -1:
        return generations
    if len(generations) >= n_gens:
        print(f"Loaded {len(generations)} generations from {filename}.")
        return generations[:n_gens]
    n_samples = n_gens - len(generations)
    print(f"Generating {n_samples} more samples...")
    generations.extend(sample_generations(model, tokenizer, dataset, question_strs, missions, node_ids, base_seed, len(generations), n_samples, temperature))
    with open(filename, 'w') as f:
        json.dump(generations, f, indent=2)
    print(f"Saved {len(generations)} generations to {filename}.")
    return generations


def load_decodings(model, tokenizer, dataset, question_strs, missions, node_ids, base_seed, filename, reset=False):    
    if not reset and os.path.exists(filename):
        with open(filename, 'r') as f:
            decodings = json.load(f)
            return decodings
    decodings = greedy_decoding(model, tokenizer, dataset, question_strs, missions, node_ids, base_seed)
    with open(filename, 'w') as f:
        json.dump(decodings, f, indent=2)
    print(f"Saved decodings to {filename}.")
    return decodings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--models_file', type=str)
    parser.add_argument('--teaches', type=str, nargs='+')
    parser.add_argument('--n_gens', type=int)
    parser.add_argument('--reset', action='store_true')
    
    args = parser.parse_args()
    set_seed(args.seed)   
    with open('eval/config/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open(f'eval/config/d{args.depth}.yaml', 'r') as f:
        config.update(yaml.safe_load(f))
    
    tokenizer, dataset = load_data(config)
    question_strs = [example['mission_strs'] for example in dataset]
    missions = [Mission(example, from_dict=True) for example in dataset]
    node_ids = [example['node_ids'] for example in dataset]

    sweep_name = config['sweep_name']
    metric_name = config['metric_for_best_model']['name']
    models_file = args.models_file    
    if models_file is None:
        models_file = f"csv/sweeps/{sweep_name}_best_{metric_name}.csv"
    df = pd.read_csv(models_file)
    df = df.drop(columns=['loss'])
    teaches_to_gen = args.teaches
    if teaches_to_gen is None:
        teaches_to_gen = df['teach'].unique()
    
    print(df)
    print(f"Sampling models with teaches: {teaches_to_gen}")
    updates = []
    for index, row in df.iterrows():
        model_name = row['name']
        sweep_id = row['sweep_id']
        teach = row['teach']
        if teach not in teaches_to_gen:
            continue
        print(f"\nLoading model: {row['teach']}")
        model = load_model(config['output_dir'], sweep_id, model_name)
        model.config.max_length = config['context_length']
        if model is None:
            print(f"Model {model_name} not found. Skipping...")
            continue    
        filename = f"{config['eval_dir']}/generations/{sweep_name}_{teach}"
        load_generations(model, tokenizer, dataset, question_strs, missions, node_ids, args.seed, args.n_gens, args.temperature, f"{filename}_t{args.temperature}.json", args.reset)
        load_decodings(model, tokenizer, dataset, question_strs, missions, node_ids, args.seed, f"{filename}_greedy.json", args.reset)