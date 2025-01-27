import argparse
import torch
import os
import json
import yaml
import numpy as np
import networkx as nx
from datasets import load_dataset
from collections import defaultdict
from ast import literal_eval
from transformers import MistralForCausalLM
from mission import Mission, MissionTokenizer
from utils import get_missions_vocab, gen_mission_str
from train import answer_questions, eval_steps

def visualize_tree(tree, u, node_ids, pars=[]):
    print('\t'*len(pars), node_ids[u])
    for v in tree[u]:
        if v not in pars:
            visualize_tree(tree, v, node_ids, pars + [u])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--ckpt', type=str, help='path to the model checkpoint')
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--data_file', type=str, help='data file')
    parser.add_argument('--training_config', type=str, default='configs/training_config.yaml')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    with open(args.training_config, 'r') as f:
        config = yaml.safe_load(f)

    model_dir = os.path.join(args.output_dir, args.ckpt)
    patient_model_dir = os.path.join(args.output_dir, args.ckpt.replace('optimal', 'search'))
    model = MistralForCausalLM.from_pretrained(model_dir)
    patient_model = MistralForCausalLM.from_pretrained(patient_model_dir)
    model.eval()
    model.cuda()
    patient_model.eval()
    patient_model.cuda()

    num_node_tokens = config['num_node_tokens']
    context_length = config['context_length']
    tokenizer = MissionTokenizer(get_missions_vocab(num_node_tokens))
    tokenizer.model_max_length = context_length
    tokenizer.padding_side = 'left'      

    data_file = os.path.join(args.data_dir, args.data_file)
    dataset = load_dataset("json", data_files={'val':data_file})
    print(dataset)

    # tokenize dataset
    def tokenize(example):
        mission = Mission(example, from_dict=True)
        node_ids = np.random.permutation(num_node_tokens)[:mission.number_of_nodes()]
        mission_str = gen_mission_str(mission, node_ids=node_ids)
        token_ids = tokenizer.encode(mission_str)
        return {'input_ids': token_ids, 'mission_strs': mission_str, 'node_ids': node_ids} 

    dataset = dataset.map(tokenize, batched=False)['val'] #, remove_columns=datasets["train"].column_names)
    print(dataset)

    question_strs = [example['mission_strs'] for example in dataset]
    answer_strs = answer_questions(model, tokenizer, question_strs)
    patient_answer_strs = answer_questions(patient_model, tokenizer, question_strs)
    dec_acc = defaultdict(list)
    ev_acc = defaultdict(list)
    patient_dec_acc = defaultdict(list)
    patient_ev_acc = defaultdict(list)
    for i, example in enumerate(dataset):
        mission = Mission(example, from_dict=True)
        node_ids = example['node_ids']                    
        question_str, answer_str, patient_answer_str = question_strs[i], answer_strs[i], patient_answer_strs[i]
        evidence, decision = eval_steps(mission, node_ids, answer_str)
        patient_evidence, patient_decision = eval_steps(mission, node_ids, patient_answer_str)
        optimal = [node_ids[u] for u in literal_eval(example['optimal'])]        
        dec_acc[len(optimal)-1].append(decision)
        ev_acc[len(optimal)-1].append(evidence)
        patient_dec_acc[len(optimal)-1].append(patient_decision)
        patient_ev_acc[len(optimal)-1].append(patient_evidence)

    for l in sorted(dec_acc.keys()):
        print(f'len: {l}, dec_acc: {np.mean(dec_acc[l])}, ev_acc: {np.mean(ev_acc[l])}')
    for l in sorted(patient_dec_acc.keys()):
        print(f'len: {l}, patient_dec_acc: {np.mean(patient_dec_acc[l])}, patient_ev_acc: {np.mean(patient_ev_acc[l])}')
