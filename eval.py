import argparse
import torch
import os
import json
from tqdm import tqdm
import yaml
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import wandb
from datasets import load_dataset
from collections import defaultdict
from ast import literal_eval
from transformers import MistralForCausalLM
from mission import Mission, MissionTokenizer
from utils import get_missions_vocab, gen_mission_str
from train import answer_questions, verify_answer



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
    # patient_model_dir = os.path.join(args.output_dir, args.ckpt)
    model = MistralForCausalLM.from_pretrained(model_dir)
    # patient_model = MistralForCausalLM.from_pretrained(patient_model_dir)
    model.eval()
    model.cuda()
    # patient_model.eval()
    # patient_model.cuda()

    num_node_tokens = config['num_node_tokens']
    context_length = config['context_length']
    tokenizer = MissionTokenizer(get_missions_vocab(num_node_tokens))
    tokenizer.model_max_length = context_length
    tokenizer.padding_side = 'left'     
    model.config.max_length = context_length

    data_file = os.path.join(args.data_dir, args.data_file)
    dataset = load_dataset("json", data_files={'val':data_file})
    # print(dataset)

    # tokenize dataset
    def tokenize(example):
        mission = Mission(example, from_dict=True)
        node_ids = np.random.permutation(num_node_tokens)[:mission.number_of_nodes()]
        mission_str = gen_mission_str(mission, node_ids=node_ids)
        token_ids = tokenizer.encode(mission_str)
        return {'input_ids': token_ids, 'mission_strs': mission_str, 'node_ids': node_ids} 

    dataset = dataset.map(tokenize, batched=False)['val'] #, remove_columns=datasets["train"].column_names)
    # print(dataset)

    num_trials = 101
    output_every = 20
    iters = list(range(0, num_trials, output_every))
    results = []

    # csv
    df = pd.read_csv("Highest_Evidence_Accuracy_per_Teach.csv")
    output_filename = "Highest_Evidence_Best-of-k_Results.csv"
    

    for model_name in df["Name"]:
        print(f"Loading model: {model_name}...")
        model_dir = os.path.join(args.output_dir, model_name, 'checkpoint-10000')
        print(model_dir)
        # patient_model_dir = os.path.join(args.output_dir, args.ckpt)
        model = MistralForCausalLM.from_pretrained(model_dir)
        # patient_model = MistralForCausalLM.from_pretrained(patient_model_dir)
        model.eval()
        model.cuda()
        model.config.max_length = context_length
        # decision_acc = np.zeros(len(dataset))
        evidence_acc = np.zeros(len(dataset))
        model_results = []
        for t in tqdm(range(num_trials)):
            question_strs = [example['mission_strs'] for example in dataset]
            answer_strs = answer_questions(model, tokenizer, question_strs, do_sample=True, temperature=0.5, top_p=0.9)
            for i, example in enumerate(dataset):
                mission = Mission(example, from_dict=True)
                node_ids = example['node_ids']                    
                question_str, answer_str = question_strs[i], answer_strs[i]        
                evidence, decision = verify_answer(mission, node_ids, answer_str)
                # decision_acc[i] |= decision
                if evidence:
                    evidence_acc[i] = 1
            if t%output_every == 0:
                model_results.append(np.mean(evidence_acc))
        results.append([model_name] + model_results)
    columns = ['Model Name'] + [f'best_of_{i+1}' for i in iters]
    results_df = pd.DataFrame(results, columns=columns)
    output_filename = "Best-of-k_Results.csv"
    results_df.to_csv(output_filename, index=False)


    exit(0)
    # dec_acc = defaultdict(list)
    # ev_acc = defaultdict(list)    
    max_worklen = 100
    # dec_correct = np.zeros((context_length, num_trials))
    # ev_correct = np.zeros((context_length, num_trials))
    plt.figure(figsize=(8, 5))
    best_worklen_dec = np.full(len(dataset), -1)
    best_worklen_evi = np.full(len(dataset), -1)
    ans_worklen = np.full((len(dataset), num_trials), -1)
    
    for t in tqdm(range(num_trials)):
        question_strs = [example['mission_strs'] for example in dataset]
        answer_strs = answer_questions(model, tokenizer, question_strs, do_sample=True, temperature=0.5, top_p=0.9)
        for i, example in enumerate(dataset):
            mission = Mission(example, from_dict=True)
            node_ids = example['node_ids']                    
            question_str, answer_str = question_strs[i], answer_strs[i]        
            evidence, decision, works = verify_answer(mission, node_ids, answer_str, return_works=True)
            worklen = len(works) - 1
            if decision:
                ans_worklen[i, t] = worklen
            if decision and (worklen < best_worklen_dec[i] or best_worklen_dec[i] == -1):
                best_worklen_dec[i] = worklen                
            if evidence and (worklen < best_worklen_evi[i] or best_worklen_evi[i] == -1):
                best_worklen_evi[i] = worklen
            
            # mistakes = np.sum(ans_worklen[i, :t+1] == -1)
            # if mistakes > 0 and mistakes < t+1:
            #     print(i)
            #     print(f'Question: {question_str}')
            #     print(f'Answer: {answer_str[:100]}')
            #     print(f'Correct: {node_ids[mission.target]}')
            #     probs = F.softmax(scores[i], dim=-1)
            #     max_prob, max_idx = torch.topk(probs, k=5, dim=-1)    
            #     print(max_prob.cpu().numpy())
            #     print(max_idx.cpu().numpy())
            #     # print(f'Decision: {decision}')
            #     # print(f'Evidence: {evidence}')
            #     # print(f'Works: {works}')
            #     # visualize_tree(mission.graph, mission.start, node_ids)
            #     input('Press Enter to continue...')
            # optimal = [node_ids[u] for u in literal_eval(example['optimal'])]
            # dec_acc[len(optimal)-1].append(decision)
            # ev_acc[len(optimal)-1].append(evidence)
            # if i < 5:
 
        if t % print_every == 0:
            dec_acc = np.zeros(context_length)
            evi_acc = np.zeros(context_length)
            for i in range(len(dataset)):
                if best_worklen_dec[i] != -1:
                    dec_acc[best_worklen_dec[i]] += 1
                if best_worklen_evi[i] != -1:
                    evi_acc[best_worklen_evi[i]] += 1
            
            for i in range(1, context_length):
                # if dec_acc[i] or evi_acc[i]:
                #     max_worklen = i
                dec_acc[i] += dec_acc[i-1]
                evi_acc[i] += evi_acc[i-1]
            dec_acc /= len(dataset)
            evi_acc /= len(dataset)
            # plt.plot(range(max_worklen+1), dec_acc[:max_worklen+1], linestyle='-', label=f"best of {t+1} decision")
            plt.plot(range(max_worklen+1), evi_acc[:max_worklen+1], linestyle='-', label=f"best of {t+1} evidence")

    # example_error = np.mean(ans_worklen == -1, axis=-1)
    # plt.hist(example_error, bins=num_trials+1, density=False, alpha=0.6, color='r')
    # plt.xlabel('Error')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of the Error per Example')

    # for l in sorted(dec_acc.keys()):
    #     print(f'len: {l}, dec_acc: {np.mean(dec_acc[l])}, ev_acc: {np.mean(ev_acc[l])}')
    # max_works_len = 0
    # for l in range(1, context_length):
    #     if np.any(dec_correct[l]) or np.any(ev_correct[l]):
    #         max_works_len = l
    #     dec_correct[l] += dec_correct[l-1]
    #     ev_correct[l] += ev_correct[l-1]

    
        # print(f'len: {l}, dec_prefix_acc: {dec_sum[l]/len(dataset)}, ev_prefix_acc: {ev_sum[l]/len(dataset)}')
    # for l in sorted(patient_dec_acc.keys()):
    #     print(f'len: {l}, patient_dec_acc: {np.mean(patient_dec_acc[l])}, patient_ev_acc: {np.mean(patient_ev_acc[l])}')
    plt.xlabel("Max Work Length")
    plt.ylabel(f"Best of up to {num_trials} Accuracy")
    plt.title(f"Best of up to {num_trials} Accuracy vs Max Work Length")
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, max_worklen+1, 10))
    plt.tight_layout()
    
    wandb.init(project="bepatient", entity="seyedparsa", name="best-of-k", resume="allow")
    wandb.log({"work_length": wandb.Image(plt)})
    print('Done!')
