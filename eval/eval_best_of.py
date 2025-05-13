import argparse
import torch
import os
from tqdm import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from eval import load_model, load_eval_data
from train import answer_questions, verify_answer, verify_aggregate, create_compute_metrics
from scipy.stats import mode
from transformers import Trainer, TrainingArguments
from mission import MissionDataCollator, Mission



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--training_config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--sweep_name', type=str, default='delta_flower_d2-s3-l5-b3_t8.5e6')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--log_num_trials', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)   
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed) 

    with open(args.training_config, 'r') as f:
        config = yaml.safe_load(f)

    tokenizer, dataset = load_eval_data(config)

    csv_path = f"csv/best_evidence_accuracy_{args.sweep_name}.csv"
    
    question_strs = [example['mission_strs'] for example in dataset]
    missions = [Mission(example, from_dict=True) for example in dataset]
    node_ids = [example['node_ids'] for example in dataset]

    num_trials = 2 ** args.log_num_trials
    iters = [2**i for i in range(args.log_num_trials + 1)]
    


    df = pd.read_csv(csv_path)
    columns = ['teach', 'name', 'run_id', 'sweep_id'] + [f'maj_{i}' for i in iters] + [f'any_{i}' for i in iters]
    results = []

    for index, row in df.iterrows():
        model_name = row['name']
        run_id, sweep_id = row['run_id'], row['sweep_id']
        print(f"Loading model: {model_name}...")
        model = load_model(config['output_dir'], sweep_id, model_name)
        model.config.max_length = config['context_length']
        if model is None:
            print(f"Model {model_name} not found. Skipping...")
            continue        

        n_answers_strs = [answer_questions(model, tokenizer, question_strs, do_sample=True, temperature=args.temperature) 
                          for _ in tqdm(range(num_trials))]

        decisions = [[] for _ in range(len(dataset))]
        evidence_acc = np.zeros(len(dataset))
        decision_results = []
        evidence_results = []
        for k in iters:
            decision_verdicts, evidence_verdicts = [], []
            for i, example in enumerate(dataset):
                answer_strs = [n_answers_strs[j][i] for j in range(k)]
                decision, evidence = verify_aggregate(missions[i], node_ids[i], answer_strs)
                decision_verdicts.append(decision)
                evidence_verdicts.append(evidence)
            decision_results.append(np.mean(decision_verdicts))
            evidence_results.append(np.mean(evidence_verdicts))

        results.append([row['teach'], model_name, run_id, sweep_id] + decision_results + evidence_results)
        print(results[-1])

    filename =f"csv/eval_k{num_trials}_t{args.temperature}_n{len(dataset)}_{args.sweep_name}.csv"
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(filename, index=False)


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
            decision, evidence, works = verify_answer(mission, node_ids, answer_str, return_works=True)
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
