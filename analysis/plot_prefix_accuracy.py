import json
import random
import matplotlib.pyplot as plt
import matplotlib
from analysis_utils import pres
import numpy as np
from tqdm import tqdm
import yaml
import argparse
from analysis_utils import confidence_interval



def load_generations(sweep_name, teach, temperature):
    generations_file = f"{eval_dir}/generations/{sweep_name}_{teach}_t{temperature}.json"
    decodings_file = f"{eval_dir}/generations/{sweep_name}_{teach}_greedy.json"
    with open(generations_file, 'r') as f:
        generations = json.load(f)

    with open(decodings_file, 'r') as f:
        decodings = json.load(f)

    return generations, decodings


def best_of_n_evidence_accuracy(generations, num_data, sequential):
    num_evidences = 0
    for i in range(num_data):
        evidence = False            
        for gen in generations:
            task = gen[i]
            if task['evidence'] == True and len(task['works']['work']) <= sequential:
                evidence = True
        if evidence:
            num_evidences += 1
    return num_evidences/num_data


def cost(sequential, parallel, budget_type):
    if budget_type == "cot":
        return sequential
    elif budget_type == "token":
        return sequential * parallel
    elif budget_type == "compute":
        return sequential * sequential * parallel
    else:
        raise ValueError("Invalid budget type")


def sequential_budget(parallel, budget_type, budget):
    res = 1
    while cost(res + 1, parallel, budget_type) <= budget:
        res += 1
    return res

def plot_within_budget_accuracy(sweep_name, teach, budget_type, temperature = 1.0):
    # budget = tokens_budget if budget_type == "token" else (cot_budget if budget_type == "cot" else compute_budget)

    if budget_type == "cot":
        min_seq_budget = 16
        budget = 50
        parallel_values = [2 ** i for i in range(5 + 1)]
    elif budget_type == "token":
        min_seq_budget = 1
        budget = 200
        parallel_values = range(1, 7)

    
    generations, decodings = load_generations(sweep_name, teach, temperature)

    num_data = len(generations[0])

    plt.figure(figsize=(8, 6))
    for parallel in parallel_values:
        # print(f"Parallel: {parallel}")    
        prefix_length = range(min_seq_budget, sequential_budget(parallel, budget_type, budget) + 1)
        accuracy = []
        for sequential in prefix_length:
            subset = [generations[i] for i in random.sample(range(len(generations)), parallel)]
            stochastic_accuracy = best_of_n_evidence_accuracy(subset, num_data, sequential)
            accuracy.append(stochastic_accuracy)
        x = [cost(sequential, parallel, budget_type) for sequential in prefix_length]        
        
        if parallel == 1:
            greedy_accuracy = np.array([best_of_n_evidence_accuracy([decodings], num_data, sequential) for sequential in prefix_length])
            greedy_accuracy = greedy_accuracy + (1 - greedy_accuracy) * 0.5
            cli_lower, cli_upper = confidence_interval(np.array(greedy_accuracy) * num_data, num_data)
            # plt.errorbar(x, greedy_accuracy, yerr=[greedy_accuracy - cli_lower, cli_upper - greedy_accuracy], label="Greedy", linestyle='--', color='#4682B4', elinewidth=1)
            plt.plot(x, greedy_accuracy, label="Greedy", linestyle='--', color='#4682B4')
        else:
            accuracy = accuracy + (1 - np.array(accuracy)) * 0.5
            cli_lower, cli_upper = confidence_interval(np.array(accuracy) * num_data, num_data)
            # plt.errorbar(x, accuracy, yerr=[accuracy - cli_lower, cli_upper - accuracy], label=pres(teach), elinewidth=1)
            plt.plot(x, accuracy, label=f"Best-of-{parallel}")

    plt.ylabel("Best-of-N Accuracy")
    if budget_type == "token":
        plt.title(f"Accuracy Across Total CoT Token Budgets")
        plt.xlabel("Total CoT Token Budget")
    else:
        plt.title(f"Accuracy Across Sequential Scales")
        plt.xlabel("Sequential Scale")

    plt.legend(loc='lower right')
    plt.grid(True)
    # plt.xticks(range(0, 31, 2))
    plt.tight_layout()
    plt.savefig(f"figures/{sweep_name}_{teach}_t{temperature}_prefix_{budget_type}_budget.png", dpi=300)
    print(f"Saved figure to figures/{sweep_name}_{teach}_t{temperature}_prefix_{budget_type}_budget.png")
    plt.savefig(f"figures/{sweep_name}_{teach}_t{temperature}_prefix_{budget_type}_budget.pdf", dpi=300, format='pdf')
    print(f"Saved figure to figures/{sweep_name}_{teach}_t{temperature}_prefix_{budget_type}_budget.pdf")


def plot_mixed_scaling(sweep_name, teach, temperature = 1.0):

    generations, decodings = load_generations(sweep_name, teach, temperature)
    num_data = len(generations[0])
    # parallel_values = [2 ** i for i in range(7 + 1)]
    exp = 6
    parallel_values = [i for i in range(1, 2**exp + 1)]
    seq_values = range(16, 40 + 1, 1)
    accuracies = []
    for parallel in tqdm(parallel_values):
        accuracy = []
        for sequential in seq_values:
            subset = [generations[i] for i in random.sample(range(len(generations)), parallel)]
            if parallel == 1:
                accuracy.append(best_of_n_evidence_accuracy([decodings], num_data, sequential))
            else:
                accuracy.append(best_of_n_evidence_accuracy(subset, num_data, sequential))
            
        accuracies.append(accuracy)
    # Make sure accuracies shape = (len(parallel_values), len(seq_values))
    accuracies = np.array(accuracies)
    accuracies = accuracies + (1 - accuracies) * 0.5
    # X, Y = np.meshgrid(seq_values, parallel_values)

    # Define extent to align pixel centers
    # dx = (seq_values[1] - seq_values[0]) / 2
    # dy = (parallel_values[1] - parallel_values[0]) / 2
    # extent = [seq_values[0] - dx, seq_values[-1] + dx, parallel_values[0] - dy, parallel_values[-1] + dy]

    # Plot heatmap first
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        accuracies,
        aspect='auto',
        cmap='viridis',
        origin='lower',
        # extent=extent,
        vmin=0.5, vmax=1.0  # or use np.min(accuracies), np.max(accuracies)
    )
    # ax.colorbar(label="Accuracy")
    # for i in range(len(parallel_values)):
    #     for j in range(len(seq_values)):
    #         value = accuracies[i, j]
    #         plt.text(
    #             seq_values[j],              # X-coordinate (Sequential)
    #             parallel_values[i],         # Y-coordinate (Parallel)
    #             f"{value:.2f}",             # Format value
    #             ha='center', va='center',   # Centered alignment
    #             fontsize=6, color='white' if value < 0.5 else 'black'
    #         )

    # Define levels and colors
    levels = [0.55, 0.6, 0.7, 0.8, 0.9]
    powers = np.array([2**i for i in [0,2,3,4,5,6]])
    ax.set_yticks(ticks=(powers-1), labels=powers)
    ax.set_xticks(ticks=range(len(seq_values))[::4], labels=seq_values[::4])
    fig.colorbar(im, label="Best-of-N Accuracy")
    # colors = ['white'] * (len(levels)) + ['red']  # Make 0.9 red

    # Plot contours
    contour = ax.contour(list(range(len(seq_values))), list(range(len(parallel_values))), accuracies, levels=levels, colors="white", linewidths=1.5)
    plt.clabel(contour, inline=True, fmt="%.2f")

    # Y-axis ticks
    # ytick_vals = [1] + [2**i for i in range(2, exp+1)]
    # ytick_labels = ['1 greedy'] + [str(v) for v in ytick_vals[1:]]
    # plt.yticks(ticks=ytick_vals, labels=ytick_labels)

    # X-axis ticks
    # plt.xticks(ticks=seq_values[::4])

    # Labels and save    
    ax.set_xlabel("Sequential Scale")
    ax.set_ylabel("Parallel Scale")
    ax.set_title(f"Parallel and Sequential Scaling of Transformer\n Trained on Graph Reasoning Task")
    fig.tight_layout()
    plt.savefig(f"figures/{sweep_name}_{teach}_t{temperature}_accuracy_contours.pdf", dpi=300, format='pdf')
    print(f"Saved figure to figures/{sweep_name}_{teach}_t{temperature}_accuracy_contours.pdf")


def plot_prefix_accuracies(sweep_name, teaches):
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(8, 6))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.axvline(x=depth * 5 + 1, color='red', linestyle='--', label="Longest Path", zorder=0, alpha=1)
    seq_values = range(depth * 3 + 1, 72 + 1, 1)
    for teach in teaches:
        decodings_file = f"{eval_dir}/generations/{sweep_name}_{teach}_greedy.json"
        with open(decodings_file, 'r') as f:
            decodings = json.load(f)
        num_data = len(decodings)            
        accuracy = []
        for sequential in seq_values:
            accuracy.append(best_of_n_evidence_accuracy([decodings], num_data, sequential))
        cli_lower, cli_upper = confidence_interval(np.array(accuracy) * num_data, num_data)
        eb = plt.errorbar(seq_values, accuracy, yerr=[accuracy - cli_lower, cli_upper - accuracy], label=pres(teach), elinewidth=1)
        # eb[-1][0].set_alpha(0.3)  # Set the alpha of the error bars
        # plt.plot(seq_values, accuracy, label=pres(teach))
    plt.ylabel("Evidence Accuracy")
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    # plt.title(f"Sequential Scaling of CoT Strategies on Bridge({depth})")
    plt.title(f"Sequential Scaling of CoT Strategies on Bridge({depth})")
    plt.xlabel("Sequential Scale")
    plt.xticks(ticks=seq_values[::4])

    handles, labels = plt.gca().get_legend_handles_labels()
    desired_order = [1 + i for i in range(len(teaches))] + [0]
    handles = [handles[i] for i in desired_order]
    labels = [labels[i] for i in desired_order]
    plt.legend(handles, labels)        
    # plt.grid(True)
    # plt.xticks(range(0, 31, 2))
    # plt.ylim(0, 0.8)
    plt.tight_layout()
    plt.savefig(f"figures/{sweep_name}_gd_prefix_accuracies.pdf", dpi=300, format='pdf')
    print(f"Saved figure to figures/{sweep_name}_gd_prefix_accuracies.pdf")

if __name__ == "__main__":
    matplotlib.rcParams.update({'font.size': 16})
    parser = argparse.ArgumentParser(description='Plot prefix accuracy')
    parser.add_argument('--depth', type=int, default=5, help='Depth of the model')
    parser.add_argument('--teaches', type=str, nargs='+', default=['dfs-pruned'], help='Teach types (cot, greedy)')
    parser.add_argument('--budget_type', type=str, default='token', help='Budget type (cot, token, compute)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    args = parser.parse_args()
    depth = args.depth
    budget_type = args.budget_type
    temperature = args.temperature
    with open(f"eval/config/base.yaml", 'r') as f:
        config = yaml.safe_load(f)
    with open(f"eval/config/d{depth}.yaml", 'r') as f:
        config.update(yaml.safe_load(f))
    eval_dir = config['eval_dir']
    sweep_name = config['sweep_name']
    teaches = args.teaches
    temperature = args.temperature
    for teach in teaches:
        plot_within_budget_accuracy(sweep_name, teach, budget_type, temperature)
        # plot_mixed_scaling(sweep_name, teach, temperature)
        break
    # plot_prefix_accuracies(sweep_name, teaches)