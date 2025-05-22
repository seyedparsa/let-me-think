from analysis_utils import load_eval, pres, confidence_interval, normal_confidence_interval
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

stages = ["path", "path-play-1", "path-play-2", "path-play-3", "path-play-4"]
depths = [1, 2 , 3, 4, 5]
num_eval = 5000


def plot_by_iteration(metric, accuracies=True):
    plt.figure(figsize=(8, 6))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    x_pos = np.arange(len(stages))
    for d in depths:
        df, sweep_name = load_eval(d, stages)
        values = []
        for stage in stages:
            if stage in df['teach'].values:                
                if metric.startswith("avg_"):
                    values.append((df[df['teach'] == stage][metric].values[0], df[df['teach'] == stage][metric.replace("avg", "std")].values[0]))
                else:
                    values.append(df[df['teach'] == stage][metric].values[0])    
        if accuracies:
            ci_lower, ci_upper = confidence_interval(np.array(values) * num_eval, num_eval)
            plt.errorbar(x_pos, values, yerr=[np.array(values) - ci_lower, ci_upper - np.array(values)], label=f"d={d}")
        else:
            means, stds = zip(*values)
            means = np.array(means)
            stds = np.array(stds)
            ci_lower, ci_upper = normal_confidence_interval(means, stds, num_eval)
            plt.errorbar(x_pos, means, yerr=[means - ci_lower, ci_upper - means], label=f"d={d}")
    plt.title(f"{pres(metric)} by RL Iteration")
    plt.xlabel("RL Iteration")
    plt.ylabel(pres(metric))
    plt.xticks(range(len(stages)), range(len(stages)))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/rl_{metric}.png", dpi=300)
    print(f"Figure saved to figures/rl_{metric}.png")


def plot_all_by_iteration():
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey='row', sharex=True)

    x_pos = np.arange(len(stages))

    # These will hold data for each plot
    decision_data = []
    evidence_data = []
    valid_data = []
    verified_data = []

    for d in depths:
        df, sweep_name = load_eval(d, stages)
        evidence_accuracy = []
        decision_accuracy = []
        valid_thought_lengths = []
        verified_thought_lengths = []

        for stage in stages:
            if stage in df['teach'].values:
                evidence_accuracy.append(df[df['teach'] == stage]['evidence_accuracy'].values[0])
                decision_accuracy.append(df[df['teach'] == stage]['decision_accuracy'].values[0])
                valid_thought_lengths.append((df[df['teach'] == stage]['avg_valid_thoughts'].values[0], df[df['teach'] == stage]['std_valid_thoughts'].values[0]))
                verified_thought_lengths.append((df[df['teach'] == stage]['avg_verified_thoughts'].values[0], df[df['teach'] == stage]['std_verified_thoughts'].values[0]))                

        decision_data.append((decision_accuracy, d))
        evidence_data.append((evidence_accuracy, d))
        valid_data.append((valid_thought_lengths, d))
        verified_data.append((verified_thought_lengths, d))

    # Plot 1: Decision Accuracy (Top Left)
    ax = axs[0, 0]
    for data, d in evidence_data:
        ci_lower, ci_upper = confidence_interval(np.array(data) * num_eval, num_eval)
        ax.errorbar(x_pos, data, yerr=[np.array(data) - ci_lower, ci_upper - np.array(data)], label=f"d={d}")
    ax.set_title("(a) Evidence Accuracy")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(stages)))
    ax.set_ylabel("Accuracy")
    ax.legend()

    # Plot 2: Evidence Accuracy (Top Right)
    ax = axs[0, 1]
    for data, d in decision_data:
        ci_lower, ci_upper = confidence_interval(np.array(data) * num_eval, num_eval)
        ax.errorbar(x_pos, data, yerr=[np.array(data) - ci_lower, ci_upper - np.array(data)], label=f"d={d}")
    ax.set_title("(b) Decision Accuracy")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(stages)))
    # ax.legend()

    # Plot 3: Valid Thought Lengths (Bottom Left)
    ax = axs[1, 0]
    for data, d in valid_data:
        means, stds = zip(*data)
        means = np.array(means)
        stds = np.array(stds)
        ci_lower, ci_upper = normal_confidence_interval(means, stds, num_eval)
        ax.errorbar(x_pos, means, yerr=[means - ci_lower, ci_upper - means], label=f"d={d}")
    ax.set_title("(c) Average Length of CoTs")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(stages)))
    ax.set_ylabel("CoT Length")
    ax.set_xlabel("RL Iteration")
    # ax.legend()

    # Plot 4: Verified Thought Lengths (Bottom Right)
    ax = axs[1, 1]
    for data, d in verified_data:
        means, stds = zip(*data)
        means = np.array(means)
        stds = np.array(stds)
        ci_lower, ci_upper = normal_confidence_interval(means, stds, num_eval)
        ax.errorbar(x_pos, means, yerr=[means - ci_lower, ci_upper - means], label=f"d={d}")
    ax.set_title("(d) Average Length of Verified CoTs")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(stages)))
    ax.set_xlabel("RL Iteration")
    # ax.legend()

    plt.tight_layout()
    plt.savefig("figures/rl_all_metrics.pdf", dpi=300, format='pdf')
    print("Figure saved to figures/rl_all_metrics.pdf")



# plot_by_iteration("decision_accuracy")
# plot_by_iteration("evidence_accuracy")
# # plot_by_iteration("avg_thoughts", False)
# plot_by_iteration("avg_valid_thoughts", False)
# plot_by_iteration("avg_verified_thoughts", False)

plot_all_by_iteration()

# x = np.arange(1, 6)
# bar_width = 0.18
# offsets = np.linspace(-bar_width*1.5, bar_width*1.5, len(stages))

# plt.figure()
# for i, stage in enumerate(stages):
#     plt.bar(x + offsets[i], accuracies[stage], width=bar_width, label=stage)
# plt.title("Evidence Accuracy by Stage")
# plt.xlabel("d")
# plt.ylabel("Evidence Accuracy")
# plt.ylim(0, 1)
# plt.xticks(x)
# plt.legend()
# plt.tight_layout()
# plt.savefig("figures/accuracy_stages.png", dpi=300)
# print("Figure saved to figures/accuracy_stages.png")