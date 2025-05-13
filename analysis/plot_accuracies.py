import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import re
import yaml
from analysis_utils import pres, confidence_interval, load_eval

# order = ["dfs", "walk-30", "walk-24", "walk-18", "path", "walk-12", "optimal"]
# order = ["dfs", "dfs-pruned", "walk-11", "walk-9", "walk-7", "path", "walk-5", "optimal"]

metrics_to_plot = ["evidence_accuracy", "decision_accuracy"]
strategies = ["dfs-pruned", "dfs",  "path", "optimal"]
num_eval = 5000





def plot_accuracies_by_depth(depths):
    decision_accuracies = {strategy: [] for strategy in strategies}
    evidence_accuracies = {strategy: [] for strategy in strategies}
    for depth in depths:
        df, sweep_name = load_eval(depth, strategies)
        for strategy in strategies:
            decision_accuracies[strategy].append(df[df['teach'] == strategy]['decision_accuracy'].values[0])
            evidence_accuracies[strategy].append(df[df['teach'] == strategy]['evidence_accuracy'].values[0])

    
    x_pos = np.arange(len(depths))
    for metric_to_plot in metrics_to_plot:
        plt.figure(figsize=(8, 6))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.title(f"(a) {pres(metric_to_plot)} of CoT Strategies on Bridge(d) Task")        
        
        for strategy in strategies:
            accuracies = decision_accuracies[strategy] if metric_to_plot == "decision_accuracy" else evidence_accuracies[strategy]
            # print(strategy, metric_to_plot, accuracies)
            ci_lower, ci_upper = confidence_interval(np.array(accuracies) * num_eval, num_eval)
            plt.errorbar(x_pos, accuracies, yerr=[np.array(accuracies) - ci_lower, ci_upper - np.array(accuracies)], label=pres(strategy))

        analytical_accuracy = 1/3 * np.array([1/4**(depth-1) for depth in depths])
        plt.plot(x_pos, analytical_accuracy, label=r"$P(\mathrm{DFS} \in D(\mathrm{S\text{-}Path}))$", color='red', linestyle='--')
        analytical_accuracy = 2/3 * np.array([1/2**(depth-1) for depth in depths])
        plt.plot(x_pos, analytical_accuracy, label=r"$P(\mathrm{DFS} \in D(\mathrm{Path}))$", color='green', linestyle='--')

        plt.xticks(x_pos, depths)
        plt.ylabel(pres(metric_to_plot))
        plt.xlabel("Depth of the Bridge Graph")
        handles, labels = plt.gca().get_legend_handles_labels()
        desired_order = [2 + i for i in range(len(strategies))] + [0, 1]
        handles = [handles[i] for i in desired_order]
        labels = [labels[i] for i in desired_order]
        plt.legend(handles, labels)        
        plt.tight_layout()
        plt.savefig(f"figures/accuracy_{sweep_name}_{metric_to_plot}.png", dpi=300)
        print(f"Figure saved to figures/accuracy_{sweep_name}_{metric_to_plot}.png")


def plot_accuracies_for_depth(depth):
    df, sweep_name = load_eval(depth, strategies)
    x = df["teach"]
    x_pos = np.arange(len(x))

    plt.figure(figsize=(4.5, 7))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for metric_to_plot in metrics_to_plot:  
        ci_lower, ci_upper = confidence_interval(df[metric_to_plot] * num_eval, num_eval)
        plt.errorbar(x_pos, df[metric_to_plot], yerr=[df[metric_to_plot] - ci_lower, ci_upper - df[metric_to_plot]], 
                    fmt='none', ecolor='black', capsize=0, alpha=1)
        color = "skyblue" if metric_to_plot == 'decision_accuracy' else "green"
        label = "Decision Accuracy" if metric_to_plot == 'decision_accuracy' else "Evidence Accuracy"
        plt.bar(x_pos, df[metric_to_plot], label=label, color=color, alpha=0.5, width=0.8, edgecolor='black')

    plt.xticks(x_pos, [pres(teach) for teach in x])
    plt.ylabel("Accuracy")
    plt.title(f"(b) Accuracy on Bridge({depth}) Task")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/accuracy_{sweep_name}.png", dpi=300)
    print(f"Figure saved to figures/accuracy_{sweep_name}.png")

depths = range(1, 6)
plot_accuracies_by_depth(depths)
for depth in depths:
    plot_accuracies_for_depth(depth)