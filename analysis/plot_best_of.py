import pandas as pd
import numpy as np
from analysis_utils import confidence_interval, pres
import matplotlib.pyplot as plt
import argparse
import yaml

order = ["dfs-pruned", "dfs", "path", "optimal"]

def compare_models(log_num_trial, temperature, num_eval, sweep_name, depth):
     # Construct the file path
     file_path = f"csv/evals/{sweep_name}_best_evidence_accuracy_sc{2 ** log_num_trial}_t{temperature}_n{num_eval}.csv"

     # Read the CSV file
     df = pd.read_csv(file_path)
     df = df[df['teach'].isin(order)]
     df['order'] = df['teach'].apply(lambda x: order.index(x))
     df = df.sort_values(by='order', ascending=True)
     df.reset_index(drop=True, inplace=True)

     # Extract the columns for 'maj' and 'any'
     maj_columns = [f'maj_{2 ** i}' for i in range(log_num_trial + 1)]
     any_columns = [f'any_{2 ** i}' for i in range(log_num_trial + 1)]
     # Create subplots for maj_values and any_values
     fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
     # Set x-axis labels as 2**k for both subplots
     axes[0].set_xticks(np.arange(log_num_trial + 1))
     axes[0].set_xticklabels([f'{2**i}' for i in range(log_num_trial + 1)])
     axes[1].set_xticks(np.arange(log_num_trial + 1))
     axes[1].set_xticklabels([f'{2**i}' for i in range(log_num_trial + 1)])
     for _, model_row in df.iterrows():
          method = model_row['name'].split('_')[0]        
          maj_values = model_row[maj_columns].values.flatten()
          any_values = model_row[any_columns].values.flatten()

          # Plot maj_values without markers
          line_maj, = axes[0].plot(np.arange(log_num_trial + 1), maj_values, 
               label=f"{pres(method)}")
          # Add error bars for maj_values
          ci_lower, ci_upper = confidence_interval(maj_values * num_eval, num_eval)
          axes[0].errorbar(np.arange(log_num_trial + 1), maj_values, 
               yerr=[maj_values - ci_lower, ci_upper - maj_values],
               fmt='none', ecolor=line_maj.get_color(), capsize=3, alpha=0.8)

          # Plot any_values without markers
          line_any, = axes[1].plot(np.arange(log_num_trial + 1), any_values, 
               label=f"{pres(method)}")
          # Add error bars for any_values
          ci_lower, ci_upper = confidence_interval(any_values * num_eval, num_eval)
          axes[1].errorbar(np.arange(log_num_trial + 1), any_values, 
               yerr=[any_values - ci_lower, ci_upper - any_values],
               fmt='none', ecolor=line_any.get_color(), capsize=3, alpha=0.8)

     
     # Add labels, title, and legend for maj_values plot
     axes[0].set_xlabel('Number of Sampled Outputs')
     axes[0].set_ylabel('Majority Decision Accuracy')
     axes[0].set_title('(a) Majority Decision Accuracy of Parallel Scaling CoT Strategies')
     #     axes[0].legend(loc='lower right')
     axes[0].grid(axis='y', linestyle='--', alpha=0.7)

     # Add labels, title, and legend for any_values plot
     analytical_accuracy = 1 - (1-2**depth/(3 * 4**(depth-1)))**(2**np.arange(log_num_trial + 1))          
     axes[1].plot(np.arange(log_num_trial + 1), analytical_accuracy,
          label=r"$P(\exists i \in [n] : \mathrm{DFS}_i \in D(\mathrm{Path}))$", color='green', linestyle='--')
     analytical_accuracy = 1 - (1-1/(3 * 4**(depth-1)))**(2**np.arange(log_num_trial + 1))     
     axes[1].plot(np.arange(log_num_trial + 1), analytical_accuracy,
          label=r"$P(\exists i \in [n] : \mathrm{DFS}_i \in D(\mathrm{S\text{-}Path}))$", color='red', linestyle='--')
     
     axes[1].set_xlabel('Number of Sampled Outputs')
     axes[1].set_ylabel('Any Evidence Accuracy')
     axes[1].set_title('(b) Any Evidence Accuracy of Parallel Scaling CoT Strategies')
     axes[1].legend()
     axes[1].grid(axis='y', linestyle='--', alpha=0.7)

     # Adjust layout
     plt.tight_layout()

     # Show the plot
     plt.savefig(f"figures/maj_any_{2 ** log_num_trial}_{sweep_name}.png", dpi=300)
     print(f"Saved figure to figures/maj_any_{2 ** log_num_trial}_{sweep_name}.png")

# Example usage
if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument('--log_num_trial', type=int, default=6)
     parser.add_argument('--temperature', type=float, default=1.0)
     parser.add_argument('--num_eval', type=int, default=5000)
     parser.add_argument('--depth', type=int, default=3)
     args = parser.parse_args()

     with open('eval/config/base.yaml', 'r') as f:
          config = yaml.safe_load(f)
     with open(f'eval/config/d{args.depth}.yaml', 'r') as f:
          config.update(yaml.safe_load(f))
     sweep_name = config['sweep_name']
     log_num_trial = args.log_num_trial
     temperature = args.temperature
     num_eval = args.num_eval

     compare_models(log_num_trial, temperature, num_eval, sweep_name, args.depth)