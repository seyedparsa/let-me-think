import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from analysis_utils import pres

# order = ["dfs", "dfs-pruned", "walk-30", "walk-24", "walk-18", "path", "walk-12", "optimal"]

walks = ["walk-30", "walk-28", "walk-26", "walk-24", "walk-22", "walk-20", "walk-18", "walk-16", "walk-14", "walk-12", "walk-10", "walk-8"]
# walks = ["walk-11", "walk-9", "walk-7", "walk-5"]

others = ["dfs", "path", "optimal"]
order = walks + others


sweep_name = "delta_flower_d2-s3-l5-b3_t8.5e6"
# sweep_name = "delta_flower_d1-s3-l5-b3_t4e6"
num_eval = 5000

plt.figure(figsize=(8, 6))
plt.grid(linestyle='--', alpha=0.3)

csv_path = f"csv/evals/{sweep_name}_best_evidence_accuracy_gd_n{num_eval}.csv"
df = pd.read_csv(csv_path)
data_df = pd.read_csv(f"csv/avg_walk_lengths_{sweep_name}.csv")

df = df[df['teach'].isin(order)]
data_df = data_df[data_df['teach'].isin(order)]

df = df.merge(data_df, on='teach', how='left')


default_x_offset = -0.5  # horizontal offset
default_y_offset = -0.03  # vertical offset

for i, row in df.iterrows():
    method = row['teach']
    x_offset = default_x_offset 
    y_offset = default_y_offset
    # if row['teach'] == "walk-30":
    #     # x_offset = -2.5
    #     y_offset = 0.02
    # if row['teach'] == "walk-28":
    #     # x_offset = 3.5
    #     y_offset = 0.02
    if row['teach'] == "walk-22":
        y_offset = 0.02
    if row['teach'] == "walk-24":
        y_offset = 0.015
        x_offset = -1
    if method in others:
        plt.text(
            row['avg_walk_length'] + x_offset,
            row['decision_accuracy'] + y_offset,
            pres(row['teach']),
            fontsize=7,
            # ha='left',  # align left since we added positive x offset
            # va='bottom'
        )    
    
    # if row['teach'] == "walk-14":
    #     x_offset = 2
    #     y_offset = 0
    # if row['teach'] == "dfs":
    #     x_offset = 2
    # plt.text(
    #     row['avg_thoughts'] + x_offset,
    #     row['decision_accuracy'] + y_offset,
    #     pres(row['teach']),
    #     fontsize=7,
    #     # ha='left',  # align left since we added positive x offset
    #     # va='bottom'
    # )


print(df)
# plt.xscale('log')
for i, row in df.iterrows():
    plt.arrow(
        row['avg_walk_length'], row['decision_accuracy'],
        row['avg_verified_thoughts'] - row['avg_walk_length'], 0,
        head_width=0.01, head_length=0.2, fc='gray', ec='black', alpha=0.1, length_includes_head=True
    )
plt.scatter(df['avg_walk_length'], df['decision_accuracy'], label="Average Length of Training Thoughts")
plt.scatter(df['avg_verified_thoughts'], df['decision_accuracy'], label="Average Length of Verified Generated Thoughts", edgecolors='black', alpha=0.3)

plt.xlabel("Average Length of Thoughts")
plt.ylabel("Decision Accuracy")
plt.xticks(ticks=[i for i in range(6, 22, 2)], 
           labels=[f"{i}" for i in range(6, 22, 2)])
plt.minorticks_off()
plt.legend(loc='lower right')
plt.title("Average Length of Training and Verified Generated Thoughts vs Decision Accuracy")
plt.savefig(f"figures/avg_thoughts_{sweep_name}.png")