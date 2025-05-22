import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from analysis_utils import pres
from scipy.stats import norm




sweep_names = [
    "flower_d1-s3-l5-b3_t6e6_hl4",
    "flower_d2-s3-l5-b3_t8.5e6_hl4",
    "flower_d3-s3-l5-b3_t10e6_hl4",
    "flower_d4-s3-l5-b3_t11.5e6_hl4",
    "flower_d5-s3-l5-b3_t13e6_hl4"
]

strategies = ["dfs-pruned", "path", "optimal"]

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(8, 6))
plt.grid(axis='y', linestyle='--', alpha=0.7)
accuracy_threshold = 0.9

method_complexities = {}
depths = [1, 2, 3, 4, 5]

for i in depths:
    csv_path = f"csv/evals/{sweep_names[i-1]}_best_evidence_accuracy_learns_t1.0_n5000.csv"
    df = pd.read_csv(csv_path)  
    df = df[df['teach'].isin(strategies)]
    for i, row in df.iterrows():
        method = row['teach']
        offset = row['offset']
        if method not in strategies:
            continue
        if method not in method_complexities:
            method_complexities[method] = {}
        if offset not in method_complexities[method]:
            method_complexities[method][offset] = []
        method_complexities[method][offset].append(row[f'reach{accuracy_threshold}'])
        
# print(method_complexities['optimal'])


colors = {
    method: plt.cm.tab10(i)  # or any color palette
    for i, method in enumerate(method_complexities)
}

for method in strategies:
    print(method)
    color = colors[method]
    runs = [method_complexities[method][offset] for offset in range(5)]
    runs = np.array(runs)

    x = np.arange(runs.shape[1])
    
    # for r in runs:
    #     plt.plot(x, r, color=color, alpha=0.2)  # individual traces

    mean = runs.mean(axis=0)
    confidence = 0.95
    z = norm.ppf((1 + confidence) / 2)
    se = np.sqrt(mean * (mean - 1) / runs.shape[0])
    margin = z * se
    plt.errorbar(x, mean, yerr=[margin, margin] , label=pres(method))  # bold average

x = np.arange(len(depths))
# plt.yscale("log", base=2)
plt.xlabel("Depth of the Bridge Graph")
plt.ylabel("Number of Samples")
plt.title("Number of Samples Needed in Parallel Scaling")
plt.xticks(x, depths)
# set y_ticks to be 2^x
# y_ticks = [2**i for i in range(0,8)]
# plt.yticks(y_ticks, [f"{i}" for i in y_ticks])
plt.legend()
plt.tight_layout()
plt.savefig("figures/flower_d-s3-l5-b3_complexity.png", dpi=300)
print("Figure saved to figures/flower_d-s3-l5-b3_complexity.png")
plt.savefig("figures/flower_d-s3-l5-b3_complexity.pdf", dpi=300, format='pdf')
print("Figure saved to figures/flower_d-s3-l5-b3_complexity.pdf")