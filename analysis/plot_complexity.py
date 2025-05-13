import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from analysis_utils import pres



sweep_names = [
    "flower_d1-s3-l5-b3_t6e6_hl4",
    "flower_d2-s3-l5-b3_t8.5e6_hl4",
    "flower_d3-s3-l5-b3_t10e6_hl4",
    "flower_d4-s3-l5-b3_t11.5e6_hl4",
    "flower_d5-s3-l5-b3_t13e6_hl4"
]

# sweep_names = [
#     "delta_flower_d1-s3-l5-b3_t4e6",
#     "delta_flower_d2-s3-l5-b3_t8.5e6",
#     "delta_flower_d3-s3-l5-b3_t10e6",
#     "delta_flower_d4-s3-l5-b3_t11e6",
# ]

accuracy_threshold = 0.9

method_complexities = {}
depths = [1, 2, 3, 4, 5]

for i in depths:
    csv_path = f"csv/evals/{sweep_names[i-1]}_best_evidence_accuracy_learns_t1.0_n5000.csv"
    df = pd.read_csv(csv_path)  
    for i, row in df.iterrows():
        method = row['teach']
        offset = row['offset']
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

for method in method_complexities:
    color = colors[method]
    runs = [method_complexities[method][offset] for offset in range(5)]
    runs = np.array(runs)

    x = np.arange(runs.shape[1])
    
    for r in runs:
        plt.plot(x, r, color=color, alpha=0.2)  # individual traces

    mean = runs.mean(axis=0)
    plt.plot(x, mean, color=color, label=method, linewidth=1)  # bold average

# plt.yscale("log", base=2)
plt.xlabel("Depth of the Bridge graph")
plt.ylabel("Number of Sampled CoTs to reach 90% accuracy")
plt.title("Number of CoTs needed in Parallel Scaling of CoT Strategies")
# set y_ticks to be 2^x
# y_ticks = [2**i for i in range(0,8)]
# plt.yticks(y_ticks, [f"{i}" for i in y_ticks])
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figures/flower_d-s3-l5-b3_complexity.png", dpi=300)
print("Figure saved to figures/flower_d-s3-l5-b3_complexity.png")