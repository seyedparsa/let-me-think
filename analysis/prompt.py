from datasets import load_dataset
from mission import Mission
from utils import gen_mission_str 
import numpy as np
from ast import literal_eval
import pandas as pd
from tqdm import tqdm
import os
import torch

seed = 42
torch.manual_seed(seed)   
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

sweep_name = "delta_flower_d3-s3-l5-b3_t10e6"
data_dir = "/work/hdd/bbjr/pmirtaheri/bepatient/data_dir/"
# data_file = "train_flower_d2-s3-l5-b3_decision_st-far_n500000.json"
data_file = "val_flower_d3-s3-l5-b3_decision_st-far_n5000.json"
dataset = load_dataset("json", data_files=os.path.join(data_dir, data_file), split="train")
# search_types = ["dfs", "dfs-pruned", "walk-30", "walk-28", "walk-26", "walk-24", "walk-22", "walk-20", "walk-18", "walk-16", "walk-14", "path", "walk-12", "walk-10", "walk-8", "optimal"]
search_types = ["dfs", "optimal", "path", "dfs-pruned"]

for i in range(0, 5):
    example = dataset[i]
    mission = Mission(example, from_dict=True)   
    node_ids = [15,2,57,23,7,3,38,55,58,4,27,37,44,46,41,12,33,43,31,19,29,54,18,11,32,16,30,26,17,6,1,5,60,49,24,34,47,9,42,61,48,14,56,51,52,35,59,25,20,36,45,0,22,10,21,40,39,13,28,50,53,8]
    directed = False
 
    for search_type in search_types:    
        work = literal_eval(example[search_type])
        clues = {'work': work, 'decision': [example['target']]}
        # node_ids = np.random.permutation(mission.number_of_nodes())
        mission_str = gen_mission_str(mission, node_ids=node_ids, clues=clues)
        print(f"Example {i} - Search Type: {search_type}")
        print(mission_str)
        print(work)

exit(0)
results = []

walk_lengths = {search_type : [] for search_type in search_types}
for example in dataset:
    for search_type in search_types:
        walk_lengths[search_type].append(len(literal_eval(example[search_type])))
for search_type in search_types:
    walk_lengths[search_type] = np.mean(walk_lengths[search_type])
    print(f"{search_type}: {walk_lengths[search_type]}")
    results.append([search_type, walk_lengths[search_type]])

df = pd.DataFrame(results, columns=["teach", "avg_walk_length"])
df.to_csv(f"csv/avg_walk_lengths_{sweep_name}.csv", index=False)


