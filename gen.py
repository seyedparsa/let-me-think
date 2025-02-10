from collections import defaultdict
import json
import argparse
import random
import os

import numpy as np
from tqdm import tqdm
from mission import Mission
import matplotlib.pyplot as plt
import wandb
import networkx as nx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--data_dir', type=str, default='', help='data directory')
    parser.add_argument('--num_train', type=int, default=5000, help='number of training samples')
    parser.add_argument('--num_val', type=int, default=5000, help='number of validation samples')
    parser.add_argument('--graph_type', type=str, default='caterpillar', help='type of graph')
    parser.add_argument('--task_type', type=str, default='decision', help='type of task')
    parser.add_argument('--search_types', type=str, default='optimal', help='type of search')
    parser.add_argument('--num_nodes', type=int, default=None, help='number of nodes')
    parser.add_argument('--max_num_nodes', type=int, default=None, help='maximum number of nodes')
    parser.add_argument('--min_num_nodes', type=int, default=2, help='minimum number of nodes')
    parser.add_argument('--depth', type=int, default=None, help='depth in caterpillar and comblock graphs')
    parser.add_argument('--width', type=int, default=None, help='width in caterpillar and comblock graphs')
    parser.add_argument('--num_matchings', type=int, default=None, help='number of matchings in caterpillar graph')
    parser.add_argument('--st_pair', type=str, default='random', help='start and target choice')

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)

    wandb.init(project="bepatient", entity="seyedparsa", name=f"walk-lengths", resume="allow")

    search_types = []
    for search_type in args.search_types.split(','):
        if search_type.startswith('walk'):
            l, r, j = map(int, search_type.split('-')[1:])
            dist = 'walk' if search_type.startswith('walks-') else 'walk_mix'
            search_types.extend([f'{dist}-{i}' for i in range(l, r + 1, j)])
        else:
            search_types.append(search_type)
    print(search_types)
    splits = ['train', 'val']
    split_sizes = [args.num_train, args.num_val]
    data = {}

    graph_name = args.graph_type
    if args.num_nodes is not None:
        graph_name += str(args.num_nodes)
    elif args.max_num_nodes is not None:
        graph_name += f'{args.min_num_nodes}-{args.max_num_nodes}'
    elif args.depth is not None:
        graph_name += f'_d{args.depth}-w{args.width}'
        if args.num_matchings is not None:
            graph_name += f'-m{args.num_matchings}'
    walk_len = defaultdict(list)
    for split, num_samples in zip(splits, split_sizes):
        data[split] = []
        for i in tqdm(range(num_samples)):
            mission = Mission(args)
            sample = {
                'graph_type': mission.graph_type,
                'task_type': mission.task_type,                
                'graph': str(mission.graph.edges),
                'start': int(mission.start),
                'target': int(mission.target),
            }
            if args.task_type == 'decision':
                sample['phrag'] = str(mission.phrag.edges)
                sample['tegrat'] = int(mission.tegrat)
            for search_type in search_types:
                search = mission.search(search_type)
                sample[search_type] = str(search)
                # print(f"search {search_type}({len(search)-1}): {sample[search_type]}")                
                walk_len[search_type].append(len(search) - 1)
            data[split].append(sample)            
            # input()

        num_search_types = len(search_types)
        fig, axes = plt.subplots(num_search_types, 1, figsize=(10, 5 * num_search_types))
        for ax, search_type in zip(axes, search_types):
            ax.hist(walk_len[search_type], bins=range(0, 101), density=True, align='left')
            ax.set_xlabel('length')
            ax.set_ylabel(f'fraction of {search_type} traces')
            ax.set_xticks(range(0, 101, 5))
            ax.set_title(search_type)
        fig.suptitle(f'Histograms of search lengths in {split} set', fontsize=16)
        plt.tight_layout()
        wandb.log({f"{split}_search_types": wandb.Image(fig)})
        plt.close(fig)
        
        file_name = f"{split}_{graph_name}_{args.task_type}_st-{args.st_pair}_n{num_samples}.json"
        file_path = os.path.join(args.data_dir, file_name)
        with open(file_path, 'w') as f:
            json.dump(data[split], f, indent=4)