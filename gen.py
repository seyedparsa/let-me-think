import json
import argparse
import random
import os

from tqdm import tqdm
from mission import Mission
import networkx as nx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--data_dir', type=str, default='/work/hdd/bbjr/pmirtaheri/bepatient/data_dir', help='data directory')
    parser.add_argument('--num_train', type=int, default=5000, help='number of training samples')
    parser.add_argument('--num_val', type=int, default=1000, help='number of validation samples')
    parser.add_argument('--graph_type', type=str, default='caterpillar', help='type of graph')
    parser.add_argument('--task_type', type=str, default='decision', help='type of task')
    parser.add_argument('--search_type', type=str, default='dfs', help='type of search')
    parser.add_argument('--num_nodes', type=int, default=10, help='number of nodes')
    parser.add_argument('--num_parts', type=int, default=None, help='number of parts in caterpillar graph')
    parser.add_argument('--num_matchings', type=int, default=2, help='number of matchings in caterpillar graph')


    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)

    splits = ['train', 'val']
    split_sizes = [args.num_train, args.num_val]
    data = {}

    for split, num_samples in zip(splits, split_sizes):
        data[split] = []
        for i in tqdm(range(num_samples)):
            mission = Mission(args)
            optimal = nx.shortest_path(mission.graph, mission.start, mission.target)
            search = mission.search(args.search_type)
            sample = {
                'graph_type': mission.graph_type,
                'task_type': mission.task_type,                
                'graph': str(mission.graph.edges),
                'start': int(mission.start),
                'target': int(mission.target),
                'search_type': args.search_type,
                'search': str(search),
                'optimal': str(optimal),
            }
            if args.task_type == 'decision':
                sample['phrag'] = str(mission.phrag.edges)
            data[split].append(sample)
    
        with open(f'{args.data_dir}/{split}_{args.graph_type}_{args.task_type}_n{num_samples}.json', 'w') as f:
            json.dump(data[split], f, indent=4)