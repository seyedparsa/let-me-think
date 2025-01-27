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
    parser.add_argument('--random_st', action='store_true', help='random start and target')

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)

    search_types = args.search_types.split(',')
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
        graph_name += f'd{args.depth}-w{args.width}'
        if args.num_matchings is not None:
            graph_name += f'-m{args.num_matchings}'
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
            for search_type in search_types:
                sample[search_type] = str(mission.search(search_type))
            data[split].append(sample)
    
        with open(f"{args.data_dir}/{split}_{graph_name}_{args.task_type}_n{num_samples}.json", 'w') as f:
            json.dump(data[split], f, indent=4)