from functools import lru_cache
import numpy as np
import networkx as nx
import random
import os
from transformers import MistralForCausalLM
import glob  
import torch

# Graph Utils
def generate_caterpillar(depth, width, num_matchings):
    # number of nodes = (depth + 1) * width
    # number of edges = depth * width * num_matchings
    graph = nx.Graph()
    for i in range(depth):
        for _ in range(num_matchings):
            perm = np.random.permutation(width)
            for k in range(width):
                u = i * width + k
                v = (i + 1) * width + perm[k]
                graph.add_edge(u, v)    
    return graph


def gen_graph(args):
    graph_type = args.graph_type
    if args.num_nodes is not None:
        num_nodes = args.num_nodes
    elif args.max_num_nodes is not None:
        num_nodes = random.randint(args.min_num_nodes, args.max_num_nodes)
    
    if graph_type == 'cycle':
        graph = nx.cycle_graph(num_nodes)
        # TODO: set largest node to be the furthest
    elif graph_type == 'tree':
        graph = nx.random_labeled_tree(num_nodes)
    elif graph_type == 'component':
        num_edges = args.num_edges
        graph = nx.dense_gnm_random_graph(num_nodes, num_edges)
        while not nx.is_connected(graph):
            graph = nx.dense_gnm_random_graph(num_nodes, num_edges)
    elif graph_type == 'caterpillar':
        depth = args.depth
        width = args.width
        num_matchings = args.num_matchings
        graph = generate_caterpillar(depth, width, num_matchings)
        while not nx.is_connected(graph):
            graph = generate_caterpillar(depth, width, num_matchings)
    elif graph_type == 'comblock':
        depth = args.depth
        width = args.width
        graph = nx.Graph()
        for i in range(depth):
            for j in range(width):
                graph.add_edge(i * width, i * width + j + 1)
    elif graph_type == 'flower':
        depth = args.depth
        short, long, dead = args.short, args.long, args.dead
        # cycle = args.petal
        # ratio = args.ratio
        # assert cycle % (ratio + 1) == 0
        petal_size = short + long + dead
        petal_edges = []
        for i in range(short - 1):
            petal_edges.append((i, i + 1))
        for i in range(long - 1):
            petal_edges.append((short + i, (short + i + 1) % (short + long - 1)))
        for i in range(dead):
            petal_edges.append((short + long + i - 1, (short + long + i) % (short + long + dead - 1)))
        if short > 0:
            petal_edges.append((short - 1, petal_size - 1))
        if long > 0:
            petal_edges.append((short, petal_size - 1))
        
        graph = nx.Graph()
        for i in range(depth):
            for u, v in petal_edges:
                graph.add_edge(i * (petal_size - 1) + u, i * (petal_size - 1) + v)
    return graph


def dfs(graph, start, target=None, return_mistakes=True, return_backtrack=True):
    marked = [False] * graph.number_of_nodes()
    stack = [start]
    walk = []
    while len(stack) > 0:
        u = stack[-1]
        if not marked[u] or return_backtrack:
            walk.append(u)
        marked[u] = True
        if u == target:
            return walk if return_mistakes else stack
        candidates = [v for v in list(graph.neighbors(u)) if not marked[v]]
        if len(candidates) > 0:
            v = random.choice(candidates)
            stack.append(v)
        else:
            stack.pop()
    return walk


def bfs(graph, start, target=None):
    marked = [False] * graph.number_of_nodes()
    queue = [start]
    walk = []
    while len(queue) > 0:
        u = queue.pop(0)
        if marked[u]:
            continue
        walk.append(u)
        marked[u] = True
        if u == target:
            return walk
        candidates = [v for v in list(graph.neighbors(u)) if not marked[v]]
        queue.extend(candidates)
    return walk


graph_dict = {}

@lru_cache(maxsize=None)
def count_walk(graph_hash, start, target, walk_len):
    if start == target:
        target = -1
    if walk_len == 0:
        return 1 if (target == -1) else 0
    cnt = 0
    for u in graph_dict[graph_hash].neighbors(start):
        cnt += count_walk(graph_hash, u, target, walk_len - 1)
    return cnt


@lru_cache(maxsize=None)
def reach_prob(graph_hash, start, target, walk_len):
    if start == target:
        return 1
    if walk_len == 0:
        return 0
    prob = 0.
    neighbors = list(graph_dict[graph_hash].neighbors(start))
    for u in neighbors:
        prob += reach_prob(graph_hash, u, target, walk_len - 1)
    return prob / len(neighbors)


def gen_walk(graph, start, target, walk_len):
    graph_hash = hash(str(sorted(graph.edges)))
    if graph_hash not in graph_dict:
        graph_dict[graph_hash] = graph
    prob = reach_prob(graph_hash, start, target, walk_len)
    if np.isclose(prob, 0):
        raise ValueError('No walk exists')
    walk = [start]
    while walk[-1] != target:
        neighbors = list(graph.neighbors(walk[-1]))
        probs = np.array([reach_prob(graph_hash, u, target, walk_len - len(walk)) for u in neighbors])
        walk.append(random.choices(neighbors, probs)[0]) 
    return walk



# String Utils
def get_missions_vocab(num_node_tokens):
    vocab = {str(i): i for i in range(num_node_tokens)}
    keywords = ['graph', 'task', 'work', 'decision', 'to', 'or']
    puncs = [':', '[', '(', ',', ')', ']']
    special_tokens = ['UNK', 'PAD', '/', 'BOS', 'EOS']
    for idx, token in enumerate(keywords + puncs + special_tokens, start=num_node_tokens):
        vocab[token] = idx
    return vocab


def gen_mission_str(mission, **kwargs):
    node_ids = kwargs.get('node_ids')
    clues = kwargs.get('clues')
    num_graph_nodes = mission.graph.number_of_nodes()
    num_total_nodes = mission.number_of_nodes()
    edges = [(node_ids[u], node_ids[v]) for u, v in mission.graph.edges()]
    if mission.task_type == 'decision':
        edges += ([(node_ids[num_graph_nodes + u], node_ids[num_graph_nodes + v]) for u, v in mission.phrag.edges()])
    edges = [(u, v) if random.randint(0, 1) == 0 else (v, u) for u, v in edges]
    random.shuffle(edges)
    edges_str = f"[{' '.join([f'{u} {v}' for u, v in edges])}]"
    # edges_str = str(edges)
    mission_str = 'graph: ' + edges_str + '\ntask: ' + str(node_ids[mission.start]) + ' to '
    if mission.task_type == 'decision':
        t_1, t_2 = mission.target, mission.tegrat
        if random.randint(0, 1) == 0:
            t_1, t_2 = t_2, t_1
        mission_str += str(node_ids[t_1]) + ' or ' + str(node_ids[t_2])
    else:
        mission_str += str(node_ids[mission.target])
    mission_str += ' / '
    if clues:
        for clue_type, clue_list in clues.items():
            clue = [node_ids[node] for node in clue_list]
            clue_str = f"[{' '.join([str(node) for node in clue])}]"
            mission_str += f'\n{clue_type}: {clue_str}'
    return mission_str


def load_model(output_dir, sweep_id, model_name):
    model_dir = os.path.join(output_dir, f"sweeps/{sweep_id}", model_name)
    if os.path.exists(os.path.join(model_dir, 'best_model')):
        print(f"Found best model")
        model_dir = os.path.join(model_dir, 'best_model')
    else:
        if os.path.exists(model_dir):
            checkpoints = glob.glob(os.path.join(model_dir, 'checkpoint-*'))
            print(f"Found checkpoints: {checkpoints}")            
            if checkpoints:
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
                print(f"Loading {checkpoints[0]}")
                model_dir = checkpoints[0]
    print(f"Loading model from {model_dir}")
    if os.path.exists(model_dir):
        model = MistralForCausalLM.from_pretrained(model_dir)
        model.eval()
        if torch.cuda.is_available():
            print("Moving model to GPU")
            model.cuda()
        return model
    else:
        print(f"Model directory {model_dir} does not exist.")
        return None  