from functools import lru_cache
import numpy as np
import networkx as nx
import random

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
        cycle = args.petal
        ratio = args.ratio
        assert cycle % (ratio + 1) == 0
        cycle_sp = cycle // (ratio + 1)
        graph = nx.Graph()
        for i in range(depth):
            nodes = list(range(i * (cycle - 1), (i + 1) * (cycle - 1) + 1))
            nodes[:cycle_sp]  = reversed(nodes[:cycle_sp])
            for j in range(cycle):
                graph.add_edge(nodes[j], nodes[(j + 1) % cycle])
    return graph


def dfs(graph, start):
    marked = [False] * graph.number_of_nodes()
    stack = [start]
    walk = []
    while len(stack) > 0:
        u = stack[-1]
        walk.append(u)
        marked[u] = True
        candidates = [v for v in list(graph.neighbors(u)) if not marked[v]]
        if len(candidates) > 0:
            v = random.choice(candidates)
            stack.append(v)
        else:
            stack.pop()
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
    
    mission_str = 'graph: ' + str(edges) + '\ntask: ' + str(node_ids[mission.start]) + ' to '
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
            mission_str += f'\n{clue_type}: {clue}'
    return mission_str

