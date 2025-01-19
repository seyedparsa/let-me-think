import numpy as np
import networkx as nx
import random

# Graph Utils
def generate_caterpillar(num_nodes, num_parts, num_matchings):
    part_size = num_nodes // num_parts
    graph = nx.Graph()
    for i in range(num_parts - 1):
        for _ in range(num_matchings):
            perm = np.random.permutation(part_size)
            for k in range(part_size):
                u = i * part_size + k
                v = (i + 1) * part_size + perm[k]
                graph.add_edge(u, v)    
    return graph


def generate_graph(args):
    graph_type, num_nodes = args.graph_type, args.num_nodes
    if graph_type == 'cycle':
        graph = nx.cycle_graph(num_nodes)
    elif graph_type == 'tree':
        graph = nx.random_labeled_tree(num_nodes)
    elif graph_type == 'component':
        num_edges = args.num_edges
        graph = nx.dense_gnm_random_graph(num_nodes, num_edges)
        while not nx.is_connected(graph):
            graph = nx.dense_gnm_random_graph(num_nodes, num_edges)
    elif graph_type == 'caterpillar':
        num_parts = args.num_parts
        num_matchings = args.num_matchings
        graph = generate_caterpillar(num_nodes, num_parts, num_matchings)
        while not nx.is_connected(graph):
            print('not connected')
            print(graph.edges)
            graph = generate_caterpillar(num_nodes, num_parts, num_matchings)
        # print('connected')
        # print(graph.edges)
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


# String Utils
def get_missions_vocab(num_node_tokens):
    vocab = {str(i): i for i in range(num_node_tokens)}
    keywords = ['Graph', 'Task', 'Search', 'Path', 'Decision', 'to', 'or']
    puncs = [':', '[', '(', ',', ')', ']']
    special_tokens = ['UNK', 'PAD', '/', 'BOS', 'EOS']
    for idx, token in enumerate(keywords + puncs + special_tokens, start=num_node_tokens):
        vocab[token] = idx
    return vocab


def gen_mission_str(mission, node_id, clues):
    edges = [(node_id[u], node_id[v]) for u, v in mission.graph.edges()]
    if mission.task_type == 'decision':
        num_graph_nodes = mission.graph.number_of_nodes()
        edges.append([(node_id[num_graph_nodes + u], node_id[num_graph_nodes + v]) for u, v in mission.phrag.edges()])
    random.shuffle(edges)     
    s, t = node_id[mission.start], node_id[mission.target]   
    mission_str = 'Graph: ' + str(edges) + '\nTask: ' + str(s) + ' to '
    if mission.task_type == 'decision':
        t_1, t_2 = t, node_id[num_graph_nodes]
        if random.randint(0, 1) == 0:
            t_1, t_2 = t_2, t_1
        mission_str += str(t_1) + ' or ' + str(t_2)
    else:
        mission_str += str(t)
    if clues:
        for clue_type, clue_list in clues.items():
            clue = [node_id[node] for node in clue_list]
            mission_str += f' /\n{clue_type}: {clue}'
    return mission_str
